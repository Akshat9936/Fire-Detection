import cv2
import numpy as np
import threading
import winsound
import time
import os
from collections import deque, defaultdict
import statistics

from ultralytics import YOLO
import torch

# =========================
# CONFIGURATION
# =========================
VIDEO_PATH       = "fire2.mp4"              # <--- set your input video here
MODEL_PATH       = "best.pt"                # your trained YOLOv8 fire/smoke model
ALLOWED_CLASSES  = {"fire", "smoke"}        # set {} to allow all, else filter by names

CONF_THRESHOLD   = 0.70                     # YOLO confidence threshold (tune 0.30–0.50)
IOU_THRESHOLD    = 0.45                     # NMS IoU
CONFIRMATION_FRAMES = 5                     # temporal confirmation (video-level)

# ---- SPEED KNOBS ----
INFER_WIDTH      = 640                      # 512–768 good range
FRAME_STRIDE     = 2                        # process every Nth frame with YOLO
USE_FP16         = True                     # GPU fp16 (ignored on CPU)
SHOW_WINDOW      = True                     # turn off for max speed
RESIZE_TO        = (960, 540)               # display size (None to keep original)
SLOW_PREVIEW     = 1                        # cv2.waitKey ms

# ---- CLASSIC (HSV + MOTION + FLICKER) AUXILIARY KNOBS ----
# Keep your original thresholds by default
FIRE_AREA_THRESHOLD = 1500
BRIGHTNESS_THRESHOLD = 180

# Motion (per-frame)
MOTION_DIFF_THRESHOLD = 18

# Flicker tracking (per-box)
FLICKER_HISTORY = 8
FLICKER_STD_THRESHOLD = 150.0   # absolute pixel std for box mask area (works well with 960x540)
# If you prefer % flicker instead of absolute, set USE_NORMALIZED_FLICKER=True
USE_NORMALIZED_FLICKER = False
FLICKER_PCT_STD_THRESHOLD = 0.035  # 3.5% (only used if USE_NORMALIZED_FLICKER=True)

# Sun/bulb guard (useful if fallback fires on bright disks)
ENABLE_CIRCULARITY_REJECT = True
CIRCULARITY_MAX = 0.85  # >0.85 ~ near circle

# Alarm (Windows)
ALARM_DURATION = 500
ALARM_FREQUENCY = 2000

def play_alarm():
    try:
        winsound.Beep(ALARM_FREQUENCY, ALARM_DURATION)
    except Exception:
        pass

# ============== Utility ==============
def letterbox_resize(image, new_w):
    """Resize keeping aspect ratio by width; returns resized image and (sx, sy) scales."""
    h, w = image.shape[:2]
    scale = new_w / float(w)
    nh = int(round(h * scale))
    resized = cv2.resize(image, (new_w, nh), interpolation=cv2.INTER_LINEAR)
    return resized, scale, scale  # same sx, sy since proportional

def build_fire_mask_hsv(frame_bgr):
    """Your original HSV fire mask with noise reduction."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower_fire = np.array([5, 80, BRIGHTNESS_THRESHOLD], dtype=np.uint8)
    upper_fire = np.array([35, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_fire, upper_fire)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    mask = cv2.dilate(mask, None, iterations=2)
    return mask

def motion_mask_from(prev_gray, gray):
    diff = cv2.absdiff(gray, prev_gray)
    _, motion_raw = cv2.threshold(diff, MOTION_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    motion_raw = cv2.GaussianBlur(motion_raw, (5, 5), 0)
    motion_raw = cv2.dilate(motion_raw, None, iterations=1)
    return motion_raw

def circularity_of(contour):
    peri = cv2.arcLength(contour, True)
    if peri == 0:
        return 0.0
    area = cv2.contourArea(contour)
    return 4.0 * np.pi * (area / (peri * peri))

# ============== Hybrid per-frame decision ==============
class BoxFlickerState:
    """Track per-box flicker using mask area history inside the box."""
    def __init__(self, maxlen=8):
        self.hist = deque(maxlen=maxlen)

    def push(self, value):
        self.hist.append(float(value))

    def ok(self):
        if len(self.hist) < self.hist.maxlen:
            return False
        try:
            if USE_NORMALIZED_FLICKER:
                # treat entries as ratios [0..1]
                s = statistics.pstdev(self.hist)
                return s >= FLICKER_PCT_STD_THRESHOLD
            else:
                s = statistics.pstdev(self.hist)
                return s >= FLICKER_STD_THRESHOLD
        except statistics.StatisticsError:
            return False

def yolo_plus_classic_decision(
    frame, detections, motion_mask_global, hsv_mask, box_states
):
    """
    Given YOLO detections (list of boxes) and global motion + hsv mask:
    - Confirm YOLO boxes with (motion OR flicker) inside the box.
    - If YOLO has no boxes, use classic fallback (HSV + motion + sun-guard).
    Returns: confirmed_boxes (list), frame_fire_detected (bool)
    """
    H, W = frame.shape[:2]
    confirmed = []

    # If YOLO found something, verify with classic cues inside each box
    if detections:
        for (x1, y1, x2, y2, conf, name) in detections:
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(W-1, x2); y2 = min(H-1, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            # HSV-only area inside box
            roi_mask = hsv_mask[y1:y2, x1:x2]
            area = int(cv2.countNonZero(roi_mask))

            # Small blobs are unreliable unless very confident
            if area <= FIRE_AREA_THRESHOLD and conf < 0.6:
                continue

            # Motion overlap inside box
            roi_motion = motion_mask_global[y1:y2, x1:x2]
            moving = cv2.countNonZero(cv2.bitwise_and(roi_mask, roi_motion)) > 0

            # Per-box flicker history
            key = (x1//16, y1//16, x2//16, y2//16)  # coarse key to reduce jitter
            state = box_states[key]
            if USE_NORMALIZED_FLICKER:
                state.push(area / float((x2 - x1) * (y2 - y1)))
            else:
                state.push(area)

            flicker_ok = state.ok()

            # Pass if dynamic (motion or flicker) OR the YOLO conf is high
            if moving or flicker_ok or conf >= 0.6:
                confirmed.append((x1, y1, x2, y2, conf, name))

    # If none confirmed and YOLO empty/missed → classic fallback on whole frame
    if not confirmed and not detections:
        contours, _ = cv2.findContours(hsv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            a = cv2.contourArea(c)
            if a <= FIRE_AREA_THRESHOLD:
                continue
            x, y, w, h = cv2.boundingRect(c)
            # motion inside this contour bbox
            roi = hsv_mask[y:y+h, x:x+w]
            roi_motion = motion_mask_global[y:y+h, x:x+w]
            moving = cv2.countNonZero(cv2.bitwise_and(roi, roi_motion)) > 0

            # optional circularity reject (sun/bulb)
            if ENABLE_CIRCULARITY_REJECT:
                circ = circularity_of(c)
                if circ > CIRCULARITY_MAX and not moving:
                    continue

            if moving or a > (5 * FIRE_AREA_THRESHOLD):
                confirmed.append((x, y, x+w, y+h, 0.50, "fire_fallback"))
                break

    return confirmed, (len(confirmed) > 0)

# ============== Single-video runner ==============
def run_video_hybrid():
    # Load model (GPU if available)
    if not os.path.isfile(MODEL_PATH):
        print(f"\n🛑 ERROR: Model not found at: {MODEL_PATH}")
        return

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(MODEL_PATH).to(device)
    try:
        model.fuse()
    except Exception:
        pass

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Cannot open video file: {VIDEO_PATH}")
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    # Warmup (important for CUDA kernels & fp16 graphs)
    dummy = np.zeros((H, W, 3), dtype=np.uint8)
    model.predict(source=dummy, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, device=model.device,
                  imgsz=INFER_WIDTH, verbose=False, half=(USE_FP16 and torch.cuda.is_available()))

    prev_gray = None
    box_states = defaultdict(lambda: BoxFlickerState(maxlen=FLICKER_HISTORY))

    frame_id = 0
    last_boxes = []
    alarm_triggered = False
    fire_confirmation_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1

        # Global precomputations (cheap)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            prev_gray = gray.copy()
            motion_mask_global = np.zeros_like(gray)
        else:
            motion_mask_global = motion_mask_from(prev_gray, gray)
            prev_gray = gray.copy()

        hsv_mask = build_fire_mask_hsv(frame)

        do_infer = (frame_id % FRAME_STRIDE == 1)

        detections = []
        if do_infer:
            # YOLO inference on downscaled frame
            infer_img, sx, sy = letterbox_resize(frame, INFER_WIDTH)
            results = model.predict(
                source=infer_img,
                conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, imgsz=INFER_WIDTH,
                device=model.device, verbose=False,
                half=(USE_FP16 and torch.cuda.is_available())
            )

            if results:
                r = results[0]
                names = r.names
                if r.boxes is not None and len(r.boxes) > 0:
                    xyxy = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    clss = r.boxes.cls.cpu().numpy().astype(int)
                    for (x1, y1, x2, y2), c, cl in zip(xyxy, confs, clss):
                        name = names.get(cl, str(cl))
                        if ALLOWED_CLASSES and (name.lower() not in {n.lower() for n in ALLOWED_CLASSES}):
                            continue
                        # map back to original coords
                        X1 = int(round(x1 / sx)); Y1 = int(round(y1 / sy))
                        X2 = int(round(x2 / sx)); Y2 = int(round(y2 / sy))
                        detections.append((X1, Y1, X2, Y2, float(c), name))

            # Hybrid confirmation
            confirmed_boxes, frame_fire_detected = yolo_plus_classic_decision(
                frame, detections, motion_mask_global, hsv_mask, box_states
            )
            last_boxes = confirmed_boxes

            # video-level temporal confirmation
            if frame_fire_detected:
                fire_confirmation_count += 1
                if fire_confirmation_count >= CONFIRMATION_FRAMES:
                    alarm_triggered = True
            else:
                fire_confirmation_count = 0
                alarm_triggered = False
        # else: reuse last_boxes & alarm state on skipped frames

        # ---- Draw / Alert ----
        if SHOW_WINDOW:
            disp = frame if RESIZE_TO is None else cv2.resize(frame, RESIZE_TO)
            scale_x = (RESIZE_TO[0] / W) if RESIZE_TO else 1.0
            scale_y = (RESIZE_TO[1] / H) if RESIZE_TO else 1.0

            for (x1, y1, x2, y2, c, name) in last_boxes:
                dx1, dy1 = int(x1 * scale_x), int(y1 * scale_y)
                dx2, dy2 = int(x2 * scale_x), int(y2 * scale_y)
                cv2.rectangle(disp, (dx1, dy1), (dx2, dy2), (0, 0, 255), 2)
                label = f"{name} {c:.2f}" if isinstance(c, float) else str(name)
                cv2.putText(disp, label, (dx1, max(20, dy1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if alarm_triggered:
                cv2.putText(disp, "FIRE/SMOKE DETECTED!", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
                threading.Thread(target=play_alarm, daemon=True).start()

            cv2.imshow("YOLO Fire (Hybrid)", disp)
            if cv2.waitKey(SLOW_PREVIEW) & 0xFF == ord('q'):
                break

    cap.release()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()

    print("\n=== Final Decision ===")
    print("ALARM TRIGGERED " if alarm_triggered else "No Fire Detected ")

if __name__ == "__main__":
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    run_video_hybrid()
