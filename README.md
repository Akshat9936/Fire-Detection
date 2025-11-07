# Fire Detection using Image Processing

This repository presents a **hybrid, vision-based fire detection system** built using **YOLOv8**, classical image-processing techniques, and temporal analysis for flicker and motion. The project demonstrates how modern deep learning can be combined with lightweight computer vision modules to achieve **real-time and accurate indoor fire detection**.

---

##  Project Overview

The system takes live or recorded video input and detects **fire** and **smoke** using a hybrid framework that fuses:

* **YOLOv8** for deep-learning–based fire/smoke object detection
* **HSV color segmentation** to isolate fire-colored pixels (orange–yellow hues)
* **Motion detection** to ensure detected regions are dynamic
* **Flicker analysis** to validate temporal irregularities typical of flames
* **Circularity filtering** to reject bright, static objects (e.g., bulbs, sunlight reflections)

Once a fire event is confirmed across multiple frames, a **threaded alarm** is triggered to provide an instant alert while ensuring smooth video playback.

---

##  Repository Structure

```
 Fire Detection Repository
 ┣  Fire-Detection
 ┃ ┗ Code implementation and detection logic (main hybrid model)
 ┣  Sample_videos
 ┃ ┗ Example videos to test the fire detection system
 ┣  UNISA_Dataset
 ┃ ┗ Partial dataset used for model evaluation (due to size constraints)
 ┣  Dataset_evaluation.py
 ┃ ┗ Python script to evaluate model performance on the dataset
 ┣  model.py
 ┃ ┗ Main program file integrating YOLOv8 and hybrid confirmation logic
 ┣  README.md
 ┗ (Other supporting files)
```

---

##  Usage

1. Clone this repository:

   ```bash
   git clone https://github.com/Akshat9936/Fire-Detection.git
   cd Fire-Detection
   ```

2. Run the main model:

   ```bash
   python model.py
   ```

3. To test with your own videos:

   * Place the video inside the `Sample_videos/` folder.
   * Update the `VIDEO_PATH` variable in the code to point to your file.

4. To evaluate the model on the UNISA dataset:

   ```bash
   python Dataset_evaluation.py
   ```

---

##  UNISA Fire Detection Dataset

The **UNISA (MIVIA/UNISA Fire Detection Dataset)** is a widely used benchmark designed for evaluating vision-based fire detection systems.
It contains **31 videos**, including:

* 14 confirmed fire sequences recorded under different indoor conditions, and
* 17 non-fire videos (e.g., reflections, colored lights, moving red/orange objects) specifically created to challenge false-alarm handling.

This dataset allows performance assessment in terms of **true positives, false positives, and overall detection accuracy**.

>  **Note:** Due to GitHub’s upload size limitations, only a subset of the dataset is provided here for reference.
> For complete testing and reproducibility, it is **strongly recommended** to download the full dataset from its original source below:

 **Original UNISA Fire Detection Dataset:**
[https://mivia.unisa.it/datasets/video-analysis-datasets/fire-detection-dataset/](https://mivia.unisa.it/datasets/video-analysis-datasets/fire-detection-dataset/)

---

##  Evaluation Script

The `Dataset_evaluation.py` program can be used to evaluate detection performance on the dataset.
It compares predicted bounding boxes and temporal detections across frames to compute:

* True Positive Rate (TPR)
* False Alarm Rate (FAR)
* Frame-wise accuracy
* Detection latency per sequence

This ensures objective benchmarking of the proposed model against the dataset’s challenging conditions.

---

##  Requirements

Install dependencies before running:

```bash
pip install opencv-python numpy ultralytics torch
```

Optional (for Windows alarm functionality):

```bash
pip install winsound
```

---

##  Key Highlights

* **Real-time inference**: ~24 FPS on standard CPU
* **High accuracy**: ~90% overall detection accuracy on mixed indoor test videos
* **Low false alarm rate** through hybrid fusion logic
* **Modular design** for easy customization and dataset evaluation

---

##  Conclusion

This project demonstrates an efficient and practical approach to **real-time indoor fire detection** using video analytics.
It integrates the reliability of deep learning with explainable classical methods, achieving **robust, fast, and cost-effective fire safety automation**.

---

