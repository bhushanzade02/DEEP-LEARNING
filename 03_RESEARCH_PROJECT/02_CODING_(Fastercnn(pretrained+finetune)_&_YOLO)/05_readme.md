# Comparative Study of CNN-Based Object Detectors and YOLO for Real-Time Object Detection

##  Project Overview
This project presents a **comparative analysis of CNN-based object detection models and YOLO (You Only Look Once)** with a focus on **accuracy vs real-time performance**.  
The study evaluates **Faster R-CNN** (two-stage detector) and **YOLOv5** (one-stage detector) using the **Pascal VOC 2007 dataset**.

The objective is to understand:
- How traditional CNN-based detectors perform compared to modern YOLO models
- Trade-offs between detection accuracy and inference speed
- Suitability of each model for real-time applications

---

## Objectives
- Implement **Faster R-CNN** using PyTorch
- Train **YOLOv5** on Pascal VOC dataset
- Evaluate models using standard object detection metrics
- Compare inference speed (FPS) and accuracy
- Provide qualitative and quantitative result analysis

---

##  Models Implemented
### 1️ Faster R-CNN
- Two-stage object detector
- Uses Region Proposal Network (RPN)
- High detection accuracy
- Slower inference speed

### 2️ YOLOv5 (Core Model)
- One-stage object detector
- End-to-end regression-based detection
- Real-time performance
- Competitive accuracy

---

##  Dataset
**Pascal VOC 2007**
- 20 object classes
- 9,963 images
- XML annotations (VOC format)

Classes include:
aeroplane, bicycle, bird, boat, bottle, bus, car, cat,
chair, cow, diningtable, dog, horse, motorbike, person,
pottedplant, sheep, sofa, train, tvmonitor


---

##  Tech Stack
- **Language:** Python 3
- **Framework:** PyTorch
- **YOLO Version:** YOLOv5
- **Platform:** Kaggle / Google Colab
- **GPU:** NVIDIA Tesla T4
- **Libraries:** torchvision, numpy, matplotlib, OpenCV

---

## Project Structure


cnn-vs-yolo/
│
├── VOCdevkit/
│ └── VOC2007/
│ ├── Annotations/
│ ├── JPEGImages/
│ └── ImageSets/
│
├── VOC_YOLO/
│ ├── images/
│ │ ├── train/
│ │ └── val/
│ ├── labels/
│ │ ├── train/
│ │ └── val/
│
├── yolov5/
│ ├── train.py
│ ├── val.py
│ ├── runs/
│ │ └── train/exp/
│ │ ├── results.png
│ │ ├── confusion_matrix.png
│ │ └── weights/
│ │ ├── best.pt
│ │ └── last.pt
│
├── voc.yaml
├── README.md
└── requirements.txt


---

## Experimental Setup
- Image Size: `640 × 640`
- Batch Size: `16`
- Epochs: `30`
- Optimizer: SGD
- Loss Functions:
  - Bounding box loss
  - Objectness loss
  - Classification loss

---

##  Evaluation Metrics
The following standard object detection metrics are used:

- **Precision**
- **Recall**
- **mAP@0.5**
- **mAP@0.5:0.95**
- **Inference Time**
- **Frames Per Second (FPS)**

---

##  Results Summary

### 🔹 YOLOv5 Performance
- **mAP@0.5:** 0.717
- **Precision:** 0.768
- **Recall:** 0.648
- **Inference Time:** 0.045 sec/image
- **FPS:** 22.15

### 🔹 Faster R-CNN Performance
- **Inference Time:** 0.136 sec/image
- **FPS:** 7.35
- Higher accuracy but slower inference

---

##  Visual Results
The project includes:
- Training & validation loss curves
- Precision–Recall curves
- Confusion matrix
- Qualitative detection outputs on validation images

These are available in:
yolov5/runs/train/exp/


---

##  Key Observations
- YOLOv5 significantly outperforms Faster R-CNN in real-time inference
- Faster R-CNN performs better on small or overlapping objects
- YOLOv5 is more suitable for real-time applications such as surveillance and autonomous systems

---

## Conclusion
The study demonstrates that **YOLOv5 provides an optimal balance between accuracy and speed**, making it the preferred choice for real-time object detection tasks.  
Faster R-CNN remains useful in applications where accuracy is prioritized over speed.

---

##  Future Work
- Train on larger datasets (COCO)
- Compare with YOLOv8
- Model pruning and quantization
- Edge-device deployment (Jetson, Raspberry Pi)

---

##  References
- J. Redmon et al., "You Only Look Once", CVPR
- Girshick et al., "Faster R-CNN", IEEE TPAMI
- Pascal VOC Challenge

---

##  Author
**Bhushan Zade**  
M.Sc. Scientific Computing  
Savitribai Phule Pune University  

---

##  Acknowledgment
This project was completed as part of a **Machine Learning / Deep Learning semester project** under academic guidance.

