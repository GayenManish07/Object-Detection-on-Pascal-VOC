# Task 1: Efficient Object Detection from Scratch

This repository contains the implementation for **Task 1** of the Computer Vision assignment. The goal was to build, train, and analyze a lightweight object detector (Faster R-CNN with MobileNetV2 backbone) trained entirely from scratch on a subset of the PASCAL VOC 2012 dataset.

## 📌 Project Overview
- **Model Architecture:** Faster R-CNN (Region-based Convolutional Neural Network).
- **Backbone:** MobileNetV2 (chosen for efficiency over ResNet).
- **Initialization:** Random weights (Trained from scratch, `weights=None`).
- **Dataset:** PASCAL VOC 2012 (Subset of 5 classes: Aeroplane, Bicycle, Car, Person, Dog).
- **Framework:** PyTorch.

## 🛠️ Training Methodology
To find the optimal balance between performance and efficiency, we conducted a comprehensive ablation study involving **9 different configurations**.

### 1. Model Scaling
We experimented with the `width_mult` parameter of MobileNetV2 to control the model capacity:
- **0.5x:** Highly compressed, targeting maximum FPS.
- **0.75x:** Balanced.
- **1.0x:** Standard MobileNetV2, targeting maximum Accuracy.

### 2. Augmentation Strategies
We implemented three levels of data augmentation to combat overfitting (since we trained from scratch):
- **None:** Only standard normalization.
- **Basic:** Random Horizontal Flip and Photometric Distortion (Color Jitter).
- **Mixup:** A robust strategy where two images are blended ($I_{mix} = \lambda I_1 + (1-\lambda)I_2$) and their bounding boxes are concatenated.

## Experimental Results & Trade-off Analysis
We trained all 9 configurations for 50 epochs on a consistent data subset to compare their metrics.

### Comparative Results Table

| Model Width | Augmentation | mAP (IoU=0.50) | mAP (IoU=0.5:0.95) | Inference FPS | Model Size (MB) | Train Time (m) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0.5** | basic | 0.5146 | 0.2196 | **56.16** | 308.37 | 216.3 |
| 0.5 | mixup | 0.5783 | 0.2640 | 47.03 | 308.37 | 216.3 |
| 0.5 | none | 0.5597 | 0.2560 | 47.20 | 308.37 | 216.3 |
| **0.75** | basic | 0.5461 | 0.2337 | 49.74 | 310.92 | 238.0 |
| 0.75 | mixup | 0.5765 | 0.2717 | 44.08 | 310.92 | 238.0 |
| 0.75 | none | 0.5573 | 0.2554 | 42.26 | 310.92 | 238.0 |
| **1.0** | basic | 0.5834 | 0.2590 | 47.12 | 314.23 | 247.7 |
| **1.0** | **mixup** | **0.6095** | **0.2949** | 41.14 | 314.23 | 247.7 |
| 1.0 | none | 0.5556 | 0.2454 | 41.47 | 314.23 | 247.7 |

### Analysis of Trade-offs
1.  **Best Accuracy:** The **1.0x Width + Mixup** configuration achieved the highest mAP_50 (0.6095). Mixup proved highly effective for training from scratch, likely forcing the model to learn more robust features than simple flipping.
2.  **Best Speed:** The **0.5x Width** models consistently achieved higher FPS (up to 56 FPS). However, this came at a significant cost to accuracy (~9% drop in mAP compared to the best model).
3.  **The "Sweet Spot":** The **0.5x + Mixup** model offers an interesting compromise, achieving respectable accuracy (0.5783 mAP) while maintaining high inference speed.

## How to Reproduce

### 1. Requirements
Ensure you have the required libraries installed:
```bash
pip install torch torchvision opencv-python pandas torchmetrics
