# Faster R-CNN for Aerial Object Detection

## Overview

This project implements an end-to-end **Faster R-CNN** based object detection pipeline for detecting aerial objects in high-resolution satellite images. The model is trained and evaluated on the **DOTA v1.0** dataset and detects **15 different object classes** using a custom training and evaluation pipeline built with PyTorch.

---

## Dataset

This project uses the **DOTA v1.0** dataset.

- **Training Set:** DOTA v1.0 Training
- **Validation Set:** DOTA v1.0 Validation

Official Website:  
https://captain-whu.github.io/DOTA/

Dataset Download:  
https://captain-whu.github.io/DOTA/dataset.html

> **Note:** The original DOTA annotations are provided as oriented bounding boxes (OBB). For compatibility with Faster R-CNN, they were converted into axis-aligned bounding boxes (AABB).

---

## Features

- Custom DOTA dataset loader
- Faster R-CNN with ResNet-50 FPN backbone
- Data augmentation
- Model training and fine-tuning
- Evaluation using Precision, Recall, F1-Score, and mAP@0.5
- Per-class detection statistics
- Training loss visualization

---

## Technologies Used

- Python
- PyTorch
- TorchVision
- OpenCV
- NumPy
- Matplotlib

---

## Project Structure

```
FasterRCNN-OBB-DOTA
│
├── train_frcnn.py
├── detection_dataset_frcnn.py
├── validation_dataset.py
├── evaluate_frcnn.py
├── evaluate_image.py
├── evaluation_utils.py
├── map.py
├── per_class_stats.py
├── count_classes.py
├── plot_loss.py
├── config.py
└── README.md
```

---

## How to Run

### Train the Model

```bash
python train_frcnn.py
```

### Evaluate the Model

```bash
python evaluate_frcnn.py
```

### Compute mAP

```bash
python map.py
```

### Generate Per-Class Statistics

```bash
python per_class_stats.py
```

### Plot Training Loss

```bash
python plot_loss.py
```

---

## Results

| Metric | Value |
|---------|-------|
| mAP@0.5 | **0.7507** |
| Precision | **0.7507** |
| Recall | **0.4408** |
| F1-Score | **0.5555** |

---

## Future Improvements

- Support oriented bounding boxes (OBB)
- Improve detection of small objects
- Experiment with different backbones and anchor configurations

---

## Author

**Sakshi S. Jangi**  
B.Tech, Electronics and Communication Engineering  
National Institute of Technology Warangal
