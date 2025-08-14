# YOLO detection and tracking

## Overview
This project provides tools for object detection using YOLOv11 for recognizing defects and infrastructure elements including cracks, fire equipment, lights, and moisture. It includes scripts for dataset preparation, merging, and training.

## Dataset Structure
The project supports YOLO format datasets with the following structure:
```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## Installation

1. Create and activate a virtual environment:
```
python -m venv venv
.\venv\Scripts\activate
source venv/bin/activate
```

2. Install dependencies:
```
pip install -r requirements.txt
```

## Scripts

### Dataset Preparation

- **Convert Box Annotations**: Convert annotations to rectangular format
  ```
  python scripts/convert_box_annotation.py --dataset_dir cracksdataset --output_dataset_dir datasets
  ```

- **Split Dataset**: Split into train/test/val with 80/10/10 proportions
  ```
  python scripts/split_dataset.py
  ```

- **Create YOLO Annotations**: Generate annotations for defect classes
  ```
  python scripts/run_annotation.py
  # uses create_dataset_yaml.py and create_yolo_annotations.py
  ```

### Dataset Merging

- **Merge Three Datasets**:
  ```
  python scripts/merge_3_datasets.py --dataset1 datasets/cracks --dataset2 datasets/light --dataset3 datasets/fire --output datasets/general
  ```

- **Merge Four Datasets**:
  ```
  python scripts/merge_4_datasets.py --dataset1 datasets/general --dataset2 datasets/light5_split --dataset3 datasets/train-curat-dataset-yolo --dataset4 datasets/moisture --output datasets/final
  ```

- **Modify Dataset Labels**:
  ```
  python scripts/changing_labels.py --dataset_dir datasets/light --output datasets/light_modified
  ```

## Training

Train the YOLOv11 model:
```
python train.py --data datasets/light5_split/data.yaml --weights yolo11s.pt
```

Alternatively, use the ultralytics YOLO command:
```
yolo detect train model=yolo11s.pt data=datasets/endterm_final/data.yaml epochs=100 imgsz=640 batch=8 device=0
yolo detect train model=models/yolo11s.pt data=datasets/final/data.yaml epochs=100 imgsz=640 batch=8 device=0
```

yolo detect train model=weights/yolo11m.pt data=avtotime3/data.yaml epochs=100 imgsz=1280 batch=-1 device=0
yolo detect train model=weights/yolo11s.pt data=avtotime/data.yaml epochs=100 imgsz=1280 batch=-1 device=0

yolo detect train model=weights/yolo11m.pt data=avtotime3/data.yaml epochs=100 imgsz=640 batch=-1 device=0
## Real-time Detection and Tracking

### Object Detection

The project includes a real-time object detection script that uses YOLOv11 with webcam input:

```
python detection.py
```

This script loads a trained YOLOv11 model and processes video frames from your webcam, displaying detection results in real-time. Press 'q' to quit the detection window.

### Object Tracking

For applications requiring object tracking over time, use the tracking script:

```
python tracking.py
```

This script combines YOLOv11 detection with DeepSORT tracking to:
- Detect objects in each frame
- Assign consistent IDs to objects across frames
- Count unique objects seen
- Display tracking information in real-time

The tracking system is particularly useful for monitoring and counting defects or equipment during inspections.

## Model Files

- `yolo11s.pt`: YOLOv11 small model (for training)
- `yolo11m.pt`: YOLOv11 medium model (for training)





Create data yaml for dataset script


