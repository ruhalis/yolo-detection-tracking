#!/usr/bin/env python3
"""
Auto-annotation script for adding vehicle labels to license plate dataset.
Uses yolo11m.pt to detect vehicles and merges with existing license plate annotations.
"""

import os
import argparse
import shutil
from pathlib import Path
from ultralytics import YOLO
import cv2
import yaml
from tqdm import tqdm


def load_existing_annotations(label_path):
    """Load existing YOLO format annotations from a label file."""
    annotations = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) == 5:
                        class_id, x_center, y_center, width, height = map(float, parts)
                        annotations.append({
                            'class_id': int(class_id),
                            'x_center': x_center,
                            'y_center': y_center,
                            'width': width,
                            'height': height,
                            'confidence': 1.0  # Existing annotations have max confidence
                        })
    return annotations


def save_annotations(annotations, label_path):
    """Save annotations to YOLO format file."""
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    with open(label_path, 'w') as f:
        for ann in annotations:
            f.write(f"{ann['class_id']} {ann['x_center']:.6f} {ann['y_center']:.6f} "
                   f"{ann['width']:.6f} {ann['height']:.6f}\n")


def detect_vehicles_in_image(model, image_path, confidence_threshold, vehicle_classes):
    """Detect vehicles in a single image using YOLO11."""
    results = model(image_path, conf=confidence_threshold, verbose=False)
    
    vehicle_annotations = []
    if results and len(results) > 0:
        result = results[0]
        if result.boxes is not None:
            boxes = result.boxes
            for i in range(len(boxes)):
                class_id = int(boxes.cls[i])
                confidence = float(boxes.conf[i])
                
                # Check if detected class is a vehicle
                if class_id in vehicle_classes:
                    # Convert xyxy to xywh normalized format
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    img_width = result.orig_shape[1]
                    img_height = result.orig_shape[0]
                    
                    # Convert to YOLO format (normalized center coordinates)
                    x_center = (x1 + x2) / 2 / img_width
                    y_center = (y1 + y2) / 2 / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height
                    
                    vehicle_annotations.append({
                        'class_id': 1,  # Unified vehicle class
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height,
                        'confidence': confidence
                    })
    
    return vehicle_annotations


def process_dataset_split(model, input_dir, output_dir, split_name, confidence_threshold, vehicle_classes):
    """Process a dataset split (train/valid/test)."""
    input_images_dir = os.path.join(input_dir, split_name, 'images')
    input_labels_dir = os.path.join(input_dir, split_name, 'labels')
    output_images_dir = os.path.join(output_dir, split_name, 'images')
    output_labels_dir = os.path.join(output_dir, split_name, 'labels')
    
    if not os.path.exists(input_images_dir):
        print(f"Warning: {input_images_dir} not found. Skipping {split_name} split.")
        return
    
    # Create output directories
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(input_images_dir).glob(f'*{ext}'))
        image_files.extend(Path(input_images_dir).glob(f'*{ext.upper()}'))
    
    print(f"Processing {len(image_files)} images in {split_name} split...")
    
    stats = {
        'total_images': len(image_files),
        'images_with_vehicles': 0,
        'total_vehicles_detected': 0,
        'license_plates_preserved': 0
    }
    
    for image_file in tqdm(image_files, desc=f"Processing {split_name}"):
        image_name = image_file.name
        image_stem = image_file.stem
        
        # Copy image to output directory
        shutil.copy2(image_file, os.path.join(output_images_dir, image_name))
        
        # Load existing license plate annotations
        input_label_path = os.path.join(input_labels_dir, f"{image_stem}.txt")
        existing_annotations = load_existing_annotations(input_label_path)
        stats['license_plates_preserved'] += len(existing_annotations)
        
        # Detect vehicles in the image
        vehicle_annotations = detect_vehicles_in_image(
            model, str(image_file), confidence_threshold, vehicle_classes
        )
        
        if vehicle_annotations:
            stats['images_with_vehicles'] += 1
            stats['total_vehicles_detected'] += len(vehicle_annotations)
        
        # Combine existing and new annotations
        all_annotations = existing_annotations + vehicle_annotations
        
        # Save combined annotations
        output_label_path = os.path.join(output_labels_dir, f"{image_stem}.txt")
        save_annotations(all_annotations, output_label_path)
    
    return stats


def create_updated_data_yaml(input_yaml_path, output_yaml_path):
    """Create updated data.yaml file with both license plate and vehicle classes."""
    # Load existing data.yaml
    with open(input_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Update paths to be relative to the new dataset location
    data['train'] = '../train/images'
    data['val'] = '../valid/images'
    data['test'] = '../test/images'
    
    # Update classes
    data['nc'] = 2
    data['names'] = ['License_Plate', 'Vehicle']
    
    # Save updated data.yaml
    with open(output_yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def main():
    parser = argparse.ArgumentParser(description='Auto-annotate vehicles in license plate dataset')
    parser.add_argument('--dataset', required=True, help='Path to input dataset directory')
    parser.add_argument('--output', required=True, help='Path to output dataset directory')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold for vehicle detection')
    parser.add_argument('--weights', default='weights/yolo11m.pt', help='Path to YOLO11 weights')
    parser.add_argument('--vehicle-classes', nargs='*', type=int, 
                       default=[1, 2, 3, 5, 7], 
                       help='COCO class IDs for vehicles (default: bicycle=1, car=2, motorcycle=3, bus=5, truck=7)')
    
    args = parser.parse_args()
    
    # Validate input dataset
    if not os.path.exists(args.dataset):
        print(f"Error: Input dataset directory '{args.dataset}' not found.")
        return
    
    # Load YOLO11 model
    print(f"Loading YOLO11 model from {args.weights}...")
    model = YOLO(args.weights)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Process each dataset split
    total_stats = {
        'total_images': 0,
        'images_with_vehicles': 0,
        'total_vehicles_detected': 0,
        'license_plates_preserved': 0
    }
    
    for split in ['train', 'valid', 'test']:
        split_stats = process_dataset_split(
            model, args.dataset, args.output, split, 
            args.confidence, set(args.vehicle_classes)
        )
        if split_stats:
            for key in total_stats:
                total_stats[key] += split_stats[key]
            
            print(f"\n{split.upper()} Split Statistics:")
            print(f"  - Total images: {split_stats['total_images']}")
            print(f"  - Images with vehicles: {split_stats['images_with_vehicles']}")
            print(f"  - Total vehicles detected: {split_stats['total_vehicles_detected']}")
            print(f"  - License plates preserved: {split_stats['license_plates_preserved']}")
    
    # Create updated data.yaml
    input_yaml = os.path.join(args.dataset, 'data.yaml')
    output_yaml = os.path.join(args.output, 'data.yaml')
    if os.path.exists(input_yaml):
        create_updated_data_yaml(input_yaml, output_yaml)
        print(f"\nUpdated data.yaml created at: {output_yaml}")
    
    # Print final statistics
    print(f"\n{'='*50}")
    print("FINAL STATISTICS")
    print(f"{'='*50}")
    print(f"Total images processed: {total_stats['total_images']}")
    print(f"Images with vehicles detected: {total_stats['images_with_vehicles']}")
    print(f"Total vehicles detected: {total_stats['total_vehicles_detected']}")
    print(f"License plates preserved: {total_stats['license_plates_preserved']}")
    print(f"Detection rate: {total_stats['images_with_vehicles']/total_stats['total_images']*100:.1f}%")
    print(f"Average vehicles per image: {total_stats['total_vehicles_detected']/total_stats['total_images']:.2f}")
    
    print(f"\nEnhanced dataset created at: {args.output}")
    print("Dataset is ready for training with both license plates and vehicles!")


if __name__ == '__main__':
    main()