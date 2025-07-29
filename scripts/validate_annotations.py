#!/usr/bin/env python3
"""
Validation script for reviewing auto-annotated license plate + vehicle dataset.
Provides visualization and statistics for quality control.
"""

import os
import argparse
import random
from pathlib import Path
import cv2
import yaml
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter


def load_annotations(label_path):
    """Load YOLO format annotations from a label file."""
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
                            'height': height
                        })
    return annotations


def draw_bounding_boxes(image, annotations, class_names, colors):
    """Draw bounding boxes on an image."""
    height, width = image.shape[:2]
    
    for ann in annotations:
        class_id = ann['class_id']
        if class_id >= len(class_names):
            continue
            
        # Convert YOLO format to pixel coordinates
        x_center = ann['x_center'] * width
        y_center = ann['y_center'] * height
        box_width = ann['width'] * width
        box_height = ann['height'] * height
        
        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)
        
        # Draw bounding box
        color = colors[class_id]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = class_names[class_id]
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return image


def generate_statistics(dataset_dir, class_names):
    """Generate comprehensive statistics for the dataset."""
    stats = {
        'splits': {},
        'overall': {
            'total_images': 0,
            'total_annotations': 0,
            'class_distribution': Counter(),
            'images_per_class': Counter(),
            'bbox_sizes': defaultdict(list)
        }
    }
    
    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(split_dir):
            continue
            
        images_dir = os.path.join(split_dir, 'images')
        labels_dir = os.path.join(split_dir, 'labels')
        
        if not os.path.exists(images_dir):
            continue
        
        split_stats = {
            'images': 0,
            'annotations': 0,
            'class_distribution': Counter(),
            'images_per_class': Counter(),
            'bbox_sizes': defaultdict(list)
        }
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(images_dir).glob(f'*{ext}'))
            image_files.extend(Path(images_dir).glob(f'*{ext.upper()}'))
        
        split_stats['images'] = len(image_files)
        
        for image_file in image_files:
            label_file = os.path.join(labels_dir, f"{image_file.stem}.txt")
            annotations = load_annotations(label_file)
            
            split_stats['annotations'] += len(annotations)
            
            image_classes = set()
            for ann in annotations:
                class_id = ann['class_id']
                split_stats['class_distribution'][class_id] += 1
                image_classes.add(class_id)
                
                # Calculate bbox area
                area = ann['width'] * ann['height']
                split_stats['bbox_sizes'][class_id].append(area)
            
            # Count images per class
            for class_id in image_classes:
                split_stats['images_per_class'][class_id] += 1
        
        stats['splits'][split] = split_stats
        
        # Update overall stats
        stats['overall']['total_images'] += split_stats['images']
        stats['overall']['total_annotations'] += split_stats['annotations']
        stats['overall']['class_distribution'].update(split_stats['class_distribution'])
        stats['overall']['images_per_class'].update(split_stats['images_per_class'])
        
        for class_id, sizes in split_stats['bbox_sizes'].items():
            stats['overall']['bbox_sizes'][class_id].extend(sizes)
    
    return stats


def print_statistics(stats, class_names):
    """Print formatted statistics."""
    print("="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    # Overall statistics
    overall = stats['overall']
    print(f"Total Images: {overall['total_images']}")
    print(f"Total Annotations: {overall['total_annotations']}")
    print(f"Average Annotations per Image: {overall['total_annotations']/overall['total_images']:.2f}")
    
    print("\nClass Distribution:")
    for class_id, count in overall['class_distribution'].items():
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
        percentage = (count / overall['total_annotations']) * 100
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    print("\nImages per Class:")
    for class_id, count in overall['images_per_class'].items():
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
        percentage = (count / overall['total_images']) * 100
        print(f"  {class_name}: {count} images ({percentage:.1f}%)")
    
    # Split-wise statistics
    print("\nSplit-wise Breakdown:")
    for split_name, split_stats in stats['splits'].items():
        print(f"\n{split_name.upper()}:")
        print(f"  Images: {split_stats['images']}")
        print(f"  Annotations: {split_stats['annotations']}")
        print(f"  Avg per image: {split_stats['annotations']/split_stats['images']:.2f}")
        
        for class_id, count in split_stats['class_distribution'].items():
            class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
            print(f"    {class_name}: {count}")


def create_visualization_samples(dataset_dir, output_dir, class_names, num_samples=20):
    """Create visualization samples for manual review."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Colors for different classes
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    
    all_images = []
    for split in ['train', 'valid', 'test']:
        images_dir = os.path.join(dataset_dir, split, 'images')
        labels_dir = os.path.join(dataset_dir, split, 'labels')
        
        if not os.path.exists(images_dir):
            continue
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        for ext in image_extensions:
            for image_file in Path(images_dir).glob(f'*{ext}'):
                label_file = os.path.join(labels_dir, f"{image_file.stem}.txt")
                if os.path.exists(label_file):
                    all_images.append((str(image_file), label_file, split))
            for image_file in Path(images_dir).glob(f'*{ext.upper()}'):
                label_file = os.path.join(labels_dir, f"{image_file.stem}.txt")
                if os.path.exists(label_file):
                    all_images.append((str(image_file), label_file, split))
    
    # Sample random images
    sample_images = random.sample(all_images, min(num_samples, len(all_images)))
    
    print(f"\nCreating {len(sample_images)} visualization samples...")
    
    for i, (image_path, label_path, split) in enumerate(sample_images):
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        # Load annotations
        annotations = load_annotations(label_path)
        
        # Draw bounding boxes
        image_with_boxes = draw_bounding_boxes(image, annotations, class_names, colors)
        
        # Save visualization
        output_filename = f"sample_{i+1:02d}_{split}_{Path(image_path).stem}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, image_with_boxes)
    
    print(f"Visualization samples saved to: {output_dir}")


def create_analysis_plots(stats, class_names, output_dir):
    """Create analysis plots for the dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Class distribution pie chart
    plt.figure(figsize=(10, 8))
    class_counts = []
    class_labels = []
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
    
    for class_id, count in stats['overall']['class_distribution'].items():
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
        class_counts.append(count)
        class_labels.append(f"{class_name}\n({count})")
    
    plt.pie(class_counts, labels=class_labels, autopct='%1.1f%%', colors=colors[:len(class_counts)])
    plt.title('Class Distribution in Dataset')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Bounding box size distribution
    plt.figure(figsize=(12, 8))
    for class_id, sizes in stats['overall']['bbox_sizes'].items():
        if sizes:
            class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
            plt.hist(sizes, bins=50, alpha=0.7, label=f"{class_name} (avg: {np.mean(sizes):.4f})")
    
    plt.xlabel('Bounding Box Area (normalized)')
    plt.ylabel('Count')
    plt.title('Bounding Box Size Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bbox_size_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Analysis plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Validate and analyze annotated dataset')
    parser.add_argument('--dataset', required=True, help='Path to dataset directory')
    parser.add_argument('--output', default='validation_output', help='Output directory for validation results')
    parser.add_argument('--samples', type=int, default=20, help='Number of visualization samples to create')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization generation')
    parser.add_argument('--no-plots', action='store_true', help='Skip analysis plots')
    
    args = parser.parse_args()
    
    # Load dataset configuration
    data_yaml_path = os.path.join(args.dataset, 'data.yaml')
    if os.path.exists(data_yaml_path):
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        class_names = data_config.get('names', ['License_Plate', 'Vehicle'])
    else:
        class_names = ['License_Plate', 'Vehicle']  # Default class names
        print("Warning: data.yaml not found, using default class names")
    
    print(f"Dataset: {args.dataset}")
    print(f"Classes: {class_names}")
    
    # Generate statistics
    stats = generate_statistics(args.dataset, class_names)
    print_statistics(stats, class_names)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Save statistics to file
    with open(os.path.join(args.output, 'statistics.txt'), 'w') as f:
        f.write("DATASET VALIDATION REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Dataset Path: {args.dataset}\n")
        f.write(f"Classes: {class_names}\n\n")
        
        overall = stats['overall']
        f.write(f"Total Images: {overall['total_images']}\n")
        f.write(f"Total Annotations: {overall['total_annotations']}\n")
        f.write(f"Average Annotations per Image: {overall['total_annotations']/overall['total_images']:.2f}\n\n")
        
        f.write("Class Distribution:\n")
        for class_id, count in overall['class_distribution'].items():
            class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
            percentage = (count / overall['total_annotations']) * 100
            f.write(f"  {class_name}: {count} ({percentage:.1f}%)\n")
    
    # Create visualizations
    if not args.no_viz:
        viz_dir = os.path.join(args.output, 'visualizations')
        create_visualization_samples(args.dataset, viz_dir, class_names, args.samples)
    
    # Create analysis plots
    if not args.no_plots:
        plots_dir = os.path.join(args.output, 'plots')
        create_analysis_plots(stats, class_names, plots_dir)
    
    print(f"\nValidation complete! Results saved to: {args.output}")
    print("\nReview the visualization samples to ensure annotation quality.")


if __name__ == '__main__':
    main()