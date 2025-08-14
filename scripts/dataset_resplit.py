#!/usr/bin/env python3
import os
import shutil
import random
import yaml
from pathlib import Path
from collections import defaultdict

# Set random seed for reproducibility
random.seed(42)

# Source and destination paths
SOURCE_DIR = "datasets/garage_aug"
DEST_DIR = "datasets/garage_aug_split"

# Split ratios (80/10/10)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

def create_directory_structure():
    """Create the destination directory structure"""
    for split in ["train", "val", "test"]:
        for subdir in ["images", "labels"]:
            os.makedirs(os.path.join(DEST_DIR, split, subdir), exist_ok=True)
    print(f"Created directory structure in {DEST_DIR}")

def get_image_label_pairs():
    """Get all image and label pairs from the source directory"""
    # Combine images from both train and valid folders
    image_label_pairs = []
    
    # Process train folder
    train_images_dir = os.path.join(SOURCE_DIR, "train", "images")
    train_labels_dir = os.path.join(SOURCE_DIR, "train", "labels")
    
    if os.path.exists(train_images_dir) and os.path.exists(train_labels_dir):
        for image_file in os.listdir(train_images_dir):
            if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            label_file = os.path.splitext(image_file)[0] + ".txt"
            if os.path.exists(os.path.join(train_labels_dir, label_file)):
                image_label_pairs.append({
                    "image_path": os.path.join(train_images_dir, image_file),
                    "label_path": os.path.join(train_labels_dir, label_file),
                    "basename": os.path.splitext(image_file)[0]
                })
    
    # Process valid folder
    valid_images_dir = os.path.join(SOURCE_DIR, "valid", "images")
    valid_labels_dir = os.path.join(SOURCE_DIR, "valid", "labels")
    
    if os.path.exists(valid_images_dir) and os.path.exists(valid_labels_dir):
        for image_file in os.listdir(valid_images_dir):
            if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            label_file = os.path.splitext(image_file)[0] + ".txt"
            if os.path.exists(os.path.join(valid_labels_dir, label_file)):
                image_label_pairs.append({
                    "image_path": os.path.join(valid_images_dir, image_file),
                    "label_path": os.path.join(valid_labels_dir, label_file),
                    "basename": os.path.splitext(image_file)[0]
                })
    
    return image_label_pairs

def split_dataset(image_label_pairs):
    """Split the dataset according to the defined ratios"""
    # Shuffle the dataset
    random.shuffle(image_label_pairs)
    
    # Calculate split indices
    total_samples = len(image_label_pairs)
    train_end = int(total_samples * TRAIN_RATIO)
    val_end = train_end + int(total_samples * VAL_RATIO)
    
    # Split the dataset
    train_set = image_label_pairs[:train_end]
    val_set = image_label_pairs[train_end:val_end]
    test_set = image_label_pairs[val_end:]
    
    return {
        "train": train_set,
        "val": val_set,
        "test": test_set
    }

def copy_files(splits):
    """Copy files to their respective destinations"""
    counts = defaultdict(int)
    
    for split_name, items in splits.items():
        for item in items:
            # Copy image
            image_src = item["image_path"]
            image_dest = os.path.join(DEST_DIR, split_name, "images", os.path.basename(image_src))
            shutil.copy2(image_src, image_dest)
            
            # Copy label
            label_src = item["label_path"]
            label_dest = os.path.join(DEST_DIR, split_name, "labels", os.path.basename(label_src))
            shutil.copy2(label_src, label_dest)
            
            counts[split_name] += 1
    
    return counts

def create_yaml_file(counts):
    """Create a data.yaml file for the new dataset"""
    yaml_path = os.path.join(DEST_DIR, "data.yaml")
    
    # Read the original yaml to get class names
    orig_yaml_path = os.path.join(SOURCE_DIR, "data.yaml")
    class_names = []
    class_count = 0
    
    if os.path.exists(orig_yaml_path):
        with open(orig_yaml_path, 'r') as f:
            orig_data = yaml.safe_load(f)
            if 'names' in orig_data:
                class_names = orig_data['names']
            if 'nc' in orig_data:
                class_count = orig_data['nc']
    
    # Create the new yaml content
    yaml_content = {
        'path': os.path.abspath(DEST_DIR),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': class_count,
        'names': class_names
    }
    
    # Write to file
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"Created YAML configuration file: {yaml_path}")

def main():
    """Main function to execute the dataset splitting process"""
    print(f"Starting dataset split process (80/10/10) from {SOURCE_DIR} to {DEST_DIR}")
    
    # Create directory structure
    create_directory_structure()
    
    # Get all image-label pairs from the source directories
    image_label_pairs = get_image_label_pairs()
    total_pairs = len(image_label_pairs)
    
    if total_pairs == 0:
        print("Error: No valid image-label pairs found in the source directory")
        return
    
    print(f"Found {total_pairs} image-label pairs")
    
    # Split the dataset
    splits = split_dataset(image_label_pairs)
    
    # Copy files to their destinations
    counts = copy_files(splits)
    
    # Create YAML configuration file
    create_yaml_file(counts)
    
    # Print summary
    print(f"\nDataset split complete!")
    print(f"Training set: {counts['train']} images ({counts['train']/total_pairs*100:.1f}%)")
    print(f"Validation set: {counts['val']} images ({counts['val']/total_pairs*100:.1f}%)")
    print(f"Test set: {counts['test']} images ({counts['test']/total_pairs*100:.1f}%)")
    print(f"\nNew dataset is ready at: {DEST_DIR}")

if __name__ == "__main__":
    main()