#!/usr/bin/env python3
"""
Dataset Augmentation Script for YOLOv11
Generates augmented versions of YOLO datasets with proper annotation handling.

Creates 6 regular augmented copies + 2 grayscale augmented copies of each image.
Augmentations: noise, brightness, exposure, blur, saturation, rotation, horizontal flip
"""

import os
import cv2
import numpy as np
import random
import shutil
import yaml
from pathlib import Path
import argparse
from typing import List, Tuple, Dict
import logging
from PIL import Image, ImageEnhance
import math

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetAugmentor:
    def __init__(self, source_dataset: str, output_dataset: str, seed: int = 42):
        """
        Initialize the dataset augmentor.
        
        Args:
            source_dataset: Path to source YOLO dataset directory
            output_dataset: Path to output augmented dataset directory
            seed: Random seed for reproducible results
        """
        self.source_dataset = Path(source_dataset)
        self.output_dataset = Path(output_dataset)
        self.seed = seed
        
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        
        # Augmentation parameters
        self.aug_params = {
            'noise_std': (0.1, 0.5),           # Gaussian noise standard deviation range
            'brightness_factor': (0.8, 1.2), # Brightness adjustment range
            'exposure_gamma': (0.8, 1.2),   # Gamma correction range
            'blur_kernel': (3, 7),          # Blur kernel size range (odd numbers)
            'saturation_factor': (0.7, 1.3), # Saturation adjustment range
            'rotation_angle': (-15, 15),    # Rotation angle range in degrees
        }
        
        # Augmentation types
        self.non_geometric_augs = ['noise', 'brightness', 'exposure', 'blur', 'saturation']
        self.geometric_augs = ['rotation', 'hflip']
        self.color_augs = ['grayscale']
        
        # Combination parameters
        self.combination_params = {
            'min_augs': 2,              # Minimum augmentations per copy
            'max_augs': 4,              # Maximum augmentations per copy
            'rotation_prob': 0.3,       # Probability of including rotation
            'flip_prob': 0.5,           # Probability of including horizontal flip
            'grayscale_prob': 0.2,      # Probability of grayscale for regular copies
        }
    
    def add_noise(self, image: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to image."""
        std = random.uniform(*self.aug_params['noise_std'])
        noise = np.random.normal(0, std, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image, noise)
        return np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    def adjust_brightness(self, image: np.ndarray) -> np.ndarray:
        """Adjust image brightness."""
        factor = random.uniform(*self.aug_params['brightness_factor'])
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Brightness(pil_image)
        bright_image = enhancer.enhance(factor)
        return cv2.cvtColor(np.array(bright_image), cv2.COLOR_RGB2BGR)
    
    def adjust_exposure(self, image: np.ndarray) -> np.ndarray:
        """Adjust image exposure using gamma correction."""
        gamma = random.uniform(*self.aug_params['exposure_gamma'])
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    def apply_blur(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur to image."""
        kernel_size = random.choice(range(self.aug_params['blur_kernel'][0], 
                                        self.aug_params['blur_kernel'][1] + 1, 2))
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def adjust_saturation(self, image: np.ndarray) -> np.ndarray:
        """Adjust image saturation."""
        factor = random.uniform(*self.aug_params['saturation_factor'])
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Color(pil_image)
        sat_image = enhancer.enhance(factor)
        return cv2.cvtColor(np.array(sat_image), cv2.COLOR_RGB2BGR)
    
    def rotate_image(self, image: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rotate image and return transformation matrix.
        
        Returns:
            Tuple of (rotated_image, transformation_matrix)
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new image dimensions
        cos_angle = np.abs(M[0, 0])
        sin_angle = np.abs(M[0, 1])
        new_w = int((h * sin_angle) + (w * cos_angle))
        new_h = int((h * cos_angle) + (w * sin_angle))
        
        # Adjust transformation matrix for new center
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        # Apply rotation
        rotated = cv2.warpAffine(image, M, (new_w, new_h), borderValue=(0, 0, 0))
        
        return rotated, M
    
    def flip_horizontal(self, image: np.ndarray) -> np.ndarray:
        """Flip image horizontally."""
        return cv2.flip(image, 1)
    
    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale and back to 3-channel."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    def transform_bbox_rotation(self, bbox: List[float], M: np.ndarray, 
                              orig_w: int, orig_h: int, new_w: int, new_h: int) -> List[float]:
        """
        Transform bounding box coordinates for rotation.
        
        Args:
            bbox: YOLO format bbox [class, x_center, y_center, width, height] (normalized)
            M: Transformation matrix from rotation
            orig_w, orig_h: Original image dimensions
            new_w, new_h: New image dimensions after rotation
        
        Returns:
            Transformed bbox in YOLO format
        """
        class_id, x_center, y_center, width, height = bbox
        
        # Convert normalized coordinates to absolute
        x_center_abs = x_center * orig_w
        y_center_abs = y_center * orig_h
        width_abs = width * orig_w
        height_abs = height * orig_h
        
        # Get corner points of bounding box
        x1 = x_center_abs - width_abs / 2
        y1 = y_center_abs - height_abs / 2
        x2 = x_center_abs + width_abs / 2
        y2 = y_center_abs + height_abs / 2
        
        corners = np.array([
            [x1, y1, 1],
            [x2, y1, 1],
            [x2, y2, 1],
            [x1, y2, 1]
        ]).T
        
        # Transform corners
        transformed_corners = M @ corners
        
        # Get new bounding box
        x_coords = transformed_corners[0, :]
        y_coords = transformed_corners[1, :]
        
        new_x1 = np.min(x_coords)
        new_y1 = np.min(y_coords)
        new_x2 = np.max(x_coords)
        new_y2 = np.max(y_coords)
        
        # Convert back to YOLO format (normalized)
        new_x_center = (new_x1 + new_x2) / 2 / new_w
        new_y_center = (new_y1 + new_y2) / 2 / new_h
        new_width = (new_x2 - new_x1) / new_w
        new_height = (new_y2 - new_y1) / new_h
        
        # Clip to [0, 1] range
        new_x_center = np.clip(new_x_center, 0, 1)
        new_y_center = np.clip(new_y_center, 0, 1)
        new_width = np.clip(new_width, 0, 1)
        new_height = np.clip(new_height, 0, 1)
        
        return [class_id, new_x_center, new_y_center, new_width, new_height]
    
    def transform_bbox_flip(self, bbox: List[float]) -> List[float]:
        """
        Transform bounding box coordinates for horizontal flip.
        
        Args:
            bbox: YOLO format bbox [class, x_center, y_center, width, height] (normalized)
        
        Returns:
            Transformed bbox in YOLO format
        """
        class_id, x_center, y_center, width, height = bbox
        # For horizontal flip, x_center becomes (1 - x_center)
        new_x_center = 1.0 - x_center
        return [class_id, new_x_center, y_center, width, height]
    
    def read_yolo_annotation(self, annotation_path: Path) -> List[List[float]]:
        """Read YOLO format annotation file."""
        if not annotation_path.exists():
            return []
        
        annotations = []
        with open(annotation_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    class_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    annotations.append([class_id] + coords)
        return annotations
    
    def write_yolo_annotation(self, annotation_path: Path, annotations: List[List[float]]):
        """Write YOLO format annotation file."""
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        with open(annotation_path, 'w') as f:
            for ann in annotations:
                class_id = int(ann[0])
                coords = ann[1:]
                f.write(f"{class_id} {' '.join([f'{x:.6f}' for x in coords])}\n")
    
    def apply_augmentation(self, image: np.ndarray, aug_type: str, 
                          annotations: List[List[float]], orig_w: int, orig_h: int) -> Tuple[np.ndarray, List[List[float]]]:
        """
        Apply specified augmentation to image and annotations.
        
        Returns:
            Tuple of (augmented_image, transformed_annotations)
        """
        if aug_type == 'noise':
            return self.add_noise(image), annotations
        elif aug_type == 'brightness':
            return self.adjust_brightness(image), annotations
        elif aug_type == 'exposure':
            return self.adjust_exposure(image), annotations
        elif aug_type == 'blur':
            return self.apply_blur(image), annotations
        elif aug_type == 'saturation':
            return self.adjust_saturation(image), annotations
        elif aug_type == 'rotation':
            angle = random.uniform(*self.aug_params['rotation_angle'])
            rotated_img, M = self.rotate_image(image, angle)
            new_h, new_w = rotated_img.shape[:2]
            
            # Transform annotations
            transformed_annotations = []
            for ann in annotations:
                new_ann = self.transform_bbox_rotation(ann, M, orig_w, orig_h, new_w, new_h)
                # Check if bbox is still valid (has positive area and is within image)
                if new_ann[3] > 0 and new_ann[4] > 0:
                    transformed_annotations.append(new_ann)
            
            return rotated_img, transformed_annotations
        elif aug_type == 'hflip':
            flipped_img = self.flip_horizontal(image)
            # Transform annotations
            transformed_annotations = []
            for ann in annotations:
                new_ann = self.transform_bbox_flip(ann)
                transformed_annotations.append(new_ann)
            return flipped_img, transformed_annotations
        elif aug_type == 'grayscale':
            return self.convert_to_grayscale(image), annotations
        else:
            raise ValueError(f"Unknown augmentation type: {aug_type}")
    
    def generate_random_augmentation_combo(self, copy_index: int, force_grayscale: bool = False, 
                                         force_hflip: bool = False) -> List[str]:
        """
        Generate a random combination of augmentations for a single copy.
        
        Args:
            copy_index: Index of the augmented copy (for seeding)
            force_grayscale: Force inclusion of grayscale
            force_hflip: Force inclusion of horizontal flip
            
        Returns:
            List of augmentation types to apply in order
        """
        # Set unique seed for this copy
        temp_seed = self.seed + copy_index * 100
        random.seed(temp_seed)
        np.random.seed(temp_seed)
        
        selected_augs = []
        
        # Always select some non-geometric augmentations
        num_non_geo = random.randint(self.combination_params['min_augs'], 
                                   min(self.combination_params['max_augs'], len(self.non_geometric_augs)))
        selected_non_geo = random.sample(self.non_geometric_augs, num_non_geo)
        selected_augs.extend(selected_non_geo)
        
        # Optionally add rotation
        if not force_hflip and random.random() < self.combination_params['rotation_prob']:
            selected_augs.append('rotation')
        
        # Add horizontal flip if forced or by probability
        if force_hflip or (not force_hflip and random.random() < self.combination_params['flip_prob']):
            selected_augs.append('hflip')
        
        # Add grayscale if forced or by probability
        if force_grayscale or (not force_grayscale and random.random() < self.combination_params['grayscale_prob']):
            selected_augs.append('grayscale')
        
        # Reset to original seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        return selected_augs
    
    def apply_random_augmentations(self, image: np.ndarray, augmentation_combo: List[str], 
                                 annotations: List[List[float]], orig_w: int, orig_h: int) -> Tuple[np.ndarray, List[List[float]]]:
        """
        Apply multiple augmentations in sequence.
        
        Args:
            image: Input image
            augmentation_combo: List of augmentation types to apply
            annotations: Original annotations
            orig_w, orig_h: Original image dimensions
            
        Returns:
            Tuple of (augmented_image, transformed_annotations)
        """
        current_image = image.copy()
        current_annotations = [ann.copy() for ann in annotations]
        current_w, current_h = orig_w, orig_h
        
        # Apply non-geometric augmentations first
        for aug_type in augmentation_combo:
            if aug_type in self.non_geometric_augs or aug_type == 'grayscale':
                current_image, current_annotations = self.apply_augmentation(
                    current_image, aug_type, current_annotations, current_w, current_h
                )
        
        # Apply geometric augmentations last (they change image dimensions)
        for aug_type in augmentation_combo:
            if aug_type in self.geometric_augs:
                current_image, current_annotations = self.apply_augmentation(
                    current_image, aug_type, current_annotations, current_w, current_h
                )
                # Update dimensions if rotation was applied
                if aug_type == 'rotation':
                    current_h, current_w = current_image.shape[:2]
        
        return current_image, current_annotations
    
    def generate_augmented_filename(self, base_name: str, augmentation_combo: List[str], copy_index: int) -> str:
        """
        Generate descriptive filename for augmented image.
        
        Args:
            base_name: Original image base name
            augmentation_combo: List of applied augmentations
            copy_index: Index of the augmented copy
            
        Returns:
            Augmented filename
        """
        # Create short abbreviations for augmentations
        aug_abbrev = {
            'noise': 'n',
            'brightness': 'b',
            'exposure': 'e', 
            'blur': 'bl',
            'saturation': 's',
            'rotation': 'r',
            'hflip': 'hf',
            'grayscale': 'g'
        }
        
        # Create abbreviated augmentation string
        aug_string = '_'.join([aug_abbrev.get(aug, aug[:2]) for aug in augmentation_combo])
        
        return f"{base_name}_aug{copy_index:02d}_{aug_string}.jpg"
    
    def augment_single_image(self, image_path: Path, annotation_path: Path, 
                           output_images_dir: Path, output_labels_dir: Path):
        """Augment a single image and its annotations with random combinations."""
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning(f"Could not read image: {image_path}")
            return
        
        orig_h, orig_w = image.shape[:2]
        
        # Read annotations
        annotations = self.read_yolo_annotation(annotation_path)
        
        # Get base filename without extension
        base_name = image_path.stem
        
        # Generate 6 regular augmented copies with random combinations
        for copy_idx in range(6):
            try:
                # Generate random augmentation combination
                aug_combo = self.generate_random_augmentation_combo(copy_idx)
                
                # Apply the combination
                aug_image, aug_annotations = self.apply_random_augmentations(
                    image, aug_combo, annotations, orig_w, orig_h
                )
                
                # Generate filename
                aug_image_name = self.generate_augmented_filename(base_name, aug_combo, copy_idx)
                aug_image_path = output_images_dir / aug_image_name
                cv2.imwrite(str(aug_image_path), aug_image)
                
                # Save augmented annotations
                aug_label_name = aug_image_name.replace('.jpg', '.txt')
                aug_label_path = output_labels_dir / aug_label_name
                self.write_yolo_annotation(aug_label_path, aug_annotations)
                
            except Exception as e:
                logger.error(f"Error applying random augmentations (copy {copy_idx}) to {image_path}: {e}")
        
        # Generate 1 horizontal flip copy with random augmentations
        try:
            # Force horizontal flip in this copy
            aug_combo = self.generate_random_augmentation_combo(6, force_hflip=True)
            
            # Apply the combination
            aug_image, aug_annotations = self.apply_random_augmentations(
                image, aug_combo, annotations, orig_w, orig_h
            )
            
            # Generate filename
            aug_image_name = self.generate_augmented_filename(base_name, aug_combo, 6)
            aug_image_path = output_images_dir / aug_image_name
            cv2.imwrite(str(aug_image_path), aug_image)
            
            # Save augmented annotations
            aug_label_name = aug_image_name.replace('.jpg', '.txt')
            aug_label_path = output_labels_dir / aug_label_name
            self.write_yolo_annotation(aug_label_path, aug_annotations)
            
        except Exception as e:
            logger.error(f"Error applying horizontal flip augmentations to {image_path}: {e}")
        
        # Generate 2 grayscale copies with random augmentations
        for gray_idx in range(2):
            try:
                copy_idx = 7 + gray_idx
                # Force grayscale in these copies
                aug_combo = self.generate_random_augmentation_combo(copy_idx, force_grayscale=True)
                
                # Apply the combination
                aug_image, aug_annotations = self.apply_random_augmentations(
                    image, aug_combo, annotations, orig_w, orig_h
                )
                
                # Generate filename
                aug_image_name = self.generate_augmented_filename(base_name, aug_combo, copy_idx)
                aug_image_path = output_images_dir / aug_image_name
                cv2.imwrite(str(aug_image_path), aug_image)
                
                # Save augmented annotations
                aug_label_name = aug_image_name.replace('.jpg', '.txt')
                aug_label_path = output_labels_dir / aug_label_name
                self.write_yolo_annotation(aug_label_path, aug_annotations)
                
            except Exception as e:
                logger.error(f"Error applying grayscale augmentations (copy {gray_idx}) to {image_path}: {e}")
    
    def create_directory_structure(self):
        """Create output directory structure for available splits only."""
        available_splits = []
        for split in ['train', 'valid', 'test']:
            if (self.source_dataset / split / 'images').exists():
                available_splits.append(split)
                (self.output_dataset / split / 'images').mkdir(parents=True, exist_ok=True)
                (self.output_dataset / split / 'labels').mkdir(parents=True, exist_ok=True)
        return available_splits
    
    def copy_original_files(self, available_splits):
        """Copy original images and annotations to output dataset for available splits only."""
        for split in available_splits:
            source_images = self.source_dataset / split / 'images'
            source_labels = self.source_dataset / split / 'labels'
            output_images = self.output_dataset / split / 'images'
            output_labels = self.output_dataset / split / 'labels'
            
            if source_images.exists():
                for image_file in source_images.iterdir():
                    if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        shutil.copy2(image_file, output_images / image_file.name)
            
            if source_labels.exists():
                for label_file in source_labels.iterdir():
                    if label_file.suffix.lower() == '.txt':
                        shutil.copy2(label_file, output_labels / label_file.name)
    
    def generate_data_yaml(self, available_splits):
        """Generate data.yaml file for augmented dataset."""
        # Read source data.yaml if it exists
        source_yaml_path = self.source_dataset / 'data.yaml'
        if source_yaml_path.exists():
            with open(source_yaml_path, 'r') as f:
                data = yaml.safe_load(f)
        else:
            # Create default structure
            data = {
                'nc': 1,
                'names': ['object']
            }
        
        # Update paths to be relative to augmented dataset, only for available splits
        if 'train' in available_splits:
            data['train'] = 'train/images'
        if 'valid' in available_splits:
            data['val'] = 'valid/images'
        if 'test' in available_splits:
            data['test'] = 'test/images'
        
        # Write augmented data.yaml
        output_yaml_path = self.output_dataset / 'data.yaml'
        with open(output_yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        
        logger.info(f"Generated data.yaml with {data['nc']} classes for splits: {available_splits}")
        return data
    
    def augment_dataset(self):
        """Augment the entire dataset."""
        logger.info(f"Starting dataset augmentation from {self.source_dataset} to {self.output_dataset}")
        
        # Create directory structure and get available splits
        available_splits = self.create_directory_structure()
        
        # Copy original files
        logger.info("Copying original files...")
        self.copy_original_files(available_splits)
        
        # Generate data.yaml
        data_config = self.generate_data_yaml(available_splits)
        
        # Process each available split
        total_images = 0
        augmented_images = 0
        
        for split in available_splits:
            source_images_dir = self.source_dataset / split / 'images'
            source_labels_dir = self.source_dataset / split / 'labels'
            output_images_dir = self.output_dataset / split / 'images'
            output_labels_dir = self.output_dataset / split / 'labels'
            
            if not source_images_dir.exists():
                logger.warning(f"Source images directory does not exist: {source_images_dir}")
                continue
            
            image_files = list(source_images_dir.glob('*.jpg')) + list(source_images_dir.glob('*.jpeg')) + list(source_images_dir.glob('*.png'))
            
            logger.info(f"Processing {len(image_files)} images in {split} split...")
            
            for image_path in image_files:
                total_images += 1
                
                # Get corresponding annotation file
                annotation_path = source_labels_dir / f"{image_path.stem}.txt"
                
                # Augment the image
                self.augment_single_image(
                    image_path, annotation_path, 
                    output_images_dir, output_labels_dir
                )
                
                augmented_images += 8  # 6 regular + 1 hflip + 2 grayscale - but we count each as separate
                
                if total_images % 50 == 0:
                    logger.info(f"Processed {total_images} images, generated {augmented_images} augmentations")
        
        logger.info(f"Dataset augmentation complete!")
        logger.info(f"Original images: {total_images}")
        logger.info(f"Total images after augmentation: {total_images + augmented_images}")
        logger.info(f"Augmentation ratio: {(total_images + augmented_images) / total_images:.1f}x")
        
        return {
            'original_images': total_images,
            'augmented_images': augmented_images,
            'total_images': total_images + augmented_images,
            'classes': data_config['nc'],
            'class_names': data_config['names']
        }


def main():
    parser = argparse.ArgumentParser(description='Augment YOLO dataset with various transformations')
    parser.add_argument('--source', '-s', required=True, 
                       help='Path to source YOLO dataset directory')
    parser.add_argument('--output', '-o', required=True, 
                       help='Path to output augmented dataset directory')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed for reproducible results')
    
    args = parser.parse_args()
    
    # Validate source dataset
    source_path = Path(args.source)
    if not source_path.exists():
        logger.error(f"Source dataset directory does not exist: {source_path}")
        return
    
    # Check if source has at least train directory
    if not (source_path / 'train' / 'images').exists():
        logger.error(f"Source dataset missing required train/images directory")
        return
    
    # Log which directories are available
    available_splits = []
    for split in ['train', 'valid', 'test']:
        if (source_path / split / 'images').exists():
            available_splits.append(split)
    
    logger.info(f"Found dataset splits: {available_splits}")
    
    # Create augmentor and run
    augmentor = DatasetAugmentor(args.source, args.output, args.seed)
    
    try:
        stats = augmentor.augment_dataset()
        
        # Print final statistics
        print("\n" + "="*50)
        print("DATASET AUGMENTATION SUMMARY")
        print("="*50)
        print(f"Source Dataset: {args.source}")
        print(f"Output Dataset: {args.output}")
        print(f"Original Images: {stats['original_images']:,}")
        print(f"Augmented Images: {stats['augmented_images']:,}")
        print(f"Total Images: {stats['total_images']:,}")
        print(f"Augmentation Ratio: {stats['total_images'] / stats['original_images']:.1f}x")
        print(f"Classes: {stats['classes']}")
        print(f"Class Names: {', '.join(stats['class_names'])}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Error during augmentation: {e}")
        raise


if __name__ == "__main__":
    main()