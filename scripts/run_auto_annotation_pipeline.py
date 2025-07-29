#!/usr/bin/env python3
"""
Complete pipeline script for auto-annotating license plate dataset with vehicles.
This script runs the entire process from annotation to validation.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, check=True, capture_output=False)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found: {command[0]}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Run complete auto-annotation pipeline')
    parser.add_argument('--input-dataset', required=True, 
                       help='Path to input license plate dataset')
    parser.add_argument('--output-dataset', required=True,
                       help='Path to output enhanced dataset')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold for vehicle detection (default: 0.5)')
    parser.add_argument('--weights', default='weights/yolo11m.pt',
                       help='Path to YOLO11 weights (default: weights/yolo11m.pt)')
    parser.add_argument('--validation-samples', type=int, default=20,
                       help='Number of validation samples to create (default: 20)')
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip validation step')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input_dataset):
        print(f"❌ Input dataset not found: {args.input_dataset}")
        return 1
    
    if not os.path.exists(args.weights):
        print(f"❌ YOLO weights not found: {args.weights}")
        print("Please ensure yolo11m.pt is available in the weights/ directory")
        return 1
    
    # Get script directory
    script_dir = Path(__file__).parent
    
    print("🚀 Starting Auto-Annotation Pipeline")
    print(f"Input Dataset: {args.input_dataset}")
    print(f"Output Dataset: {args.output_dataset}")
    print(f"Confidence Threshold: {args.confidence}")
    print(f"YOLO Weights: {args.weights}")
    
    # Step 1: Auto-annotate vehicles
    auto_annotate_script = script_dir / "auto_annotate_vehicles.py"
    auto_annotate_cmd = [
        sys.executable, str(auto_annotate_script),
        "--dataset", args.input_dataset,
        "--output", args.output_dataset,
        "--confidence", str(args.confidence),
        "--weights", args.weights
    ]
    
    if not run_command(auto_annotate_cmd, "Auto-annotating vehicles"):
        return 1
    
    # Step 2: Validate annotations (if not skipped)
    if not args.skip_validation:
        validation_script = script_dir / "validate_annotations.py"
        validation_output = f"{args.output_dataset}_validation"
        
        validation_cmd = [
            sys.executable, str(validation_script),
            "--dataset", args.output_dataset,
            "--output", validation_output,
            "--samples", str(args.validation_samples)
        ]
        
        if not run_command(validation_cmd, "Validating annotations"):
            print("⚠️  Validation failed, but enhanced dataset was created successfully")
        else:
            print(f"\n📊 Validation results saved to: {validation_output}")
            print("📸 Review the visualization samples to ensure annotation quality")
    
    print(f"\n🎉 Pipeline completed successfully!")
    print(f"Enhanced dataset available at: {args.output_dataset}")
    print(f"\nTo use the enhanced dataset for training:")
    print(f"yolo detect train model=yolo11s.pt data={args.output_dataset}/data.yaml epochs=100 imgsz=640 batch=8")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())