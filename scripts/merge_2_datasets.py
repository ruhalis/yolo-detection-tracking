import os
import shutil
import argparse

# ----- Dataset-specific mappings -----
# Update these dictionaries with the mapping from original label indices to unified class names for each dataset.

dataset1_mapping = {  # cat detection
    0: "License_Plate",
    1: "Vehicle"
}
dataset2_mapping = { # garage aug split
    0: "Vehicle",
    1: "License_Plate"
}

# ----- Unified class mapping -----
# This dictionary maps unified class names to new indices.
new_class_dict = {
    "License_Plate": 0,
    "Vehicle": 1
}
# ----- Function to convert a single label file -----
def convert_label_file(input_path, output_path, dataset_mapping):
    """
    Reads a YOLO label file from input_path, converts the original class indices
    to unified indices using the provided dataset_mapping and new_class_dict,
    and writes the new content to output_path.
    """
    with open(input_path, "r") as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            print(f"Skipping malformed line in {input_path}: {line}")
            continue
        try:
            orig_index = int(parts[0])
        except ValueError:
            print(f"Skipping non-integer class index in {input_path}: {line}")
            continue

        # Get the class name from the dataset-specific mapping.
        orig_class = dataset_mapping.get(orig_index)
        if orig_class is None:
            print(f"No mapping defined for index {orig_index} in {input_path}")
            continue

        # Map to the new unified index.
        new_index = new_class_dict.get(orig_class)
        if new_index is None:
            print(f"No new index defined for unified class '{orig_class}' in {input_path}")
            continue

        new_line = " ".join([str(new_index)] + parts[1:])
        new_lines.append(new_line)
    
    with open(output_path, "w") as f:
        for nl in new_lines:
            f.write(nl + "\n")

# ----- Function to process one split of one dataset -----
def process_split(dataset_path, split, output_dir, prefix, dataset_mapping):
    """
    Processes one split (e.g., train, valid, or test) of a dataset.
    It expects that dataset_path/<split> contains "images" and "labels" folders.
    Converted label files are written to output_dir/<split>/labels,
    and corresponding image files are copied to output_dir/<split>/images.
    Filenames are prefixed with the provided prefix.
    """
    split_dir = os.path.join(dataset_path, split)
    images_dir = os.path.join(split_dir, "images")
    labels_dir = os.path.join(split_dir, "labels")

    out_images_dir = os.path.join(output_dir, split, "images")
    out_labels_dir = os.path.join(output_dir, split, "labels")
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)

    for label_file in os.listdir(labels_dir):
        if not label_file.endswith(".txt"):
            continue

        base_name = os.path.splitext(label_file)[0]
        new_label_filename = f"{prefix}_{label_file}"
        input_label_path = os.path.join(labels_dir, label_file)
        output_label_path = os.path.join(out_labels_dir, new_label_filename)

        # Convert the label file using the appropriate mapping.
        convert_label_file(input_label_path, output_label_path, dataset_mapping)
        print(f"Processed label file: {output_label_path}")

        # Copy the corresponding image file.
        image_found = False
        for ext in [".jpg", ".jpeg", ".png"]:
            image_filename = base_name + ext
            input_image_path = os.path.join(images_dir, image_filename)
            if os.path.exists(input_image_path):
                new_image_filename = f"{prefix}_{image_filename}"
                output_image_path = os.path.join(out_images_dir, new_image_filename)
                shutil.copy2(input_image_path, output_image_path)
                print(f"Copied image file: {output_image_path}")
                image_found = True
                break
        if not image_found:
            print(f"Warning: No image found for label file {label_file}")

# ----- Main function to process two datasets -----
def main():
    parser = argparse.ArgumentParser(
        description="Merge two YOLO datasets with different class mappings into one unified dataset."
    )
    parser.add_argument("--dataset1", type=str, required=True,
                        help="Path to the first dataset (with train, valid, test folders).")
    parser.add_argument("--dataset2", type=str, required=True,
                        help="Path to the second dataset (with train, valid, test folders).")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to the output merged dataset directory.")
    args = parser.parse_args()

    # Define the splits to process.
    splits = ["train", "valid", "test"]

    for split in splits:
        # Process dataset1 using its mapping.
        prefix1 = f"ds1_{split}"
        process_split(args.dataset1, split, args.output, prefix1, dataset1_mapping)
        
        # Process dataset2 using its mapping.
        prefix2 = f"ds2_{split}"
        process_split(args.dataset2, split, args.output, prefix2, dataset2_mapping)

    # Create a unified classes file in the output directory.
    classes_file = os.path.join(args.output, "classes.txt")
    with open(classes_file, "w") as f:
        # Order the unified classes by new index.
        sorted_classes = sorted(new_class_dict.items(), key=lambda x: x[1])
        for class_name, _ in sorted_classes:
            f.write(class_name + "\n")
    print(f"Unified classes file written to: {classes_file}")
    print("Merge complete.")

if __name__ == "__main__":
    main()