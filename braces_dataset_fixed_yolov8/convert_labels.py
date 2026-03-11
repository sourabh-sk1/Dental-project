"""
Label Conversion Script for Teeth Braces Detection
====================================================
This script converts the original 28-class dataset to a simplified 2-class format:
- Class 0: correct_brace
- Class 1: incorrect_brace

The original dataset contains detailed tooth and bracket annotations.
This conversion focuses on bracket placement status only.
"""

import os
import shutil
from pathlib import Path

# Define class mappings based on original data.yaml
# Correct bracket classes (original indices)
CORRECT_BRACKET_CLASSES = {
    1,   # c bracket correct
    6,   # ci bracket correct
    8,   # cm bracket correct
    10,  # f1 bracket correct
    12,  # f2 bracket correct
    14,  # li bracket correct
    16,  # lm bracket correct
    18,  # mi bracket correct
    20,  # p1 bracket correct
    22,  # p1 bracket correct
}

# Incorrect bracket classes (original indices)
INCORRECT_BRACKET_CLASSES = {
    2,   # c bracket incorrect
    7,   # ci bracket incorrect
    9,   # cm bracket incorrect
    11,  # f1 bracket incorrect
    13,  # f2 bracket incorrect
    15,  # li bracket incorrect
    17,  # lm bracket incorrect
    19,  # mi bracket incorrect
    21,  # p1 bracket incorrect
    23,  # p2 bracket incorrect
}


def convert_label_file(input_path: str, output_path: str) -> bool:
    """
    Convert a single label file from 28 classes to 2 classes.
    
    Args:
        input_path: Path to original label file
        output_path: Path to save converted label file
        
    Returns:
        True if file was converted and has valid detections, False otherwise
    """
    try:
        with open(input_path, 'r') as f:
            lines = f.readlines()
        
        converted_lines = []
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
                
            class_id = int(parts[0])
            bbox = parts[1:]  # Remaining parts are the bounding box coordinates
            
            # Map to new classes
            if class_id in CORRECT_BRACKET_CLASSES:
                new_class_id = 0  # correct_brace
                converted_lines.append(f"0 {' '.join(bbox)}\n")
            elif class_id in INCORRECT_BRACKET_CLASSES:
                new_class_id = 1  # incorrect_brace
                converted_lines.append(f"1 {' '.join(bbox)}\n")
            # Skip tooth-only annotations (not bracket placements)
        
        # Only write if we have valid bracket detections
        if converted_lines:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.writelines(converted_lines)
            return True
        return False
        
    except Exception as e:
        print(f"Error converting {input_path}: {e}")
        return False


def convert_dataset(base_path: str):
    """
    Convert all label files in train, valid, and test folders.
    
    Args:
        base_path: Path to the dataset root directory
    """
    base_path = Path(base_path)
    
    # Directories to process
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        labels_dir = base_path / split / 'labels'
        images_dir = base_path / split / 'images'
        
        if not labels_dir.exists():
            print(f"Warning: {labels_dir} does not exist, skipping...")
            continue
            
        # Process each label file
        converted_count = 0
        skipped_count = 0
        
        for label_file in labels_dir.glob('*.txt'):
            if convert_label_file(str(label_file), str(label_file)):
                converted_count += 1
            else:
                # Remove empty label files
                label_file.unlink()
                skipped_count += 1
        
        print(f"{split}: Converted {converted_count} files, skipped {skipped_count} files (no bracket detections)")


if __name__ == "__main__":
    # Get the directory where this script is located
    dataset_path = Path(__file__).parent
    print(f"Converting labels in: {dataset_path}")
    convert_dataset(dataset_path)
    print("Label conversion complete!")

