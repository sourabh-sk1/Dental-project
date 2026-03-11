"""
Enhanced YOLOv8 Detection Script for Teeth Braces Detection
=============================================================
This script performs inference on dental images to detect orthodontic
braces brackets and determine if they are correctly or incorrectly placed.

Features:
- Position-based tooth inference (identifies which tooth)
- Enhanced confidence threshold (0.40 - as per requirements)
- NMS tuning for better detection
- Duplicate detection removal
- Color-coded bounding boxes (Green=Correct, Red=Incorrect)
- Detailed labels with tooth name, status, and confidence
- Post-processing to remove duplicate detections

Usage:
    python detect.py --image path/to/image.jpg
    python detect.py --image path/to/image.jpg --conf 0.5 --save
    python detect.py --image path/to/image.jpg --imgsz 1280
"""

import os
import sys
from pathlib import Path
import argparse
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

# Add utils to path for tooth mapping
sys.path.append(str(Path(__file__).parent))

from utils.tooth_mapping import (
    TOOTH_NAMES,
    TOOTH_SHORT_CODES,
    TOOTH_DISPLAY_NAMES,
    TOOTH_ARCH,
    TOOTH_COLORS,
    get_tooth_from_position,
    format_detection_label,
    get_color_for_detection,
    get_color_dark_for_detection,
)


# =============================================================================
# CONSTANTS
# =============================================================================

# Confidence threshold as per requirements
DEFAULT_CONFIDENCE = 0.40

# Class names mapping
CLASS_NAMES = {
    0: 'Correct Brace',
    1: 'Incorrect Brace',
}


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def draw_enhanced_detections(
    image: np.ndarray,
    results,
    show_confidence: bool = True,
    show_tooth_name: bool = True,
    thickness: int = 3
) -> Tuple[np.ndarray, Dict]:
    """
    Draw enhanced bounding boxes with tooth inference on the image.
    
    Args:
        image: Input image as numpy array (BGR format)
        results: YOLO detection results
        show_confidence: Whether to show confidence scores
        show_tooth_name: Whether to show tooth name
        thickness: Line thickness for bounding boxes
        
    Returns:
        Tuple of (annotated image, detection summary)
    """
    # Get the result for the first (and only) image
    result = results[0]
    
    # Create a copy of the image for drawing
    annotated_image = image.copy()
    
    # Get image shape for tooth inference
    image_shape = image.shape[:2]  # (height, width)
    
    # Initialize counters and storage
    correct_count = 0
    incorrect_count = 0
    detections = []
    
    # Get bounding boxes, confidence scores, and class IDs
    boxes = result.boxes
    
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Get confidence score
            conf = float(box.conf[0].cpu().numpy())
            
            # Skip low confidence detections (threshold = 0.40 as per requirements)
            if conf < DEFAULT_CONFIDENCE:
                continue
            
            # Get class ID
            class_id = int(box.cls[0].cpu().numpy())
            
            # Determine correctness
            is_correct = (class_id == 0)  # Class 0 = correct, Class 1 = incorrect
            
            # Get colors
            color = get_color_for_detection(is_correct)
            color_dark = get_color_dark_for_detection(is_correct)
            
            # Infer tooth from position
            # Convert to normalized xywh for tooth inference
            img_h, img_w = image_shape
            x_center = ((x1 + x2) / 2) / img_w
            y_center = ((y1 + y2) / 2) / img_h
            width = (x2 - x1) / img_w
            height = (y2 - y1) / img_h
            
            norm_bbox = (x_center, y_center, width, height)
            tooth_key = get_tooth_from_position(norm_bbox, image_shape)
            
            # Get tooth name for display
            tooth_short = TOOTH_SHORT_CODES.get(tooth_key, 'UNK')
            tooth_display = TOOTH_DISPLAY_NAMES.get(tooth_key, 'Unknown')
            
            # Draw enhanced bounding box with glow effect
            # Outer glow (thicker, darker)
            cv2.rectangle(
                annotated_image,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color_dark,
                thickness + 3
            )
            
            # Main rectangle
            cv2.rectangle(
                annotated_image,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                thickness
            )
            
            # Prepare label text
            status = "Correct" if is_correct else "Incorrect"
            if show_confidence and show_tooth_name:
                label = f"{tooth_short} | {status} | {conf:.2f}"
            elif show_tooth_name:
                label = f"{tooth_short} | {status}"
            elif show_confidence:
                label = f"{status} | {conf:.2f}"
            else:
                label = status
            
            # Get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )
            
            # Draw text background
            text_bg_y1 = int(y1) - text_height - 15
            text_bg_y2 = int(y1)
            
            # Ensure text background is within image bounds
            if text_bg_y1 < 0:
                text_bg_y1 = int(y1)
                text_bg_y2 = int(y1) + text_height + 10
            
            # Draw filled rectangle behind text
            cv2.rectangle(
                annotated_image,
                (int(x1), text_bg_y1),
                (int(x1) + text_width + 15, text_bg_y2),
                color,
                -1
            )
            
            # Draw text with shadow for better visibility
            # Shadow
            cv2.putText(
                annotated_image,
                label,
                (int(x1) + 4, text_bg_y2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                3
            )
            # Main text
            cv2.putText(
                annotated_image,
                label,
                (int(x1) + 3, text_bg_y2 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            
            # Draw confidence bar below the box
            if show_confidence:
                bar_height = 6
                bar_y = int(y2) + 5
                bar_x = int(x1)
                bar_width = int(x2 - x1)
                
                # Ensure bar is within image bounds
                if bar_y + bar_height < image.shape[0]:
                    # Background bar
                    cv2.rectangle(
                        annotated_image,
                        (bar_x, bar_y),
                        (bar_x + bar_width, bar_y + bar_height),
                        (40, 40, 40),
                        -1
                    )
                    
                    # Confidence fill
                    fill_width = int(bar_width * conf)
                    cv2.rectangle(
                        annotated_image,
                        (bar_x, bar_y),
                        (bar_x + fill_width, bar_y + bar_height),
                        color,
                        -1
                    )
            
            # Update counters
            if is_correct:
                correct_count += 1
            else:
                incorrect_count += 1
            
            # Store detection info
            detections.append({
                'class': 'Correct Brace' if is_correct else 'Incorrect Brace',
                'class_id': class_id,
                'tooth_key': tooth_key,
                'tooth_name': tooth_display,
                'tooth_short': tooth_short,
                'arch': TOOTH_ARCH.get(tooth_key, 'Unknown'),
                'is_correct': is_correct,
                'status': 'Correct' if is_correct else 'Incorrect',
                'confidence': conf,
                'confidence_percent': f"{conf * 100:.1f}%",
                'bbox': [float(x1), float(y1), float(x2), float(y2)]
            })
    
    # Create summary
    summary = {
        'total_detections': len(detections),
        'correct_braces': correct_count,
        'incorrect_braces': incorrect_count,
        'detections': detections
    }
    
    return annotated_image, summary


# =============================================================================
# DETECTION FUNCTIONS
# =============================================================================

def load_model(weights_path: str = 'runs/detect/train/weights/best.pt') -> YOLO:
    """
    Load the trained YOLO model.
    
    Args:
        weights_path: Path to the trained model weights
        
    Returns:
        Loaded YOLO model
    """
    # Check if custom weights path is provided
    if not os.path.exists(weights_path):
        # Try to find weights in default locations
        default_weights = [
            'runs/detect/train/weights/best.pt',
            '../braces_dataset_fixed_yolov8/runs/detect/train/weights/best.pt',
            'yolov8l.pt',  # Fallback to pretrained model
        ]
        for w in default_weights:
            if os.path.exists(w):
                weights_path = w
                print(f"Using weights: {weights_path}")
                break
        else:
            print(f"⚠️ Warning: No weights found, using yolov8l.pt")
            weights_path = 'yolov8l.pt'
    
    print(f"🤖 Loading model from: {weights_path}")
    model = YOLO(weights_path)
    
    return model


def detect_braces(
    model: YOLO,
    image_path: str,
    conf: float = DEFAULT_CONFIDENCE,
    iou: float = 0.45,
    imgsz: int = 960
) -> List:
    """
    Run detection on a single image with enhanced settings.
    
    Args:
        model: Loaded YOLO model
        image_path: Path to input image
        conf: Confidence threshold (default: 0.40 for better precision)
        iou: IoU threshold for NMS (default: 0.45)
        imgsz: Image size for inference
        
    Returns:
        Detection results
    """
    # Run inference with optimized settings
    results = model(
        image_path,
        conf=conf,       # Higher confidence threshold for better precision
        iou=iou,        # IoU threshold for NMS
        imgsz=imgsz,    # Higher resolution for better accuracy
        verbose=False,   # Suppress verbose output
        Agnostic_nms=False,
        augment=False,
        retina_masks=False,
    )
    
    return results


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: First bounding box [x1, y1, x2, y2]
        box2: Second bounding box [x1, y1, x2, y2]
        
    Returns:
        IoU value between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    # Check if there is an intersection
    if x2_i > x1_i and y2_i > y1_i:
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
    else:
        intersection = 0
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    # Calculate IoU
    iou = intersection / union if union > 0 else 0
    
    return iou


def apply_post_processing(
    detections: List[Dict],
    iou_threshold: float = 0.5
) -> List[Dict]:
    """
    Apply post-processing to remove duplicate detections.
    
    This function:
    1. Sorts detections by confidence (highest first)
    2. Removes detections with high IoU overlap
    3. Keeps only the highest confidence detection in overlapping regions
    
    Args:
        detections: List of detection dictionaries
        iou_threshold: IoU threshold for duplicate removal (default: 0.5)
        
    Returns:
        Filtered list of detections
    """
    if len(detections) <= 1:
        return detections
    
    # Sort by confidence (highest first)
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    # Remove duplicates based on IoU
    filtered = []
    for det in detections:
        is_duplicate = False
        for kept_det in filtered:
            # Calculate IoU
            iou = calculate_iou(det['bbox'], kept_det['bbox'])
            
            if iou > iou_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            filtered.append(det)
    
    return filtered


# =============================================================================
# DISPLAY AND OUTPUT FUNCTIONS
# =============================================================================

def display_results_console(summary: Dict):
    """
    Display detection results in the console with formatted output.
    
    Args:
        summary: Detection summary dictionary
    """
    print("\n" + "=" * 70)
    print("🦷 TEETH BRACES DETECTION RESULTS")
    print("=" * 70)
    
    print(f"\n📊 Summary:")
    print(f"   Total Detections: {summary['total_detections']}")
    print(f"   ✓ Correct Braces:  {summary['correct_braces']}")
    print(f"   ✗ Incorrect Braces: {summary['incorrect_braces']}")
    
    if summary['detections']:
        print("\n📋 Detailed Detections:")
        print("-" * 70)
        print(f"{'#':<3} {'Tooth':<10} {'Status':<12} {'Confidence':<15} {'Position'}")
        print("-" * 70)
        
        for i, det in enumerate(summary['detections'], 1):
            status_icon = "✓" if det['is_correct'] else "✗"
            print(
                f"{i:<3} {det['tooth_short']:<10} "
                f"{status_icon} {det['status']:<10} "
                f"{det['confidence']:.2f} ({det['confidence_percent']})   "
                f"{det['arch']}"
            )
        
        print("-" * 70)
        
        # Warnings for incorrect braces
        if summary['incorrect_braces'] > 0:
            print("\n⚠️  WARNINGS:")
            for det in summary['detections']:
                if not det['is_correct']:
                    print(f"   • {det['tooth_name']} has incorrect brace placement")
    
    print("\n" + "=" * 70)


def save_results(
    image: np.ndarray,
    output_path: str,
    create_preview: bool = True
):
    """
    Save annotated image to file.
    
    Args:
        image: Annotated image
        output_path: Path to save the image
        create_preview: Whether to create a smaller preview
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Save the image
    cv2.imwrite(output_path, image)
    print(f"\n💾 Annotated image saved to: {output_path}")
    
    # Create and save a smaller preview
    if create_preview:
        preview_path = output_path.replace('.jpg', '_preview.jpg')
        height, width = image.shape[:2]
        max_dim = 800
        scale = min(max_dim / height, max_dim / width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        preview = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        cv2.imwrite(preview_path, preview)
        print(f"💾 Preview saved to: {preview_path}")


def show_image(
    image: np.ndarray,
    title: str = "Teeth Braces Detection"
):
    """
    Display image using matplotlib.
    
    Args:
        image: Image to display (BGR format)
        title: Window title
    """
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure
    plt.figure(figsize=(14, 10))
    plt.imshow(image_rgb)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def create_results_csv(summary: Dict, output_path: str = 'detection_results.csv'):
    """
    Create a CSV file with detection results.
    
    Args:
        summary: Detection summary dictionary
        output_path: Path to save CSV file
    """
    import csv
    
    if not summary['detections']:
        print("No detections to save.")
        return
    
    # Write CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'ID', 'Tooth Name', 'Short Code', 'Arch',
            'Status', 'Confidence', 'Confidence %',
            'BBox X1', 'BBox Y1', 'BBox X2', 'BBox Y2'
        ])
        
        for i, det in enumerate(summary['detections'], 1):
            writer.writerow([
                i,
                det['tooth_name'],
                det['tooth_short'],
                det['arch'],
                det['status'],
                f"{det['confidence']:.4f}",
                det['confidence_percent'],
                f"{det['bbox'][0]:.1f}",
                f"{det['bbox'][1]:.1f}",
                f"{det['bbox'][2]:.1f}",
                f"{det['bbox'][3]:.1f}"
            ])
    
    print(f"📊 Results saved to: {output_path}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """
    Main function for running detection from command line.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Enhanced Teeth Braces Detection using YOLOv8',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect braces in an image (default confidence 0.40)
  python detect.py --image path/to/image.jpg
  
  # Detect and save the result
  python detect.py --image path/to/image.jpg --save
  
  # Detect with custom confidence threshold
  python detect.py --image path/to/image.jpg --conf 0.5
  
  # Detect with larger image size for better accuracy
  python detect.py --image path/to/image.jpg --imgsz 1280 --show
  
  # Batch process multiple images
  python detect.py --batch path/to/images/
        """
    )
    
    parser.add_argument(
        '--image', '-i',
        type=str,
        help='Path to input image'
    )
    
    parser.add_argument(
        '--batch', '-b',
        type=str,
        help='Path to directory with images for batch processing'
    )
    
    parser.add_argument(
        '--weights', '-w',
        type=str,
        default='runs/detect/train/weights/best.pt',
        help='Path to model weights (default: runs/detect/train/weights/best.pt)'
    )
    
    parser.add_argument(
        '--conf', '-c',
        type=float,
        default=DEFAULT_CONFIDENCE,
        help=f'Confidence threshold (default: {DEFAULT_CONFIDENCE})'
    )
    
    parser.add_argument(
        '--iou', '-iou',
        type=float,
        default=0.45,
        help='IoU threshold for NMS (default: 0.45)'
    )
    
    parser.add_argument(
        '--imgsz', '-im',
        type=int,
        default=960,
        help='Inference image size (default: 960)'
    )
    
    parser.add_argument(
        '--save', '-s',
        action='store_true',
        help='Save annotated image'
    )
    
    parser.add_argument(
        '--show', '-sh',
        action='store_true',
        help='Display annotated image'
    )
    
    parser.add_argument(
        '--csv',
        action='store_true',
        help='Save results to CSV file'
    )
    
    parser.add_argument(
        '--no-tooth-name',
        action='store_true',
        help='Hide tooth name from labels'
    )
    
    parser.add_argument(
        '--no-confidence',
        action='store_true',
        help='Hide confidence from labels'
    )
    
    args = parser.parse_args()
    
    # Check if image or batch is provided
    if not args.image and not args.batch:
        parser.error("Either --image or --batch must be provided")
    
    # Print banner
    print("\n" + "=" * 70)
    print("🦷 Teeth Braces Detection - Enhanced Version")
    print("=" * 70)
    print(f"🔧 Configuration:")
    print(f"   Confidence Threshold: {args.conf} (minimum required: {DEFAULT_CONFIDENCE})")
    print(f"   IoU Threshold: {args.iou}")
    print(f"   Image Size: {args.imgsz}")
    print(f"   Show Tooth Names: {not args.no_tooth_name}")
    print(f"   Show Confidence: {not args.no_confidence}")
    
    # Load the model
    model = load_model(args.weights)
    
    # Determine which images to process
    if args.image:
        image_paths = [args.image]
    else:
        # Batch processing
        image_dir = Path(args.batch)
        if not image_dir.exists():
            print(f"❌ Error: Directory not found: {image_dir}")
            sys.exit(1)
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}
        image_paths = [
            str(f) for f in image_dir.iterdir() 
            if f.is_file() and f.suffix in image_extensions
        ]
        
        if not image_paths:
            print(f"❌ Error: No images found in {image_dir}")
            sys.exit(1)
        
        print(f"\n📁 Found {len(image_paths)} images for batch processing")
    
    # Process each image
    for idx, image_path in enumerate(image_paths, 1):
        print(f"\n{'='*70}")
        print(f"📷 Processing {idx}/{len(image_paths)}: {image_path}")
        print(f"{'='*70}")
        
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"⚠️ Warning: Image not found: {image_path}")
            continue
        
        # Read the image
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"⚠️ Warning: Could not read image: {image_path}")
            continue
        
        print(f"📐 Image shape: {image.shape}")
        
        # Run detection
        results = detect_braces(model, image_path, conf=args.conf, iou=args.iou, imgsz=args.imgsz)
        
        # Draw detections with enhanced visualization
        annotated_image, summary = draw_enhanced_detections(
            image,
            results,
            show_confidence=not args.no_confidence,
            show_tooth_name=not args.no_tooth_name
        )
        
        # Apply post-processing to remove duplicates
        if summary['detections']:
            summary['detections'] = apply_post_processing(summary['detections'], iou_threshold=0.5)
            summary['total_detections'] = len(summary['detections'])
            # Recalculate counts
            summary['correct_braces'] = sum(1 for d in summary['detections'] if d['is_correct'])
            summary['incorrect_braces'] = sum(1 for d in summary['detections'] if not d['is_correct'])
        
        # Display results
        display_results_console(summary)
        
        # Save results if requested
        if args.save:
            output_path = image_path.replace('.jpg', '_result.jpg')
            if args.batch:
                output_path = str(image_dir / f"{Path(image_path).stem}_result.jpg")
            save_results(annotated_image, output_path)
        
        # Save CSV if requested
        if args.csv and summary['detections']:
            csv_path = image_path.replace('.jpg', '_results.csv')
            if args.batch:
                csv_path = str(image_dir / f"{Path(image_path).stem}_results.csv")
            create_results_csv(summary, csv_path)
        
        # Show image if requested
        if args.show:
            show_image(annotated_image)
    
    print(f"\n{'='*70}")
    print("✅ Detection complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

