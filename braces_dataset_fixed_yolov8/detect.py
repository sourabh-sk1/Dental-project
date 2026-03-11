"""
Teeth Braces Detection Script - Enhanced Version
=================================================
This script performs inference on a single image to detect 
whether orthodontic teeth braces brackets are correctly or incorrectly placed.

The model distinguishes between:
- Class 0: correct_brace - Properly positioned brackets (Green)
- Class 1: incorrect_brace - Misaligned or improperly positioned brackets (Red)

Usage:
    python detect.py --image path/to/image.jpg
    python detect.py --image path/to/image.jpg --save
    python detect.py --image path/to/image.jpg --conf 0.5 --imgsz 1280
"""

import os
import sys
from pathlib import Path
import argparse
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Define color scheme for visualization
COLORS = {
    0: (0, 255, 0),     # Green for correct braces
    1: (0, 0, 255),     # Red for incorrect braces
}

# Class names mapping
CLASS_NAMES = {
    0: 'Correct Brace',
    1: 'Incorrect Brace',
}


def load_model(weights_path: str = 'runs/detect/train/weights/best.pt'):
    """
    Load the trained YOLO model.
    
    Args:
        weights_path: Path to the trained model weights
        
    Returns:
        Loaded YOLO model
    """
    # Check if custom weights path is provided
    if not os.path.exists(weights_path):
        # Try to find weights in default location
        default_weights = 'runs/detect/train/weights/best.pt'
        if os.path.exists(default_weights):
            weights_path = default_weights
            print(f"Using default weights: {weights_path}")
        else:
            print(f"Warning: Model weights not found at {weights_path}")
            print("Will attempt to use a pretrained model...")
            weights_path = 'yolov8n.pt'
    
    print(f"Loading model from: {weights_path}")
    model = YOLO(weights_path)
    
    return model


def detect_braces(model, image_path: str, conf: float = 0.25, iou: float = 0.45, imgsz: int = 1280):
    """
    Run detection on a single image with high accuracy.
    
    Args:
        model: Loaded YOLO model
        image_path: Path to input image
        conf: Confidence threshold (0-1)
        iou: IoU threshold for NMS
        imgsz: Image size for inference (higher = better accuracy)
        
    Returns:
        Detection results
    """
    # Run inference with higher image size for better accuracy
    results = model(
        image_path,
        conf=conf,      # Confidence threshold
        iou=iou,        # IoU threshold for NMS
        imgsz=imgsz,    # Higher resolution for better accuracy
        verbose=True    # Show detailed output
    )
    
    return results


def draw_detections(image: np.ndarray, results, enhanced: bool = True) -> tuple:
    """
    Draw bounding boxes and labels on the image with enhanced styling.
    
    Args:
        image: Input image as numpy array
        results: YOLO detection results
        enhanced: Whether to use enhanced visualization
        
    Returns:
        Tuple of (annotated image, detection summary)
    """
    # Get the result for the first (and only) image
    result = results[0]
    
    # Create a copy of the image for drawing
    annotated_image = image.copy()
    
    # Initialize counters
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
            
            # Get class ID
            class_id = int(box.cls[0].cpu().numpy())
            
            # Get color for this class
            color = COLORS.get(class_id, (255, 255, 255))
            
            if enhanced:
                # Enhanced visualization with gradient effect
                # Outer glow effect
                cv2.rectangle(
                    annotated_image,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (int(color[0]*0.5), int(color[1]*0.5), int(color[2]*0.5)),
                    6
                )
                
                # Main rectangle
                cv2.rectangle(
                    annotated_image,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    color,
                    3
                )
                
                # Prepare label text with high confidence display
                label = f"{CLASS_NAMES[class_id]}: {conf*100:.1f}%"
                
                # Get text size for background
                (text_width, text_height), baseline = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    2
                )
                
                # Draw filled rectangle behind text
                cv2.rectangle(
                    annotated_image,
                    (int(x1), int(y1) - text_height - 15),
                    (int(x1) + text_width + 10, int(y1)),
                    color,
                    -1
                )
                
                # Draw text with shadow
                cv2.putText(
                    annotated_image,
                    label,
                    (int(x1) + 5, int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                
                # Draw confidence bar
                bar_height = 6
                bar_y = int(y2) + 5
                bar_x = int(x1)
                bar_width = int(x2 - x1)
                
                # Background bar
                cv2.rectangle(
                    annotated_image,
                    (bar_x, bar_y),
                    (bar_x + bar_width, bar_y + bar_height),
                    (50, 50, 50),
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
            else:
                # Standard visualization
                cv2.rectangle(
                    annotated_image,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    color,
                    2
                )
                
                label = f"{CLASS_NAMES[class_id]}: {conf:.2f}"
                
                (text_width, text_height), baseline = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    1
                )
                
                cv2.rectangle(
                    annotated_image,
                    (int(x1), int(y1) - text_height - 10),
                    (int(x1) + text_width, int(y1)),
                    color,
                    -1
                )
                
                cv2.putText(
                    annotated_image,
                    label,
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
            
            # Update counters
            if class_id == 0:
                correct_count += 1
            elif class_id == 1:
                incorrect_count += 1
            
            # Store detection info
            detections.append({
                'class': CLASS_NAMES[class_id],
                'confidence': conf,
                'confidence_percent': f"{conf*100:.1f}%",
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


def display_results(image: np.ndarray, summary: dict):
    """
    Display detection results in the console.
    
    Args:
        image: Annotated image
        summary: Detection summary dictionary
    """
    print("\n" + "=" * 60)
    print("DETECTION RESULTS - Enhanced")
    print("=" * 60)
    
    print(f"\nTotal Braces Detected: {summary['total_detections']}")
    print(f"  ✓ Correct Braces: {summary['correct_braces']}")
    print(f"  ✗ Incorrect Braces: {summary['incorrect_braces']}")
    
    if summary['detections']:
        print("\nDetailed Detections:")
        print("-" * 60)
        for i, det in enumerate(summary['detections'], 1):
            print(f"{i}. {det['class']} - Confidence: {det['confidence_percent']}")
            bbox = det['bbox']
            print(f"   Bounding Box: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
    
    print("\n" + "=" * 60)


def save_results(image: np.ndarray, output_path: str):
    """
    Save annotated image to file.
    
    Args:
        image: Annotated image
        output_path: Path to save the image
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Save the image
    cv2.imwrite(output_path, image)
    print(f"\nAnnotated image saved to: {output_path}")


def show_image(image: np.ndarray, title: str = "Teeth Braces Detection - Enhanced"):
    """
    Display image using matplotlib.
    
    Args:
        image: Image to display (BGR format)
        title: Window title
    """
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    plt.imshow(image_rgb)
    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function for running detection from command line.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Teeth Braces Detection using YOLOv8 - Enhanced Version',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect braces in an image with high accuracy
  python detect.py --image path/to/image.jpg
  
  # Detect and save the result
  python detect.py --image path/to/image.jpg --save
  
  # Detect with custom confidence threshold
  python detect.py --image path/to/image.jpg --conf 0.5
  
  # Detect with higher image size for better accuracy
  python detect.py --image path/to/image.jpg --imgsz 1280
  
  # Detect and display the result
  python detect.py --image path/to/image.jpg --show
        """
    )
    
    parser.add_argument(
        '--image', '-i',
        type=str,
        required=True,
        help='Path to input image'
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
        default=0.25,
        help='Confidence threshold (default: 0.25)'
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
        default=1280,
        help='Inference image size (default: 1280 for higher accuracy)'
    )
    
    parser.add_argument(
        '--enhanced', '-e',
        action='store_true',
        default=True,
        help='Use enhanced visualization with confidence bars'
    )
    
    parser.add_argument(
        '--no-enhanced',
        action='store_true',
        help='Disable enhanced visualization'
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
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
        sys.exit(1)
    
    # Determine enhanced mode
    enhanced = args.enhanced and not args.no_enhanced
    
    print("=" * 60)
    print("Teeth Braces Detection - Enhanced Version")
    print("=" * 60)
    print(f"Input Image: {args.image}")
    print(f"Confidence Threshold: {args.conf}")
    print(f"IoU Threshold: {args.iou}")
    print(f"Image Size: {args.imgsz}")
    print(f"Enhanced Visualization: {enhanced}")
    
    # Load the model
    model = load_model(args.weights)
    
    # Read the image
    print(f"\nProcessing image: {args.image}")
    image = cv2.imread(args.image)
    
    if image is None:
        print(f"Error: Could not read image at {args.image}")
        sys.exit(1)
    
    print(f"Image shape: {image.shape}")
    
    # Run detection with high accuracy
    print("\nRunning detection with high accuracy...")
    results = detect_braces(model, args.image, conf=args.conf, iou=args.iou, imgsz=args.imgsz)
    
    # Draw detections on the image with enhanced visualization
    annotated_image, summary = draw_detections(image, results, enhanced=enhanced)
    
    # Display results in console
    display_results(annotated_image, summary)
    
    # Save results if requested
    if args.save:
        output_path = args.image.replace('.jpg', '_result.jpg')
        save_results(annotated_image, output_path)
    
    # Show image if requested
    if args.show:
        show_image(annotated_image)
    
    print("\nDetection complete!")
    
    return annotated_image, summary


if __name__ == "__main__":
    main()

