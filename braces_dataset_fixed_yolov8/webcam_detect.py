"""
Real-time Webcam Detection for Teeth Braces - Enhanced Version
===============================================================
This script captures video from webcam and performs real-time 
detection of orthodontic teeth braces brackets with high accuracy.

The model distinguishes between:
- Class 0: correct_brace - Properly positioned brackets (Green)
- Class 1: incorrect_brace - Misaligned or improperly positioned brackets (Red)

Usage:
    python webcam_detect.py
    python webcam_detect.py --conf 0.5 --imgsz 1280
    python webcam_detect.py --device 0
"""

import os
import sys
from pathlib import Path
import argparse
import cv2
import numpy as np
from ultralytics import YOLO
import time

# Define color scheme for visualization
COLORS = {
    0: (0, 255, 0),     # Green for correct braces
    1: (0, 0, 255),     # Red for incorrect braces
}

# Class names mapping
CLASS_NAMES = {
    0: 'Correct',
    1: 'Incorrect',
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


def draw_detections_enhanced(frame: np.ndarray, boxes, frame_count: int, show_conf_bar: bool = True) -> tuple:
    """
    Draw bounding boxes and labels on the video frame with enhanced styling.
    
    Args:
        frame: Video frame as numpy array
        boxes: YOLO detection boxes
        frame_count: Current frame number
        show_conf_bar: Whether to show confidence bars
        
    Returns:
        Tuple of (annotated frame, detection summary)
    """
    # Create a copy of the frame for drawing
    annotated_frame = frame.copy()
    
    # Initialize counters
    correct_count = 0
    incorrect_count = 0
    
    # Check if there are any detections
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
            
            # Draw rectangle with thicker lines for visibility - enhanced version
            # Outer glow effect
            cv2.rectangle(
                annotated_frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (int(color[0]*0.5), int(color[1]*0.5), int(color[2]*0.5)),
                6  # Thicker outer glow
            )
            
            # Main rectangle
            cv2.rectangle(
                annotated_frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                3  # Thicker line for video
            )
            
            # Prepare label text with high confidence display
            label = f"{CLASS_NAMES[class_id]}: {conf*100:.1f}%"
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,  # Slightly larger font for video
                2     # Thicker text
            )
            
            # Draw filled rectangle behind text
            cv2.rectangle(
                annotated_frame,
                (int(x1), int(y1) - text_height - 15),
                (int(x1) + text_width + 10, int(y1)),
                color,
                -1  # Filled rectangle
            )
            
            # Draw text
            cv2.putText(
                annotated_frame,
                label,
                (int(x1) + 5, int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,  # Font scale
                (255, 255, 255),  # Text color (white)
                2     # Thickness
            )
            
            # Draw confidence bar if enabled
            if show_conf_bar:
                bar_height = 6
                bar_y = int(y2) + 5
                bar_x = int(x1)
                bar_width = int(x2 - x1)
                
                # Background bar
                cv2.rectangle(
                    annotated_frame,
                    (bar_x, bar_y),
                    (bar_x + bar_width, bar_y + bar_height),
                    (50, 50, 50),
                    -1
                )
                
                # Confidence fill
                fill_width = int(bar_width * conf)
                cv2.rectangle(
                    annotated_frame,
                    (bar_x, bar_y),
                    (bar_x + fill_width, bar_y + bar_height),
                    color,
                    -1
                )
            
            # Update counters
            if class_id == 0:
                correct_count += 1
            elif class_id == 1:
                incorrect_count += 1
    
    # Add info overlay
    info_text = f"Frame: {frame_count} | Correct: {correct_count} | Incorrect: {incorrect_count}"
    
    # Draw semi-transparent background for info text
    h, w = annotated_frame.shape[:2]
    overlay = annotated_frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + len(info_text) * 12, 50), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
    
    # Draw info text
    cv2.putText(
        annotated_frame,
        info_text,
        (15, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )
    
    # Add status message
    if incorrect_count > 0:
        status = "⚠️ INCORRECT BRACES DETECTED!"
        status_color = (0, 0, 255)  # Red
    elif correct_count > 0:
        status = "✓ All Braces Correct"
        status_color = (0, 255, 0)  # Green
    else:
        status = "No braces detected"
        status_color = (255, 255, 255)  # White
    
    # Draw status at bottom of frame
    cv2.putText(
        annotated_frame,
        status,
        (15, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        status_color,
        3
    )
    
    return annotated_frame, (correct_count, incorrect_count)


def run_webcam_detection(
    model,
    camera_index: int = 0,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 1280,
    show_fps: bool = True,
    show_conf_bar: bool = True
):
    """
    Run real-time detection on webcam feed with high accuracy.
    
    Args:
        model: Loaded YOLO model
        camera_index: Webcam device index (0 for default camera)
        conf: Confidence threshold (0-1)
        iou: IoU threshold for NMS
        imgsz: Image size for inference (higher = better accuracy)
        show_fps: Whether to show FPS counter
        show_conf_bar: Whether to show confidence bars
        
    Returns:
        None
    """
    # Initialize webcam capture
    print(f"\nInitializing webcam (index: {camera_index})...")
    cap = cv2.VideoCapture(camera_index)
    
    # Check if webcam is available
    if not cap.isOpened():
        print(f"Error: Could not open webcam at index {camera_index}")
        print("Make sure your webcam is connected and accessible.")
        sys.exit(1)
    
    # Set video frame properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Get actual video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"Video properties: {frame_width}x{frame_height} @ {fps} FPS")
    print(f"Inference image size: {imgsz}")
    print(f"Confidence threshold: {conf}")
    print("\nControls:")
    print("  'q' or 'ESC' - Quit")
    print("  's' - Take screenshot")
    print("  'h' - Toggle confidence bars")
    print("\nStarting detection with high accuracy... Press 'q' or ESC to exit.")
    
    # Initialize FPS calculation
    frame_count = 0
    fps_display = 0
    start_time = time.time()
    fps_update_interval = 10  # Update FPS every 10 frames
    
    # Create output directory for screenshots
    screenshot_dir = 'screenshots'
    os.makedirs(screenshot_dir, exist_ok=True)
    
    # Main detection loop
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to read frame from webcam")
            break
        
        frame_count += 1
        
        # Update FPS every few frames
        if frame_count % fps_update_interval == 0:
            elapsed = time.time() - start_time
            fps_display = fps_update_interval / elapsed
            start_time = time.time()
        
        # Run detection on frame with high accuracy
        # Use stream=True for faster inference on video
        results = model(
            frame,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            verbose=False,
            stream=True
        )
        
        # Process results
        for result in results:
            # Get boxes
            boxes = result.boxes
            
            # Draw detections on frame with enhanced visualization
            annotated_frame, counts = draw_detections_enhanced(
                frame, boxes, frame_count, show_conf_bar
            )
            
            # Add FPS counter if enabled
            if show_fps:
                fps_text = f"FPS: {fps_display:.1f}"
                # Draw FPS background
                cv2.rectangle(
                    annotated_frame,
                    (frame_width - 140, 15),
                    (frame_width - 10, 50),
                    (0, 0, 0),
                    -1
                )
                cv2.putText(
                    annotated_frame,
                    fps_text,
                    (frame_width - 130, 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),  # Yellow
                    2
                )
            
            # Display the frame
            cv2.imshow('Teeth Braces Detection - Enhanced', annotated_frame)
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        
        # Quit on 'q' or ESC
        if key == ord('q') or key == 27:  # 27 is ESC
            print("\nQuitting...")
            break
        
        # Take screenshot on 's'
        elif key == ord('s'):
            screenshot_path = os.path.join(
                screenshot_dir,
                f"screenshot_{frame_count}.jpg"
            )
            cv2.imwrite(screenshot_path, annotated_frame)
            print(f"Screenshot saved: {screenshot_path}")
        
        # Toggle confidence bars on 'h'
        elif key == ord('h'):
            show_conf_bar = not show_conf_bar
            print(f"Confidence bars: {'ON' if show_conf_bar else 'OFF'}")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    print("\nWebcam detection stopped.")
    print(f"Total frames processed: {frame_count}")


def main():
    """
    Main function for running webcam detection from command line.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Real-time Teeth Braces Detection using Webcam - Enhanced Version',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run detection on default webcam with high accuracy
  python webcam_detect.py
  
  # Run with custom confidence threshold
  python webcam_detect.py --conf 0.5
  
  # Use higher image size for better accuracy
  python webcam_detect.py --imgsz 1280
  
  # Use specific camera
  python webcam_detect.py --device 1
  
  # Hide FPS counter
  python webcam_detect.py --no-fps
  
  # Hide confidence bars
  python webcam_detect.py --no-conf-bar
        """
    )
    
    parser.add_argument(
        '--weights', '-w',
        type=str,
        default='runs/detect/train/weights/best.pt',
        help='Path to model weights (default: runs/detect/train/weights/best.pt)'
    )
    
    parser.add_argument(
        '--device', '-d',
        type=int,
        default=0,
        help='Webcam device index (default: 0)'
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
        '--no-fps',
        action='store_true',
        help='Hide FPS counter'
    )
    
    parser.add_argument(
        '--no-conf-bar',
        action='store_true',
        help='Hide confidence bars'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Teeth Braces Detection - Real-time Webcam (Enhanced)")
    print("=" * 60)
    print(f"Webcam Index: {args.device}")
    print(f"Confidence Threshold: {args.conf}")
    print(f"IoU Threshold: {args.iou}")
    print(f"Image Size: {args.imgsz}")
    print(f"Show FPS: {not args.no_fps}")
    print(f"Show Confidence Bars: {not args.no_conf_bar}")
    
    # Load the model
    model = load_model(args.weights)
    
    # Run webcam detection
    try:
        run_webcam_detection(
            model=model,
            camera_index=args.device,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            show_fps=not args.no_fps,
            show_conf_bar=not args.no_conf_bar
        )
    except KeyboardInterrupt:
        print("\n\nDetection interrupted by user.")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

