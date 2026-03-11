"""
YOLOv8 Training Script for Teeth Braces Placement Detection
=============================================================
This script trains a YOLOv8 model to detect whether orthodontic 
teeth braces brackets are correctly or incorrectly placed.

The model distinguishes between:
- Class 0: correct_brace - Properly positioned brackets
- Class 1: incorrect_brace - Misaligned or improperly positioned brackets

Usage:
    python train.py
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import YOLOv8 from Ultralytics
from ultralytics import YOLO


def train_model(
    data_yaml: str = 'data.yaml',
    model_name: str = 'yolov8s.pt',
    epochs: int = 150,
    imgsz: int = 1280,
    batch: int = 8,
    project: str = 'runs/detect',
    name: str = 'train',
    exist_ok: bool = False,
    device: str = 'cpu',
    patience: int = 50,
    save: bool = True,
    plots: bool = True,
    lr0: float = 0.0001,
    lrf: float = 0.001,
    optimizer: str = 'AdamW',
    weight_decay: float = 0.0005,
    box: float = 7.5,
    cls: float = 0.5,
    dfl: float = 1.5,
    degrees: float = 15.0,
    translate: float = 0.15,
    scale: float = 0.6,
    flipud: float = 0.0,
    fliplr: float = 0.5,
    mosaic: float = 1.0,
    mixup: float = 0.15,
    copy_paste: float = 0.1,
    amp: bool = True,
):
    """
    Train YOLOv8 model for braces detection.
    
    Args:
        data_yaml: Path to dataset configuration file
        model_name: Name of pretrained YOLOv8 model (yolov8n.pt, yolov8s.pt, etc.)
        epochs: Number of training epochs
        imgsz: Input image size for training
        batch: Batch size for training
        project: Directory to save training results
        name: Name of the training run
        exist_ok: Whether to overwrite existing results
        device: Device to use for training ('cpu', '0', '0,1,2,3')
        patience: Early stopping patience (epochs without improvement)
        save: Whether to save trained model checkpoints
        plots: Whether to generate training plots
        
    Returns:
        Trained YOLO model
    """
    print("=" * 60)
    print("Teeth Braces Placement Detection - YOLOv8 Training")
    print("=" * 60)
    
    # Load pretrained YOLOv8 model
    # Using yolov8n.pt (nano) - smallest and fastest version
    # Other options: yolov8s.pt (small), yolov8m.pt (medium), yolov8l.pt (large)
    print(f"\nLoading pretrained model: {model_name}")
    model = YOLO(model_name)
    
    # Training configuration
    # These parameters are optimized for braces detection - HIGHER ACCURACY
    results = model.train(
        data=data_yaml,           # Dataset configuration file
        epochs=epochs,            # Number of training epochs
        imgsz=imgsz,              # Input image size (1280 for better accuracy)
        batch=batch,              # Batch size
        project=project,          # Save directory
        name=name,                # Experiment name
        exist_ok=exist_ok,        # Overwrite existing
        device=device,            # Computing device
        patience=patience,        # Early stopping
        save=save,                # Save checkpoints
        plots=plots,              # Generate plots
        verbose=True,              # Detailed output
        # Save directory fix - prevent nested directories
        save_dir=f"{project}/{name}",
        # Learning rate settings - optimized for better convergence
        lr0=lr0,                 # Initial learning rate
        lrf=lrf,                 # Final learning rate factor
        # Optimizer settings
        optimizer=optimizer,     # AdamW optimizer
        weight_decay=weight_decay, # L2 regularization
        # Loss function weights - balanced for classification
        box=box,                 # Box loss weight
        cls=cls,                 # Classification loss weight
        dfl=dfl,                 # DFL loss weight
        # Data augmentation settings - enhanced for better generalization
        degrees=degrees,         # Random rotation
        translate=translate,     # Random translation
        scale=scale,             # Random scaling
        flipud=flipud,           # No vertical flip
        fliplr=fliplr,           # Horizontal flip
        mosaic=mosaic,           # Mosaic augmentation
        mixup=mixup,             # Mixup augmentation
        copy_paste=copy_paste,   # Copy-paste augmentation
        # Training stability
        amp=amp,                 # Automatic mixed precision
    )
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nBest model saved to: runs/detect/{name}/weights/best.pt")
    print(f"Last model saved to: runs/detect/{name}/weights/last.pt")
    
    return model


def validate_model(weights_path: str = 'runs/detect/train/weights/best.pt'):
    """
    Validate trained model on validation set.
    
    Args:
        weights_path: Path to trained model weights
    """
    print("\n" + "=" * 60)
    print("Running Validation")
    print("=" * 60)
    
    # Load trained model
    model = YOLO(weights_path)
    
    # Run validation
    metrics = model.val()
    
    # Print metrics
    print(f"\nValidation Results:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall: {metrics.box.mr:.4f}")
    
    return metrics


def export_model(weights_path: str = 'runs/detect/train/weights/best.pt', format: str = 'onnx'):
    """
    Export trained model to different formats.
    
    Args:
        weights_path: Path to trained model weights
        format: Export format (onnx, torchscript, coreml, etc.)
    """
    print(f"\nExporting model to {format} format...")
    
    model = YOLO(weights_path)
    exported_path = model.export(format=format)
    
    print(f"Model exported to: {exported_path}")
    
    return exported_path


if __name__ == "__main__":
    # Training parameters - Optimized for BETTER ACCURACY and HIGH CONFIDENCE
    CONFIG = {
        'data_yaml': 'data.yaml',
        'model_name': 'yolov8s.pt',     # Use yolov8s.pt (small) for better accuracy - larger than nano
        'epochs': 150,                  # More epochs = better learning (increased from 100)
        'imgsz': 1280,                 # Higher resolution for better accuracy (was 640)
        'batch': 8,                    # Smaller batch for better generalization
        'device': 'cpu',               # Use '0' for GPU, 'mps' for Apple Silicon
        'patience': 50,
        'lr0': 0.0001,                # Much lower learning rate for fine-tuning (was 0.001)
        'lrf': 0.001,                 # Final learning rate factor
        # Optimizer settings for better convergence
        'optimizer': 'AdamW',          # AdamW optimizer for better generalization
        'weight_decay': 0.0005,       # L2 regularization
        # Loss function weights for balanced learning
        'box': 7.5,                    # Box loss weight
        'cls': 0.5,                    # Classification loss weight
        'dfl': 1.5,                   # DFL loss weight
        # Augmentation settings - enhanced for better generalization
        'degrees': 15,               # Random rotation (increased)
        'translate': 0.15,          # Random translation (increased)
        'scale': 0.6,                # Random scaling (increased)
        'flipud': 0.0,               # No vertical flip
        'fliplr': 0.5,               # Horizontal flip
        'mosaic': 1.0,               # Mosaic augmentation
        'mixup': 0.15,               # Mixup augmentation (added)
        'copy_paste': 0.1,           # Copy-paste augmentation (added)
        # Training stability
        'close_mosaic': 10,          # Disable mosaic in last 10 epochs
        'amp': True,                 # Automatic mixed precision (if available)
        # Validation settings
        'val': True,                 # Validate during training
        'plots': True,               # Generate training plots
    }
    
    print("\nTraining Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\nStarting training...")
    
    # Train the model
    model = train_model(**CONFIG)
    
    # After training, validate the model
    print("\nValidating trained model...")
    validate_model()
    
    print("\n" + "=" * 60)
    print("All tasks completed successfully!")
    print("=" * 60)
    print("\nTo run detection on an image:")
    print("  python detect.py --image path/to/image.jpg")
    print("\nTo run the web app:")
    print("  streamlit run app.py")

