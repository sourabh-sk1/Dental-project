"""
Enhanced YOLOv8 Training Script for Teeth Braces Detection
=============================================================
This script trains a YOLOv8 model with optimized hyperparameters for
improved accuracy and confidence in detecting orthodontic braces.

Key Improvements Over Current Model:
- Uses yolov8l.pt (large) model for better small object detection
- 250 epochs with cosine learning rate schedule
- 960 image size for improved detail capture
- Advanced augmentation for small object generalization
- AdamW optimizer with appropriate learning rate

Why Larger Model is Better for Small Objects:
1. Larger models have more parameters to learn finer details
2. Larger receptive fields capture more context
3. Better feature extraction for small, detailed objects like braces brackets
4. Higher capacity to distinguish between correct/incorrect placement

Usage:
    python train.py

Command Line:
    python train.py --model yolov8l.pt --epochs 250 --imgsz 960 --batch 16
"""

import os
import sys
from pathlib import Path
import argparse
from ultralytics import YOLO


def train_model(
    data_yaml: str = 'data.yaml',
    model_name: str = 'yolov8l.pt',
    epochs: int = 250,
    imgsz: int = 960,
    batch: int = 16,
    project: str = 'runs/detect',
    name: str = 'train',
    exist_ok: bool = False,
    device: str = '0',
    patience: int = 50,
    save: bool = True,
    plots: bool = True,
    workers: int = 8,
    # Learning rate settings
    lr0: float = 0.001,
    lrf: float = 0.01,
    cos_lr: bool = True,
    # Optimizer settings
    optimizer: str = 'AdamW',
    weight_decay: float = 0.0005,
    # Loss function weights
    box: float = 7.5,
    cls: float = 0.5,
    dfl: float = 1.5,
    # Augmentation settings - optimized for small object detection
    hsv_h: float = 0.015,
    hsv_s: float = 0.7,
    hsv_v: float = 0.4,
    degrees: float = 10.0,
    translate: float = 0.1,
    scale: float = 0.5,
    flipud: float = 0.0,
    fliplr: float = 0.5,
    mosaic: float = 1.0,
    mixup: float = 0.1,
    copy_paste: float = 0.1,
    # Training stability
    amp: bool = True,
    close_mosaic: int = 10,
    val: bool = True,
    verbose: bool = True,
):
    """
    Train YOLOv8 model for braces detection with optimized settings.
    
    Args:
        data_yaml: Path to dataset configuration file
        model_name: Name of pretrained YOLOv8 model
        epochs: Number of training epochs
        imgsz: Input image size for training
        batch: Batch size for training
        project: Directory to save training results
        name: Name of the training run
        exist_ok: Whether to overwrite existing results
        device: Device to use for training ('cpu', '0', '0,1,2,3', 'mps')
        patience: Early stopping patience (epochs without improvement)
        save: Whether to save trained model checkpoints
        plots: Whether to generate training plots
        workers: Number of data loading workers
        lr0: Initial learning rate
        lrf: Final learning rate factor
        cos_lr: Use cosine learning rate schedule
        optimizer: Optimizer type ('AdamW', 'SGD', 'Adam')
        weight_decay: L2 regularization
        box: Box loss weight
        cls: Classification loss weight
        dfl: DFL loss weight
        hsv_h: HSV augmentation - Hue
        hsv_s: HSV augmentation - Saturation
        hsv_v: HSV augmentation - Value
        degrees: Random rotation
        translate: Random translation
        scale: Random scaling
        flipud: Vertical flip probability
        fliplr: Horizontal flip probability
        mosaic: Mosaic augmentation probability
        mixup: Mixup augmentation probability
        copy_paste: Copy-paste augmentation probability
        amp: Automatic mixed precision
        close_mosaic: Disable mosaic in last N epochs
        val: Validate during training
        verbose: Detailed output
        
    Returns:
        Trained YOLO model and results
    """
    print("=" * 70)
    print("Teeth Braces Placement Detection - Enhanced YOLOv8 Training")
    print("=" * 70)
    print(f"\n📊 Training Configuration:")
    print(f"   Model: {model_name}")
    print(f"   Epochs: {epochs}")
    print(f"   Image Size: {imgsz}")
    print(f"   Batch Size: {batch}")
    print(f"   Optimizer: {optimizer}")
    print(f"   Learning Rate: {lr0}")
    print(f"   Cos LR Schedule: {cos_lr}")
    print(f"   Workers: {workers}")
    
    print(f"\n🔧 Augmentation Settings (Optimized for Small Objects):")
    print(f"   HSV: H={hsv_h}, S={hsv_s}, V={hsv_v}")
    print(f"   Degrees: {degrees}, Translate: {translate}, Scale: {scale}")
    print(f"   Mosaic: {mosaic}, Mixup: {mixup}, Copy-Paste: {copy_paste}")
    
    # Check device availability
    if device == '0':
        try:
            import torch
            if not torch.cuda.is_available():
                print("\n⚠️ CUDA not available, falling back to CPU")
                device = 'cpu'
        except:
            print("\n⚠️ CUDA check failed, using CPU")
            device = 'cpu'
    
    # Check for Apple Silicon MPS
    if device == 'mps':
        try:
            import torch
            if not torch.backends.mps.is_available():
                print("\n⚠️ MPS not available, falling back to CPU")
                device = 'cpu'
        except:
            pass
    
    # Determine dataset path
    data_path = Path(data_yaml)
    if not data_path.exists():
        # Try alternative paths
        alt_paths = [
            Path('../braces_dataset_fixed_yolov8/data.yaml'),
            Path('braces_dataset_fixed_yolov8/data.yaml'),
            Path(__file__).parent.parent / 'braces_dataset_fixed_yolov8' / 'data.yaml'
        ]
        for alt_path in alt_paths:
            if alt_path.exists():
                data_yaml = str(alt_path)
                print(f"\n📂 Using dataset: {data_yaml}")
                break
    
    # Load pretrained YOLOv8 model
    # Using yolov8l.pt (large) - significantly better for small object detection
    # Compared to yolov8n (nano) and yolov8s (small):
    # - More parameters: 43.7M vs 3.2M (n) vs 11.2M (s)
    # - Better feature extraction layers
    # - Higher capacity for learning fine details
    print(f"\n🔄 Loading pretrained model: {model_name}")
    model = YOLO(model_name)
    
    # Training configuration with optimized hyperparameters
    print("\n🚀 Starting training...")
    
    results = model.train(
        # Dataset and basic settings
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        exist_ok=exist_ok,
        device=device,
        workers=workers,
        verbose=verbose,
        
        # Save settings
        save=save,
        save_period=10,  # Save every 10 epochs
        plots=plots,
        
        # Early stopping
        patience=patience,
        close_mosaic=close_mosaic,
        
        # Learning rate settings - optimized for better convergence
        lr0=lr0,
        lrf=lrf,
        cos_lr=cos_lr,
        
        # Optimizer settings
        optimizer=optimizer,
        weight_decay=weight_decay,
        momentum=0.937,  # SGD momentum
        
        # Loss function weights - balanced for classification
        box=box,
        cls=cls,
        dfl=dfl,
        
        # Data augmentation settings - ENHANCED for small object detection
        # HSV augmentation helps with color variations in dental images
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        
        # Geometric augmentation
        degrees=degrees,
        translate=translate,
        scale=scale,
        flipud=flipud,
        fliplr=fliplr,
        
        # Advanced augmentation - CRITICAL for small objects like braces
        mosaic=mosaic,        # Combine 4 images - helps learn context
        mixup=mixup,          # Blend images - reduces overfitting
        copy_paste=copy_paste, # Copy-paste objects - increases variety
        
        # Training stability
        amp=amp,              # Automatic mixed precision
        val=val,              # Validate during training
    )
    
    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print("=" * 70)
    print(f"\n📁 Best model saved to: {project}/{name}/weights/best.pt")
    print(f"📁 Last model saved to: {project}/{name}/weights/last.pt")
    
    # Print training metrics if available
    if hasattr(results, 'results_dict'):
        print("\n📈 Final Training Metrics:")
        metrics = results.results_dict
        if 'metrics/mAP50(B)' in metrics:
            print(f"   mAP@50: {metrics.get('metrics/mAP50(B)', 0):.4f}")
        if 'metrics/mAP50-95(B)' in metrics:
            print(f"   mAP@50-95: {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
        if 'metrics/precision(B)' in metrics:
            print(f"   Precision: {metrics.get('metrics/precision(B)', 0):.4f}")
        if 'metrics/recall(B)' in metrics:
            print(f"   Recall: {metrics.get('metrics/recall(B)', 0):.4f}")
    
    return model, results


def validate_model(weights_path: str = 'runs/detect/train/weights/best.pt'):
    """
    Validate trained model on validation set with detailed metrics.
    
    Args:
        weights_path: Path to trained model weights
        
    Returns:
        Validation metrics dictionary
    """
    print("\n" + "=" * 70)
    print("📊 Running Model Validation")
    print("=" * 70)
    
    # Check if weights exist
    if not os.path.exists(weights_path):
        print(f"❌ Error: Model weights not found at {weights_path}")
        # Try alternative paths
        alt_paths = [
            'runs/detect/train/weights/best.pt',
            '../braces_dataset_fixed_yolov8/runs/detect/train/weights/best.pt',
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                weights_path = alt_path
                print(f"📂 Using alternative path: {weights_path}")
                break
        else:
            return None
    
    # Load trained model
    model = YOLO(weights_path)
    
    # Run validation with detailed metrics
    metrics = model.val(
        plots=True,
        verbose=True
    )
    
    # Print detailed metrics
    print("\n📈 Validation Results:")
    print("-" * 50)
    print(f"  mAP@50:           {metrics.box.map50:.4f}")
    print(f"  mAP@50-95:        {metrics.box.map:.4f}")
    print(f"  Precision:       {metrics.box.mp:.4f}")
    print(f"  Recall:           {metrics.box.mr:.4f}")
    
    # Per-class metrics
    if hasattr(metrics, 'box') and hasattr(metrics.box, 'ap_class_index'):
        print("\n📋 Per-Class Metrics:")
        class_indices = metrics.box.ap_class_index
        if len(class_indices) > 0:
            for i, idx in enumerate(class_indices):
                if i < len(metrics.box.ap):
                    class_name = "Correct Brace" if idx == 0 else "Incorrect Brace"
                    print(f"  {class_name}: AP={metrics.box.ap[i]:.4f}")
    
    return metrics


def export_model(weights_path: str, format: str = 'onnx'):
    """
    Export trained model to different formats.
    
    Args:
        weights_path: Path to trained model weights
        format: Export format ('onnx', 'torchscript', 'coreml', 'tflite', etc.)
        
    Returns:
        Path to exported model
    """
    print(f"\n🔄 Exporting model to {format} format...")
    
    model = YOLO(weights_path)
    exported_path = model.export(format=format)
    
    print(f"✅ Model exported to: {exported_path}")
    
    return exported_path


def main():
    """
    Main function for running training from command line.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Enhanced YOLOv8 Training for Teeth Braces Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings (recommended)
  python train.py
  
  # Train with custom settings
  python train.py --model yolov8x.pt --epochs 300 --imgsz 1280
  
  # Train on CPU (slower but works)
  python train.py --device cpu --epochs 100 --batch 8
  
  # Train on Apple Silicon
  python train.py --device mps --epochs 200
  
  # Validate trained model
  python train.py --validate --weights runs/detect/train/weights/best.pt
  
  # Export model to ONNX
  python train.py --export --weights runs/detect/train/weights/best.pt --format onnx
        """
    )
    
    # Model settings
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='yolov8l.pt',
        choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
        help='YOLOv8 model to use (default: yolov8l.pt - Large model for small objects)'
    )
    
    # Training settings
    parser.add_argument(
        '--data', '-d',
        type=str,
        default='data.yaml',
        help='Path to dataset YAML file (default: data.yaml)'
    )
    
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=250,
        help='Number of training epochs (default: 250)'
    )
    
    parser.add_argument(
        '--imgsz', '-i',
        type=int,
        default=960,
        help='Input image size (default: 960 - good balance for small objects)'
    )
    
    parser.add_argument(
        '--batch', '-b',
        type=int,
        default=16,
        help='Batch size (default: 16)'
    )
    
    # Device settings
    parser.add_argument(
        '--device', '-dev',
        type=str,
        default='0',
        help='Device to use: cpu, 0, 1, 2, 3, or mps (default: 0 for GPU)'
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=8,
        help='Number of data loading workers (default: 8)'
    )
    
    # Learning rate settings
    parser.add_argument(
        '--lr0',
        type=float,
        default=0.001,
        help='Initial learning rate (default: 0.001)'
    )
    
    parser.add_argument(
        '--lrf',
        type=float,
        default=0.01,
        help='Final learning rate factor (default: 0.01)'
    )
    
    parser.add_argument(
        '--cos-lr',
        action='store_true',
        default=True,
        help='Use cosine learning rate schedule (default: True)'
    )
    
    # Optimizer settings
    parser.add_argument(
        '--optimizer',
        type=str,
        default='AdamW',
        choices=['AdamW', 'SGD', 'Adam'],
        help='Optimizer to use (default: AdamW)'
    )
     
    # Project settings
    parser.add_argument(
        '--project',
        type=str,
        default='runs/detect',
        help='Project directory (default: runs/detect)'
    )
    
    parser.add_argument(
        '--name', '-n',
        type=str,
        default='train',
        help='Experiment name (default: train)'
    )
    
    # Action flags
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run validation on trained model'
    )
    
    parser.add_argument(
        '--export',
        action='store_true',
        help='Export trained model'
    )
    
    parser.add_argument(
        '--weights',
        type=str,
        default='runs/detect/train/weights/best.pt',
        help='Path to model weights for validation/export'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        default='onnx',
        choices=['onnx', 'torchscript', 'coreml', 'tflite', 'saved_model'],
        help='Export format (default: onnx)'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print("\n" + "=" * 70)
    print("🦷 Teeth Braces Detection - Enhanced Training Pipeline")
    print("=" * 70)
    
    # Run validation if requested
    if args.validate:
        validate_model(args.weights)
        return
    
    # Export model if requested
    if args.export:
        export_model(args.weights, args.format)
        return
    
    # Otherwise, run training
    print("\n🎯 Starting training with the following configuration:")
    print(f"   Model: {args.model}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Image Size: {args.imgsz}")
    print(f"   Batch Size: {args.batch}")
    print(f"   Device: {args.device}")
    print(f"   Optimizer: {args.optimizer}")
    
    # Train the model
    model, results = train_model(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        lr0=args.lr0,
        lrf=args.lrf,
        cos_lr=args.cos_lr,
        optimizer=args.optimizer,
        project=args.project,
        name=args.name,
    )
    
    # After training, validate the model
    print("\n🔍 Running validation on trained model...")
    weights_path = f"{args.project}/{args.name}/weights/best.pt"
    validate_model(weights_path)
    
    print("\n" + "=" * 70)
    print("🎉 All tasks completed successfully!")
    print("=" * 70)
    print("\n📌 Next Steps:")
    print("   1. Run detection: python detect.py --image path/to/image.jpg")
    print("   2. Run web app: streamlit run streamlit_app.py")
    print(f"   3. Model saved at: {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()

