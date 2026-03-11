"""
Evaluation Script for Teeth Braces Detection Model
=====================================================
This script evaluates the trained YOLOv8 model on the validation/test set
and provides comprehensive metrics including:

- mAP@50 (mean Average Precision at IoU=0.5)
- mAP@50-95 (mean Average Precision at IoU=0.5:0.95)
- Precision
- Recall
- Confusion Matrix

Usage:
    python evaluate.py
    python evaluate.py --weights runs/detect/train/weights/best.pt
    python evaluate.py --data data.yaml --split test
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

# Add utils to path
sys.path.append(str(Path(__file__).parent))

from utils.tooth_mapping import CLASS_NAMES


# =============================================================================
# CONSTANTS
# =============================================================================

CLASS_NAMES_MAP = {
    0: 'Correct Brace',
    1: 'Incorrect Brace',
}

# Color scheme for plots
COLORS = {
    'correct': '#10B981',    # Green
    'incorrect': '#EF4444',  # Red
}


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def load_model(weights_path: str = 'runs/detect/train/weights/best.pt'):
    """
    Load the trained YOLO model.
    
    Args:
        weights_path: Path to model weights
        
    Returns:
        Loaded YOLO model
    """
    if not os.path.exists(weights_path):
        # Try alternative paths
        alt_paths = [
            'runs/detect/train/weights/best.pt',
            '../braces_dataset_fixed_yolov8/runs/detect/train/weights/best.pt',
        ]
        for w in alt_paths:
            if os.path.exists(w):
                weights_path = w
                break
        else:
            print(f"❌ Error: Model weights not found at {weights_path}")
            return None
    
    print(f"🤖 Loading model from: {weights_path}")
    model = YOLO(weights_path)
    return model


def evaluate_model(
    model,
    data_yaml: str = 'data.yaml',
    imgsz: int = 960,
    conf: float = 0.40,
    iou: float = 0.45,
    split: str = 'valid'
):
    """
    Evaluate the model on validation/test set.
    
    Args:
        model: Loaded YOLO model
        data_yaml: Path to dataset YAML
        imgsz: Image size for evaluation
        conf: Confidence threshold
        iou: IoU threshold
        split: Dataset split to evaluate ('valid' or 'test')
        
    Returns:
        Evaluation metrics dictionary
    """
    print(f"\n📊 Evaluating model on {split} set...")
    
    # Run validation
    metrics = model.val(
        data=data_yaml,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        split=split,
        plots=True,
        verbose=True
    )
    
    return metrics


def compute_confusion_matrix(
    model,
    data_yaml: str = 'data.yaml',
    imgsz: int = 960,
    conf: float = 0.40,
    iou: float = 0.45,
    split: str = 'valid'
):
    """
    Compute confusion matrix for the model predictions.
    
    Args:
        model: Loaded YOLO model
        data_yaml: Path to dataset YAML
        imgsz: Image size
        conf: Confidence threshold
        iou: IoU threshold
        split: Dataset split
        
    Returns:
        Confusion matrix array
    """
    print("\n🔄 Computing confusion matrix...")
    
    # Load dataset info
    import yaml
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    
    # Get image directory
    if split == 'test':
        img_dir = Path(data['test'])
    else:
        img_dir = Path(data['val'])
    
    # Make path absolute
    if not img_dir.is_absolute():
        data_dir = Path(data_yaml).parent
        img_dir = data_dir / img_dir
    
    # Get all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in img_dir.glob('*') if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"⚠️ No images found in {img_dir}")
        return None
    
    print(f"📁 Found {len(image_files)} images in {split} set")
    
    # Collect predictions and ground truth
    y_true = []
    y_pred = []
    
    for img_path in image_files[:100]:  # Limit for speed
        # Get corresponding label file
        label_file = img_path.with_suffix('.txt')
        
        if not label_file.exists():
            continue
        
        # Read ground truth
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        true_classes = []
        for line in lines:
            parts = line.strip().split()
            if parts:
                cls = int(parts[0])
                true_classes.append(cls)
        
        # Run prediction
        results = model(img_path, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
        result = results[0]
        
        pred_classes = []
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls = int(box.cls[0].cpu().numpy())
                pred_classes.append(cls)
        
        # Add to lists (using majority voting for multi-object images)
        if true_classes:
            y_true.append(max(true_classes, key=true_classes.count))
        if pred_classes:
            y_pred.append(max(pred_classes, key=pred_classes.count))
    
    # Compute confusion matrix
    if y_true and y_pred:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    else:
        cm = np.zeros((2, 2), dtype=int)
    
    return cm


def plot_confusion_matrix(cm: np.ndarray, save_path: str = 'confusion_matrix.png'):
    """
    Plot and save confusion matrix.
    
    Args:
        cm: Confusion matrix array
        save_path: Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    
    # Create labels
    labels = ['Correct Brace', 'Incorrect Brace']
    
    # Plot
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                annot_kws={'size': 14})
    
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Confusion matrix saved to: {save_path}")


def plot_metrics(metrics, save_path: str = 'metrics.png'):
    """
    Plot evaluation metrics.
    
    Args:
        metrics: YOLO metrics object
        save_path: Path to save the plot
    """
    # Extract metrics
    map50 = metrics.box.map50
    map = metrics.box.map
    precision = metrics.box.mp
    recall = metrics.box.mr
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart for metrics
    metric_names = ['mAP@50', 'mAP@50-95', 'Precision', 'Recall']
    metric_values = [map50, map, precision, recall]
    colors = ['#667eea', '#764ba2', '#10B981', '#F59E0B']
    
    axes[0].bar(metric_names, metric_values, color=colors, edgecolor='white', linewidth=2)
    axes[0].set_ylim(0, 1)
    axes[0].set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Score', fontsize=12)
    
    # Add value labels on bars
    for i, v in enumerate(metric_values):
        axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=11, fontweight='bold')
    
    # Radar chart for metrics
    angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
    values = metric_values + [metric_values[0]]  # Complete the loop
    angles += angles[:1]
    
    axes[1].polar()
    axes[1].fill(angles, values, color='#667eea', alpha=0.25)
    axes[1].plot(angles, values, color='#667eea', linewidth=2)
    axes[1].set_xticks(angles[:-1])
    axes[1].set_xticklabels(metric_names)
    axes[1].set_ylim(0, 1)
    axes[1].set_title('Performance Radar', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Metrics plot saved to: {save_path}")


def print_detailed_report(metrics, cm: np.ndarray = None):
    """
    Print detailed evaluation report.
    
    Args:
        metrics: YOLO metrics object
        cm: Confusion matrix
    """
    print("\n" + "=" * 70)
    print("🦷 TEETH BRACES DETECTION - EVALUATION REPORT")
    print("=" * 70)
    
    # Main metrics
    print("\n📈 MAIN METRICS:")
    print("-" * 50)
    print(f"  mAP@50:           {metrics.box.map50:.4f}")
    print(f"  mAP@50-95:        {metrics.box.map:.4f}")
    print(f"  Precision:        {metrics.box.mp:.4f}")
    print(f"  Recall:           {metrics.box.mr:.4f}")
    
    # Per-class metrics
    print("\n📋 PER-CLASS METRICS:")
    print("-" * 50)
    
    # Get per-class precision/recall if available
    if hasattr(metrics.box, 'ap_class_index'):
        class_indices = metrics.box.ap_class_index
        if len(class_indices) > 0:
            for i, idx in enumerate(class_indices):
                class_name = CLASS_NAMES_MAP.get(idx, f'Class {idx}')
                ap = metrics.box.ap[i] if i < len(metrics.box.ap) else 0
                print(f"  {class_name}:")
                print(f"    AP: {ap:.4f}")
    
    # Confusion matrix
    if cm is not None:
        print("\n📊 CONFUSION MATRIX:")
        print("-" * 50)
        print(f"                    Predicted")
        print(f"                  Correct  Incorrect")
        print(f"  Actual Correct:   {cm[0,0]:5d}    {cm[0,1]:5d}")
        print(f"  Actual Incorrect:{cm[1,0]:5d}    {cm[1,1]:5d}")
        
        # Calculate additional metrics from confusion matrix
        if cm.sum() > 0:
            tn, fp, fn, tp = cm.ravel()
            
            # Accuracy
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            
            # Specificity
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # F1 Score
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"\n  Additional Metrics:")
            print(f"    Accuracy:    {accuracy:.4f}")
            print(f"    Specificity:{specificity:.4f}")
            print(f"    F1 Score:   {f1:.4f}")
    
    print("\n" + "=" * 70)
    print("✅ Evaluation Complete!")
    print("=" * 70)


def export_report(metrics, cm: np.ndarray, output_path: str = 'evaluation_report.csv'):
    """
    Export evaluation report to CSV.
    
    Args:
        metrics: YOLO metrics object
        cm: Confusion matrix
        output_path: Path to save CSV
    """
    # Create report data
    report = {
        'Metric': ['mAP@50', 'mAP@50-95', 'Precision', 'Recall'],
        'Value': [
            metrics.box.map50,
            metrics.box.map,
            metrics.box.mp,
            metrics.box.mr
        ]
    }
    
    df = pd.DataFrame(report)
    df.to_csv(output_path, index=False)
    
    print(f"📊 Evaluation report saved to: {output_path}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """
    Main function for running evaluation.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Evaluate Teeth Braces Detection Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate with default settings
  python evaluate.py
  
  # Evaluate on test set
  python evaluate.py --split test
  
  # Evaluate with custom weights
  python evaluate.py --weights runs/detect/train/weights/best.pt
  
  # Evaluate with custom image size
  python evaluate.py --imgsz 1280
        """
    )
    
    parser.add_argument(
        '--weights', '-w',
        type=str,
        default='runs/detect/train/weights/best.pt',
        help='Path to model weights'
    )
    
    parser.add_argument(
        '--data', '-d',
        type=str,
        default='data.yaml',
        help='Path to dataset YAML file'
    )
    
    parser.add_argument(
        '--imgsz', '-i',
        type=int,
        default=960,
        help='Image size for evaluation'
    )
    
    parser.add_argument(
        '--conf', '-c',
        type=float,
        default=0.40,
        help='Confidence threshold'
    )
    
    parser.add_argument(
        '--iou',
        type=float,
        default=0.45,
        help='IoU threshold'
    )
    
    parser.add_argument(
        '--split', '-s',
        type=str,
        default='valid',
        choices=['valid', 'test'],
        help='Dataset split to evaluate'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='.',
        help='Output directory for plots and reports'
    )
    
    args = parser.parse_args()
    
    # Determine dataset path
    data_yaml = args.data
    if not os.path.exists(data_yaml):
        alt_paths = [
            '../braces_dataset_fixed_yolov8/data.yaml',
            'braces_dataset_fixed_yolov8/data.yaml',
        ]
        for p in alt_paths:
            if os.path.exists(p):
                data_yaml = p
                print(f"📂 Using dataset: {data_yaml}")
                break
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print banner
    print("\n" + "=" * 70)
    print("🦷 Teeth Braces Detection - Model Evaluation")
    print("=" * 70)
    print(f"\n🔧 Configuration:")
    print(f"   Weights: {args.weights}")
    print(f"   Dataset: {data_yaml}")
    print(f"   Image Size: {args.imgsz}")
    print(f"   Confidence: {args.conf}")
    print(f"   IoU: {args.iou}")
    print(f"   Split: {args.split}")
    
    # Load model
    model = load_model(args.weights)
    
    if model is None:
        print("❌ Failed to load model")
        return
    
    # Evaluate model
    print("\n🚀 Starting evaluation...")
    metrics = evaluate_model(
        model,
        data_yaml=data_yaml,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        split=args.split
    )
    
    # Compute confusion matrix
    cm = compute_confusion_matrix(
        model,
        data_yaml=data_yaml,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        split=args.split
    )
    
    # Save plots
    if cm is not None:
        plot_confusion_matrix(cm, str(output_dir / 'confusion_matrix.png'))
    
    plot_metrics(metrics, str(output_dir / 'metrics.png'))
    
    # Print detailed report
    print_detailed_report(metrics, cm)
    
    # Export report
    if cm is not None:
        export_report(metrics, cm, str(output_dir / 'evaluation_report.csv'))
    
    print("\n✅ Evaluation complete!")
    print(f"📁 Output files saved to: {output_dir}")


if __name__ == "__main__":
    main()

