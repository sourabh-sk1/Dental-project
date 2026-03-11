"""
Tooth Mapping Utilities
=======================
This module provides utilities for mapping bounding box positions to tooth names
and formatting detection results for display.

Since the current dataset only has 2 classes (correct/incorrect brace), we use
position-based inference to determine which tooth the brace is attached to.

Dental Arch Mapping Strategy:
- Upper Arch (top half of image): Upper teeth
- Lower Arch (bottom half of image): Lower teeth
- Left/Right: Based on horizontal position

Teeth in each quadrant (from center outward):
1. Central Incisor (CI) - closest to midline
2. Lateral Incisor (LI) - next
3. Canine - next
4. First Premolar (P1) - next
5. Second Premolar (P2) - farthest from midline
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


# =============================================================================
# TOOTH NAME MAPPINGS
# =============================================================================

# Full tooth names for display
TOOTH_NAMES = {
    # Upper arch - right side (patient's right, our left)
    'ur_ci': 'Upper Right Central Incisor',
    'ur_li': 'Upper Right Lateral Incisor',
    'ur_canine': 'Upper Right Canine',
    'ur_p1': 'Upper Right First Premolar',
    'ur_p2': 'Upper Right Second Premolar',
    # Upper arch - left side (patient's left, our right)
    'ul_ci': 'Upper Left Central Incisor',
    'ul_li': 'Upper Left Lateral Incisor',
    'ul_canine': 'Upper Left Canine',
    'ul_p1': 'Upper Left First Premolar',
    'ul_p2': 'Upper Left Second Premolar',
    # Lower arch - right side
    'lr_ci': 'Lower Right Central Incisor',
    'lr_li': 'Lower Right Lateral Incisor',
    'lr_canine': 'Lower Right Canine',
    'lr_p1': 'Lower Right First Premolar',
    'lr_p2': 'Lower Right Second Premolar',
    # Lower arch - left side
    'll_ci': 'Lower Left Central Incisor',
    'll_li': 'Lower Left Lateral Incisor',
    'll_canine': 'Lower Left Canine',
    'll_p1': 'Lower Left First Premolar',
    'll_p2': 'Lower Left Second Premolar',
}

# Short codes for compact display
TOOTH_SHORT_CODES = {
    'ur_ci': 'UR-CI', 'ur_li': 'UR-LI', 'ur_canine': 'UR-CAN',
    'ur_p1': 'UR-P1', 'ur_p2': 'UR-P2',
    'ul_ci': 'UL-CI', 'ul_li': 'UL-LI', 'ul_canine': 'UL-CAN',
    'ul_p1': 'UL-P1', 'ul_p2': 'UL-P2',
    'lr_ci': 'LR-CI', 'lr_li': 'LR-LI', 'lr_canine': 'LR-CAN',
    'lr_p1': 'LR-P1', 'lr_p2': 'LR-P2',
    'll_ci': 'LL-CI', 'll_li': 'LL-LI', 'll_canine': 'LL-CAN',
    'll_p1': 'LL-P1', 'll_p2': 'LL-P2',
}

# Human-readable short names (without arch prefix)
TOOTH_DISPLAY_NAMES = {
    'ur_ci': 'Central Incisor', 'ur_li': 'Lateral Incisor', 'ur_canine': 'Canine',
    'ur_p1': 'First Premolar', 'ur_p2': 'Second Premolar',
    'ul_ci': 'Central Incisor', 'ul_li': 'Lateral Incisor', 'ul_canine': 'Canine',
    'ul_p1': 'First Premolar', 'ul_p2': 'Second Premolar',
    'lr_ci': 'Central Incisor', 'lr_li': 'Lateral Incisor', 'lr_canine': 'Canine',
    'lr_p1': 'First Premolar', 'lr_p2': 'Second Premolar',
    'll_ci': 'Central Incisor', 'll_li': 'Lateral Incisor', 'll_canine': 'Canine',
    'll_p1': 'First Premolar', 'll_p2': 'Second Premolar',
}

# Arch information
TOOTH_ARCH = {
    'ur_ci': 'Upper', 'ur_li': 'Upper', 'ur_canine': 'Upper', 'ur_p1': 'Upper', 'ur_p2': 'Upper',
    'ul_ci': 'Upper', 'ul_li': 'Upper', 'ul_canine': 'Upper', 'ul_p1': 'Upper', 'ul_p2': 'Upper',
    'lr_ci': 'Lower', 'lr_li': 'Lower', 'lr_canine': 'Lower', 'lr_p1': 'Lower', 'lr_p2': 'Lower',
    'll_ci': 'Lower', 'll_li': 'Lower', 'll_canine': 'Lower', 'll_p1': 'Lower', 'll_p2': 'Lower',
}

# =============================================================================
# TOOTH TYPE ABBREVIATIONS
# =============================================================================

TOOTH_TYPE_CODES = {
    'ci': 'Central Incisor',
    'li': 'Lateral Incisor',
    'canine': 'Canine',
    'p1': 'First Premolar',
    'p2': 'Second Premolar',
}

# Mapping from position index to tooth type
TOOTH_POSITION_MAP = {
    0: 'ci',      # Central Incisor - closest to center
    1: 'li',      # Lateral Incisor
    2: 'canine',  # Canine
    3: 'p1',      # First Premolar
    4: 'p2',      # Second Premolar
}

# =============================================================================
# COLOR CONFIGURATION
# =============================================================================

# Bounding box colors (BGR format for OpenCV)
TOOTH_COLORS = {
    # Correct braces - Green shades
    'correct': (0, 255, 0),        # Bright green
    'correct_dark': (0, 180, 0),   # Darker green for glow effect
    # Incorrect braces - Red shades
    'incorrect': (0, 0, 255),      # Bright red
    'incorrect_dark': (0, 0, 180), # Darker red for glow effect
}

# Status colors for UI display (RGB)
UI_COLORS = {
    'correct': '#10B981',   # Green
    'incorrect': '#EF4444', # Red
    'warning': '#F59E0B',   # Amber
    'info': '#3B82F6',      # Blue
}

# =============================================================================
# POSITION-BASED TOOTH INFERENCE
# =============================================================================

def get_tooth_from_position(
    bbox: Tuple[float, float, float, float],
    image_shape: Tuple[int, int],
    quadrant_bias: str = 'balanced'
) -> str:
    """
    Infer the tooth type based on bounding box position in the image.
    
    The dental arch is divided into quadrants:
    - Upper: y < 0.5 of image height
    - Lower: y >= 0.5 of image height
    - Right (patient's right, our left): x < 0.5 of image width
    - Left (patient's left, our right): x >= 0.5 of image width
    
    Within each quadrant, teeth are ordered from center outward:
    Central Incisor -> Lateral Incisor -> Canine -> P1 -> P2
    
    Args:
        bbox: Bounding box in format (x_center, y_center, width, height)
              Values normalized to 0-1 range
        image_shape: Image dimensions (height, width)
        quadrant_bias: Bias for quadrant detection ('balanced', 'upper_left', 'upper_right', etc.)
    
    Returns:
        Tooth key string (e.g., 'ur_ci', 'ul_canine', 'lr_p1')
    
    Example:
        >>> bbox = (0.25, 0.3, 0.1, 0.15)  # x_center, y_center, width, height
        >>> image_shape = (640, 640)
        >>> tooth = get_tooth_from_position(bbox, image_shape)
        >>> print(tooth)  # 'ur_ci' (Upper Right Central Incisor)
    """
    x_center, y_center, width, height = bbox
    
    # Determine arch (upper/lower)
    if y_center < 0.5:
        arch = 'upper'
    else:
        arch = 'lower'
    
    # Determine side (left/right from patient's perspective)
    # Note: In medical imaging, patient's right appears on our left
    if x_center < 0.5:
        side = 'right'  # Patient's right = our left
    else:
        side = 'left'   # Patient's left = our right
    
    # Determine tooth type based on horizontal distance from center
    # Closer to center = Central Incisor, farther = Premolars
    distance_from_center = abs(x_center - 0.5)
    
    # Map distance to tooth type (5 teeth per quadrant)
    # Adjust thresholds based on typical dental arch proportions
    if distance_from_center < 0.08:
        tooth_type = 'ci'      # Central Incisor
    elif distance_from_center < 0.15:
        tooth_type = 'li'       # Lateral Incisor
    elif distance_from_center < 0.22:
        tooth_type = 'canine'   # Canine
    elif distance_from_center < 0.30:
        tooth_type = 'p1'       # First Premolar
    else:
        tooth_type = 'p2'       # Second Premolar
    
    # Build tooth key
    arch_prefix = 'u' if arch == 'upper' else 'l'
    side_prefix = 'r' if side == 'right' else 'l'
    
    tooth_key = f"{arch_prefix}{side_prefix}_{tooth_type}"
    
    return tooth_key


def get_tooth_from_yolo_box(box, image_shape: Tuple[int, int]) -> str:
    """
    Extract tooth type from YOLO box result.
    
    Args:
        box: YOLO box object with xywhn (normalized xywh) attribute
        image_shape: Image dimensions (height, width)
    
    Returns:
        Tooth key string
    """
    # Get normalized bounding box
    bbox = box.xywhn[0].cpu().numpy()  # x_center, y_center, width, height (normalized)
    return get_tooth_from_position(bbox, image_shape)


# =============================================================================
# FORMATTING FUNCTIONS
# =============================================================================

def format_detection_label(
    tooth_key: str,
    is_correct: bool,
    confidence: float,
    short: bool = True
) -> str:
    """
    Format a detection label for display.
    
    Args:
        tooth_key: Tooth key from get_tooth_from_position
        is_correct: Whether the brace placement is correct
        confidence: Confidence score (0-1)
        short: Whether to use short format
    
    Returns:
        Formatted label string
    
    Example:
        >>> label = format_detection_label('ur_ci', True, 0.92, short=True)
        >>> print(label)
        'UR-CI | Correct | 0.92'
    """
    # Get tooth name
    if short:
        tooth_name = TOOTH_SHORT_CODES.get(tooth_key, 'Unknown')
    else:
        tooth_name = TOOTH_DISPLAY_NAMES.get(tooth_key, 'Unknown Tooth')
    
    # Get status
    status = "Correct" if is_correct else "Incorrect"
    
    # Format confidence
    conf_str = f"{confidence:.2f}"
    
    # Build label
    if short:
        label = f"{tooth_name} | {status} | {conf_str}"
    else:
        arch = TOOTH_ARCH.get(tooth_key, 'Unknown')
        label = f"{arch} {tooth_name} | {status} Brace | Confidence: {conf_str}"
    
    return label


def format_tooth_name(tooth_key: str, short: bool = False) -> str:
    """
    Get human-readable tooth name.
    
    Args:
        tooth_key: Tooth key from get_tooth_from_position
        short: Whether to use short format
    
    Returns:
        Tooth name string
    """
    if short:
        return TOOTH_SHORT_CODES.get(tooth_key, 'Unknown')
    else:
        return TOOTH_DISPLAY_NAMES.get(tooth_key, 'Unknown Tooth')


def create_detection_summary(detections: List[Dict], image_shape: Tuple[int, int]) -> Dict:
    """
    Create a summary of all detections with tooth inference.
    
    Args:
        detections: List of detection dictionaries from YOLO
        image_shape: Image dimensions (height, width)
    
    Returns:
        Summary dictionary with counts and detailed detections
    """
    # Initialize counters
    summary = {
        'total_detections': len(detections),
        'correct_braces': 0,
        'incorrect_braces': 0,
        'upper_teeth': 0,
        'lower_teeth': 0,
        'detections': [],
        'warnings': []
    }
    
    # Process each detection
    for i, det in enumerate(detections):
        # Get class info
        class_name = det.get('class', '')
        is_correct = 'correct' in class_name.lower()
        
        # Get bounding box
        bbox = det.get('bbox', [0, 0, 0, 0])
        x_center = (bbox[0] + bbox[2]) / 2  # Convert from x1,y1,x2,y2 to x_center
        y_center = (bbox[1] + bbox[3]) / 2
        
        # Normalize to 0-1 range for tooth inference
        img_h, img_w = image_shape[:2]
        norm_bbox = (x_center / img_w, y_center / img_h, 
                     (bbox[2] - bbox[0]) / img_w, (bbox[3] - bbox[1]) / img_h)
        
        # Infer tooth from position
        tooth_key = get_tooth_from_position(norm_bbox, image_shape)
        
        # Get tooth name
        tooth_name = format_tooth_name(tooth_key, short=True)
        arch = TOOTH_ARCH.get(tooth_key, 'Unknown')
        
        # Update counters
        if is_correct:
            summary['correct_braces'] += 1
        else:
            summary['incorrect_braces'] += 1
            summary['warnings'].append(f"Incorrect brace detected on {tooth_name}")
        
        if arch == 'Upper':
            summary['upper_teeth'] += 1
        else:
            summary['lower_teeth'] += 1
        
        # Create enhanced detection record
        enhanced_det = {
            'id': i + 1,
            'tooth_key': tooth_key,
            'tooth_name': tooth_name,
            'arch': arch,
            'is_correct': is_correct,
            'status': 'Correct' if is_correct else 'Incorrect',
            'confidence': det.get('confidence', 0.0),
            'confidence_percent': f"{det.get('confidence', 0.0) * 100:.1f}%",
            'bbox': bbox
        }
        
        summary['detections'].append(enhanced_det)
    
    return summary


def get_arch_from_tooth(tooth_key: str) -> str:
    """Get the dental arch (Upper/Lower) for a tooth."""
    return TOOTH_ARCH.get(tooth_key, 'Unknown')


def get_tooth_position_description(tooth_key: str) -> str:
    """Get a human-readable description of tooth position."""
    parts = tooth_key.split('_')
    if len(parts) != 2:
        return "Unknown position"
    
    arch = parts[0]
    tooth = parts[1]
    
    arch_name = "Upper" if arch == 'u' else "Lower"
    side = "Right" if arch[1] == 'r' else "Left"
    
    tooth_name = TOOTH_DISPLAY_NAMES.get(tooth_key, tooth.upper())
    
    return f"{arch_name} {side} {tooth_name}"


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def get_color_for_detection(is_correct: bool) -> Tuple[int, int, int]:
    """
    Get color for bounding box based on correctness.
    
    Args:
        is_correct: Whether the brace placement is correct
    
    Returns:
        Color tuple in BGR format
    """
    return TOOTH_COLORS['correct'] if is_correct else TOOTH_COLORS['incorrect']


def get_color_dark_for_detection(is_correct: bool) -> Tuple[int, int, int]:
    """
    Get darker color for bounding box glow effect.
    
    Args:
        is_correct: Whether the brace placement is correct
    
    Returns:
        Color tuple in BGR format
    """
    return TOOTH_COLORS['correct_dark'] if is_correct else TOOTH_COLORS['incorrect_dark']


# =============================================================================
# CLASS MAPPINGS
# =============================================================================

# Detection class names (for model with 2 classes)
CLASS_NAMES = {
    0: 'Correct Brace',
    1: 'Incorrect Brace',
}

# Combined tooth + status labels (for reference in future re-annotation)
COMBINED_LABELS = [
    'ci_correct',      # Central Incisor - Correct
    'ci_incorrect',    # Central Incisor - Incorrect
    'li_correct',      # Lateral Incisor - Correct
    'li_incorrect',    # Lateral Incisor - Incorrect
    'canine_correct',  # Canine - Correct
    'canine_incorrect',# Canine - Incorrect
    'p1_correct',      # First Premolar - Correct
    'p1_incorrect',    # First Premolar - Incorrect
    'p2_correct',      # Second Premolar - Correct
    'p2_incorrect',    # Second Premolar - Incorrect
]

# Human-readable mapping for combined labels
COMBINED_LABELS_READABLE = {
    'ci_correct': 'Central Incisor - Correct',
    'ci_incorrect': 'Central Incisor - Incorrect',
    'li_correct': 'Lateral Incisor - Correct',
    'li_incorrect': 'Lateral Incisor - Incorrect',
    'canine_correct': 'Canine - Correct',
    'canine_incorrect': 'Canine - Incorrect',
    'p1_correct': 'First Premolar - Correct',
    'p1_incorrect': 'First Premolar - Incorrect',
    'p2_correct': 'Second Premolar - Correct',
    'p2_incorrect': 'Second Premolar - Incorrect',
}

