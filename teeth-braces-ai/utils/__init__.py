"""
Teeth Braces Detection AI - Utils Package
=========================================
This package contains utility functions for tooth mapping and visualization.
"""

from .tooth_mapping import (
    TOOTH_NAMES,
    TOOTH_SHORT_CODES,
    TOOTH_DISPLAY_NAMES,
    TOOTH_ARCH,
    TOOTH_COLORS,
    TOOTH_TYPE_CODES,
    TOOTH_POSITION_MAP,
    get_tooth_from_position,
    get_tooth_from_yolo_box,
    format_detection_label,
    format_tooth_name,
    create_detection_summary,
    get_arch_from_tooth,
    get_tooth_position_description,
    get_color_for_detection,
    get_color_dark_for_detection,
    CLASS_NAMES,
    COMBINED_LABELS,
    COMBINED_LABELS_READABLE,
)

__all__ = [
    'TOOTH_NAMES',
    'TOOTH_SHORT_CODES',
    'TOOTH_DISPLAY_NAMES',
    'TOOTH_ARCH',
    'TOOTH_COLORS',
    'TOOTH_TYPE_CODES',
    'TOOTH_POSITION_MAP',
    'get_tooth_from_position',
    'get_tooth_from_yolo_box',
    'format_detection_label',
    'format_tooth_name',
    'create_detection_summary',
    'get_arch_from_tooth',
    'get_tooth_position_description',
    'get_color_for_detection',
    'get_color_dark_for_detection',
    'CLASS_NAMES',
    'COMBINED_LABELS',
    'COMBINED_LABELS_READABLE',
]

