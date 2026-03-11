"""
Streamlit Web App for Teeth Braces Detection
==============================================
This web application provides a user-friendly interface for detecting
orthodontic teeth braces brackets and determining if they are correctly
or incorrectly placed.

Features:
- Upload dental images
- Automatic braces detection
- Visual bounding boxes with labels
- Confidence scores display
- Results summary

Usage:
    streamlit run app.py
"""

import os
# Disable Streamlit telemetry/onboarding prompt for automated runs so the app
# doesn't block waiting for interactive input. This must be set before
# importing `streamlit`.
os.environ.setdefault("STREAMLIT_DISABLE_TELEMETRY", "1")
import sys
from pathlib import Path
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile

# Page configuration
st.set_page_config(
    page_title="Teeth Braces Detection AI",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Subtitle styling */
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Success box styling */
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #D4EDDA;
        border: 1px solid #C3E6CB;
        color: #155724;
        margin: 0.5rem 0;
    }
    
    /* Warning box styling */
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #F8D7DA;
        border: 1px solid #F5C6CB;
        color: #721C24;
        margin: 0.5rem 0;
    }
    
    /* Info box styling */
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #D1ECF1;
        border: 1px solid #BEE5EB;
        color: #0C5460;
        margin: 0.5rem 0;
    }
    
    /* Custom button styling */
    .stButton > button {
        background-color: #2E86AB;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
    }
    
    /* Metric card styling */
    .metric-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #F8F9FA;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# Class names and colors
CLASS_NAMES = {
    0: 'Correct Brace',
    1: 'Incorrect Brace',
}

COLORS = {
    0: (0, 255, 0),     # Green for correct
    1: (0, 0, 255),     # Red for incorrect
}


@st.cache_resource
def load_model(weights_path: str = 'runs/detect/train/weights/best.pt'):
    """
    Load the trained YOLO model (cached for performance).
    
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
        else:
            st.warning(f"Model weights not found at {weights_path}")
            st.info("Using pretrained YOLOv8 model for demonstration...")
            weights_path = 'yolov8n.pt'
    
    try:
        model = YOLO(weights_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def draw_detections(image: np.ndarray, results) -> tuple:
    """
    Draw bounding boxes and labels on the image.
    
    Args:
        image: Input image as numpy array
        results: YOLO detection results
        
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
            
            # Draw rectangle
            cv2.rectangle(
                annotated_image,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                3  # Thicker line for visibility
            )
            
            # Prepare label text
            label = f"{CLASS_NAMES[class_id]}: {conf:.1%}"
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,  # Font scale
                2     # Thickness
            )
            
            # Draw filled rectangle behind text
            cv2.rectangle(
                annotated_image,
                (int(x1), int(y1) - text_height - 15),
                (int(x1) + text_width + 10, int(y1)),
                color,
                -1  # Filled rectangle
            )
            
            # Draw text
            cv2.putText(
                annotated_image,
                label,
                (int(x1) + 5, int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,  # Font scale
                (255, 255, 255),  # Text color (white)
                2     # Thickness
            )
            
            # Update counters
            if class_id == 0:
                correct_count += 1
            elif class_id == 1:
                incorrect_count += 1
            
            # Store detection info
            detections.append({
                'class': CLASS_NAMES[class_id],
                'class_id': class_id,
                'confidence': conf,
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


def process_image(image: Image.Image, model, conf: float = 0.25) -> tuple:
    """
    Process a PIL image with the detection model.
    
    Args:
        image: PIL Image object
        model: Loaded YOLO model
        conf: Confidence threshold
        
    Returns:
        Tuple of (annotated image, detection summary)
    """
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Convert RGB to BGR for OpenCV
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_array
    
    # Run detection
    results = model(img_bgr, conf=conf, verbose=False)
    
    # Draw detections
    annotated_image, summary = draw_detections(img_bgr, results)
    
    # Convert back to RGB for display
    annotated_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    return annotated_rgb, summary


def main():
    """
    Main function for the Streamlit web app.
    """
    # App header
    st.markdown('<p class="main-title">🦷 Teeth Braces Placement Detection AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Detect and classify orthodontic braces bracket placement</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # Model settings
    st.sidebar.header("Model Configuration")
    
    weights_path = st.sidebar.text_input(
        "Model Weights Path",
        value="runs/detect/train/weights/best.pt",
        help="Path to the trained YOLO model weights"
    )
    
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Minimum confidence score for detections"
    )
    
    # About section
    st.sidebar.markdown("---")
    st.sidebar.header("ℹ️ About")
    st.sidebar.markdown("""
    **Teeth Braces Detection AI**
    
    This application uses YOLOv8 to detect orthodontic 
    braces brackets and classify them as correctly 
    or incorrectly placed.
    
    **Classes:**
    - 🟢 Correct Brace - Properly positioned
    - 🔴 Incorrect Brace - Misaligned
    """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    # File uploader
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a dental image...",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a dental image for braces detection"
        )
        
        # Sample images info
        st.info("💡 Tip: For best results, use clear, well-lit dental photos.")
    
    # Process uploaded image
    if uploaded_file is not None:
        # Load and display original image
        image = Image.open(uploaded_file).convert('RGB')
        
        with col1:
            st.subheader("📷 Original Image")
            st.image(image, use_container_width=True)
        
        # Load model
        model = load_model(weights_path)
        
        if model is not None:
            # Process image
            with st.spinner('🔍 Analyzing image...'):
                annotated_image, summary = process_image(image, model, conf_threshold)
            
            # Display results
            with col2:
                st.subheader("🔎 Detection Results")
                
                # Display annotated image
                st.image(annotated_image, use_container_width=True)
                
                # Results summary
                st.markdown("### 📊 Results Summary")
                
                # Metrics
                m1, m2, m3 = st.columns(3)
                
                with m1:
                    st.metric(
                        label="Total Detections",
                        value=summary['total_detections']
                    )
                
                with m2:
                    st.metric(
                        label="✅ Correct",
                        value=summary['correct_braces'],
                        delta_color="normal"
                    )
                
                with m3:
                    st.metric(
                        label="❌ Incorrect",
                        value=summary['incorrect_braces'],
                        delta_color="inverse"
                    )
                
                # Detailed detections
                if summary['detections']:
                    st.markdown("### 📋 Detailed Detections")
                    
                    for i, det in enumerate(summary['detections'], 1):
                        conf_percent = f"{det['confidence'] * 100:.1f}%"
                        
                        if det['class_id'] == 0:
                            st.markdown(
                                f'<div class="success-box">'
                                f'**{i}. {det["class"]}** - Confidence: {conf_percent}'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f'<div class="warning-box">'
                                f'**{i}. {det["class"]}** - Confidence: {conf_percent}'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                else:
                    st.warning("No braces detected in the image.")
                
                # Overall status
                st.markdown("---")
                if summary['incorrect_braces'] > 0:
                    st.markdown(
                        '<div class="warning-box">'
                        '⚠️ **Attention Required!** Some braces appear to be incorrectly placed.'
                        '</div>',
                        unsafe_allow_html=True
                    )
                elif summary['correct_braces'] > 0:
                    st.markdown(
                        '<div class="success-box">'
                        '✓ **All braces appear to be correctly placed!**'
                        '</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.info("No braces detected in the uploaded image.")
        
        else:
            st.error("Failed to load the detection model. Please check the model weights path.")
    
    else:
        # Show placeholder when no image is uploaded
        with col2:
            st.subheader("🔎 Results")
            st.info("👈 Upload an image to start detection")
            
            # Demo image option
            st.markdown("---")
            st.markdown("### Don't have an image?")
            st.markdown("You can test the app with any dental image.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666;'>"
        "🦷 Teeth Braces Detection AI | Powered by YOLOv8"
        "</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

