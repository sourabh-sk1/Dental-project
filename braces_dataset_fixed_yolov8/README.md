# 🦷 Teeth Braces Placement Detection using YOLOv8

A deep learning computer vision project that analyzes dental images to detect orthodontic teeth braces brackets and determines whether they are correctly or incorrectly placed.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Image Detection](#image-detection)
  - [Webcam Detection](#webcam-detection)
  - [Web Application](#web-application)
- [Model Performance](#model-performance)
- [Classes](#classes)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## 📖 Overview

This project uses YOLOv8 (You Only Look Once, Version 8) for real-time object detection to identify and classify orthodontic braces brackets in dental images. The model can distinguish between:

- ✅ **Correct Brace** - Properly positioned brackets
- ❌ **Incorrect Brace** - Misaligned or improperly positioned brackets

## 📁 Project Structure

```
teeth-braces-ai/
│
├── dataset/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
│
├── data.yaml              # Dataset configuration
├── requirements.txt       # Python dependencies
├── convert_labels.py      # Label conversion script
├── train.py             # YOLOv8 training script
├── detect.py            # Image detection script
├── webcam_detect.py     # Real-time webcam detection
└── app.py              # Streamlit web application
```

## 📊 Dataset

The dataset is pre-labeled in YOLOv8 format with:
- **28 original classes** (detailed tooth and bracket types)
- **2 simplified classes** for main detection:
  - Class 0: `correct_brace` - Properly positioned brackets
  - Class 1: `incorrect_brace` - Misaligned brackets

### Dataset Location

The dataset should be placed in the project directory with the following structure:
```
dataset/
├── train/images/    # Training images
├── train/labels/   # Training labels (YOLO format)
├── valid/images/   # Validation images
├── valid/labels/  # Validation labels
└── test/images/   # Test images
```

## 💻 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Install Dependencies

```bash
# Clone or navigate to the project directory
cd teeth-braces-ai

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### Dependencies

The following packages will be installed:
- `ultralytics` - YOLOv8 library
- `opencv-python` - Image processing
- `numpy` - Numerical operations
- `streamlit` - Web interface
- `matplotlib` - Visualization
- `Pillow` - Image handling
- `PyYAML` - Configuration parsing
- `tqdm` - Progress bars

## 🚀 Usage

### Step 1: Convert Labels (If Needed)

If you're starting with the original 28-class dataset, convert to simplified 2-class format:

```bash
python convert_labels.py
```

### Step 2: Train the Model

Train YOLOv8 on your dataset:

```bash
python train.py
```

**Custom Training Options:**

```bash
# Train with custom parameters
python train.py --epochs 150 --batch 8 --imgsz 800

# Train with GPU
python train.py --device 0

# Train with different model size (s, m, l, x)
python train.py --model yolov8s.pt
```

**Training Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 100 | Number of training epochs |
| `--batch` | 16 | Batch size |
| `--imgsz` | 640 | Input image size |
| `--device` | cpu | Device to use (cpu or 0) |
| `--model` | yolov8n.pt | Model variant |

The trained model will be saved to:
- `runs/detect/train/weights/best.pt` (best model)
- `runs/detect/train/weights/last.pt` (last checkpoint)

### Step 3: Run Detection on Images

Detect braces in a single image:

```bash
python detect.py --image path/to/image.jpg
```

**Options:**

```bash
# Save the result
python detect.py --image path/to/image.jpg --save

# Display the result
python detect.py --image path/to/image.jpg --show

# Adjust confidence threshold
python detect.py --image path/to/image.jpg --conf 0.5

# Use custom model weights
python detect.py --image path/to/image.jpg --weights runs/detect/train/weights/best.pt
```

### Step 4: Real-time Webcam Detection

Run detection using your webcam:

```bash
python webcam_detect.py
```

**Controls:**
- `q` or `ESC` - Quit
- `s` - Take screenshot

**Options:**

```bash
# Use specific camera
python webcam_detect.py --device 1

# Adjust confidence
python webcam_detect.py --conf 0.5

# Hide FPS counter
python webcam_detect.py --no-fps
```

### Step 5: Launch Web Application

Start the Streamlit web app:

```bash
streamlit run app.py
```

The web app will open in your browser at `http://localhost:8501`.

**Web App Features:**
- Upload dental images
- Automatic braces detection
- Visual bounding boxes
- Confidence scores
- Results summary

## 📈 Model Performance

After training, metrics are saved in the `runs/detect/train/` directory:
- `results.png` - Training curves
- `confusion_matrix.png` - Confusion matrix
- `PR_curve.png` - Precision-Recall curve

### Key Metrics

| Metric | Description |
|--------|-------------|
| mAP50 | Mean Average Precision at IoU=0.50 |
| mAP50-95 | Mean Average Precision at IoU=0.50:0.95 |
| Precision | Ratio of correct detections |
| Recall | Ratio of detected objects |

## 🔍 Classes

### Simplified Classes (2-class model)

| Class ID | Name | Description |
|----------|------|-------------|
| 0 | correct_brace | Properly positioned brackets |
| 1 | incorrect_brace | Misaligned or improperly placed |

### Original Classes (28-class)

The original dataset includes detailed tooth type and bracket status:
- Central incisor, lateral incisor, canine, premolars, molars
- Each with correct/incorrect bracket variants

## ⚠️ Troubleshooting

### Common Issues

**1. Out of Memory (OOM) Errors**
```bash
# Reduce batch size
python train.py --batch 8

# Reduce image size
python train.py --imgsz 416
```

**2. No Detections**
- Lower the confidence threshold: `--conf 0.1`
- Check if the model is trained properly
- Verify image format is supported

**3. Webcam Not Working**
- Check camera index: Try `--device 1` or `--device 2`
- Ensure camera is not used by another application

**4. Module Not Found**
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

**5. Streamlit App Issues**
```bash
# Clear Streamlit cache
streamlit cache clear

# Reinstall Streamlit
pip install streamlit --upgrade
```

## 📝 Examples

### Example 1: Complete Training Workflow

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Convert labels (if needed)
python convert_labels.py

# 3. Train model
python train.py

# 4. Validate model
python -c "from train import validate_model; validate_model()"

# 5. Run detection
python detect.py --image test.jpg --save
```

### Example 2: Quick Inference

```bash
# Single image detection
python detect.py --image path/to/dental_image.jpg --show

# Batch detection
for img in *.jpg; do
    python detect.py --image "$img" --save
done
```

### Example 3: Web App Deployment

```bash
# Run locally
streamlit run app.py

# For external access
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- [Ultralytics](https://ultralytics.com/) - YOLOv8 implementation
- [OpenCV](https://opencv.org/) - Computer vision library
- [Streamlit](https://streamlit.io/) - Web framework

---

**Note:** This project is designed for demonstration and educational purposes. For clinical applications, please consult with dental professionals and ensure compliance with relevant regulations.

