
# Dental Braces Detection AI

A professional YOLOv8-based computer vision solution for detecting orthodontic braces in dental images and assessing placement accuracy.

This repository includes a Streamlit demo, dataset utilities, and training scripts for orthodontic bracket detection.

## Features

- Single image inference with instant visual feedback
- Batch processing for multiple dental images
- Tooth position mapping for central incisors, lateral incisors, canines, and premolars
- Color-coded output for correct vs incorrect brace placement
- Confidence scores for each detection
- CSV export for result review and reporting
- Built-in model performance metrics

## Deployment

To deploy this application on Streamlit Cloud:

1. Fork the repository to your GitHub account.
2. Create or sign in to a Streamlit Cloud account.
3. Connect your GitHub repository.
4. Deploy — Streamlit will detect `streamlit_app.py` and install dependencies automatically.

## Repository Structure

- `teeth-braces-ai/`
  - `streamlit_app.py` — Streamlit web application entry point
  - `requirements.txt` — Python dependencies
  - `data.yaml` — YOLOv8 dataset configuration
  - `best.pt` — Trained YOLOv8 model weights
  - `.streamlit/` — Streamlit configuration files
  - `utils/` — Helper modules for tooth mapping and inference
- `braces_dataset_fixed_yolov8/`
  - `train.py` — Training pipeline for YOLOv8
  - `detect.py` — Inference script
  - `webcam_detect.py` — Webcam-based detection example
  - `convert_labels.py` — Dataset label conversion utility
  - `data.yaml` — Dataset configuration
  - `runs/` — Generated training and detection outputs

## Local Development

### Prerequisites

- Python 3.9 or newer
- `pip`
- Optional: virtual environment

### Installation

```bash
git clone https://github.com/sourabh-sk1/Dental-project.git
cd Dental-project/teeth-braces-ai
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your browser.



## Model Performance

| Metric      | Value |
|------------|:-----:|
| mAP@50     | 0.879 |
| mAP@50-95  | 0.494 |
| Precision  | 0.850 |
| Recall     | 0.800 |

## Usage Recommendations

- Use clear, well-lit dental images for best detection results.
- Increase the confidence threshold to reduce false positives.
- Use larger inference image sizes for improved accuracy, with slower processing.

## Configuration Notes

- **Confidence threshold**: Higher values yield more precise results at the cost of recall.
- **Image size**: Lower settings are faster, higher settings improve localization accuracy.

## License

This project is licensed under the MIT License.

## Acknowledgments

- Ultralytics for YOLOv8
- Open-source dental imaging contributors

