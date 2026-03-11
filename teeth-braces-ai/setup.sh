#!/bin/bash
# Setup script for Teeth Braces Detection AI
# This script installs dependencies and prepares the app for deployment

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Downloading YOLOv8 model..."
python -c "from ultralytics import YOLO; model = YOLO('yolov8l.pt')"

echo "Setup complete!"

