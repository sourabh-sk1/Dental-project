
Teeth Braces Placement Detection AI

A Streamlit web application for detecting orthodontic teeth braces and determining if they are correctly or incorrectly placed using YOLOv8.

 
Live Demo

Deploy this app to Streamlit Cloud for free free

 
Features

Single Image Detection - Upload a dental image and get instant results
Batch Processing - Process multiple images at once
Tooth Identification - Identifies tooth position (Central Incisor, Lateral Incisor, Canine, Premolars)
Color-Coded Results - Green = Correct Brace, Red = Incorrect Brace
Confidence Scores - View confidence levels for each detection
CSV Export - Download detection results as CSV
Model Performance Metrics - View mAP, Precision, Recall metrics

 
 Quick Deploy to Streamlit Cloud

Fork this repository to your GitHub account

Create a Streamlit Cloud account at https://streamlit.io/cloud

Connect your GitHub and select this repository

Deploy! - Streamlit will automatically detect streamlit_app.py and install dependencies

 
 Project Structure

teeth-braces-ai/
├── streamlit_app.py       # Main Streamlit web application
├── best.pt               # Trained YOLOv8 model weights (~87MB)
├── requirements.txt      # Python dependencies
├── data.yaml            # Dataset configuration
├── .streamlit/
│   └── config.toml      # Streamlit configuration
├── utils/
│   ├── __init__.py
│   └── tooth_mapping.py # Tooth position mapping utilities
└── README.md


 
Local Development

Prerequisites

Python 3.9+
pip
Installation

Clone the repository
git clone https://github.com/your-username/teeth-braces-ai.git
cd teeth-braces-ai
Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies
pip install -r requirements.txt
Run the app
streamlit run streamlit_app.py
The app will open at http://localhost:8501


 
Model Performance

Metric	Value
mAP@50	0.879
mAP@50-95	0.494
Precision	0.850
Recall	0.800

 Usage Tips

For best results: Use clear, well-lit dental photos
Confidence threshold: Adjust in sidebar (default: 40%)
Image size: Larger sizes = better accuracy but slower processing

Configuration

Confidence Threshold

Adjust the minimum confidence score in the sidebar. Higher values = fewer but more confident detections.

Image Size

Choose inference image size:

640: Fastest
960: Balanced (recommended)
1280: Most accurate

Detection Classes

Correct Brace (Green) - Properly positioned orthodontic brackets
Incorrect Brace (Red) - Misaligned or improperly placed brackets


 License

 MIT License


 Acknowledgments

 Ultralytics for YOLOv8
 Open source dental datasets contributors
