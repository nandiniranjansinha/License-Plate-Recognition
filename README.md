# 🚗 License Plate Recognition System

Automated license plate detection and recognition using **YOLOv8** + **EasyOCR**. Detects plates in video, reads text with OCR, and outputs annotated results.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple.svg)
![EasyOCR](https://img.shields.io/badge/OCR-EasyOCR-green.svg)

## ✨ Features

- 🚘 **Real-time license plate detection** using a fine-tuned **YOLOv8** model  
- 🔤 **Text recognition** via **EasyOCR** with intelligent character correction  
- ⚙️ **Text stabilization** using temporal majority voting (smooth, consistent results)  
- 🖼️ **Visual overlays** with zoomed license plate previews and recognized text  
- 🎥 Supports **4K video input** and **live webcam feed**

**Plate Format**: `XX00XXX` (2 letters + 2 numbers + 3 letters, e.g., `AB12CDE`)

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/nandiniranjansinha/License-Plate-Recognition.git
cd License-Plate-Recognition

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install ultralytics easyocr numpy
```

### 2. Get Input Video
Download sample: [Pexels Traffic Video](https://www.pexels.com/video/traffic-flow-in-the-highway-2103099/)  
Save as `vehicle_video.mp4` in project root

### 3. Run
```bash
python easyocryoloLIVE.py
```

Output saved as `output_with_licensev4.mp4`. Press `q` to stop early.

### Use Webcam Instead
```python
# In easyocryoloLIVE.py, change:
input_video = 0  # Instead of 'vehicle_video.mp4'
```

## 🧠 How It Works

```
Video → YOLOv8 Detection → Crop Plate → Preprocess → 
EasyOCR → Character Correction → Regex Validation → Stabilization → Output Video
```

| Module                     | Purpose                                                    |
| -------------------------- | ---------------------------------------------------------- |
| **YOLOv8**                 | Detects number plate bounding boxes                        |
| **Preprocessing**          | Grayscale + Otsu Threshold + 2× Resize for OCR             |
| **EasyOCR**                | Extracts alphanumeric text                                 |
| **Correction Logic**       | Maps OCR misreads (`0→O`, `I→1`, `8→B`, etc.)              |
| **Regex Validation**       | Ensures correct plate pattern `^[A-Z]{2}[0-9]{2}[A-Z]{3}$` |
| **Temporal Stabilization** | Majority voting over 10 frames per plate                   |


## 🏋️ Model Training

**Training Notebook**: [Google Colab](https://colab.research.google.com/drive/17yDy7LW9lDdRGcO6geMqmVLejS4Kp7hb?usp=sharing)

### Quick Training Guide
```python
# In Google Colab
!pip install ultralytics

from ultralytics import YOLO
model = YOLO('yolov8n.pt')

# Train (adjust data.yaml path)
model.train(data='path/to/data.yaml', epochs=100, imgsz=640, batch=16)

# Download weights
from google.colab import files
files.download('runs/detect/train/weights/best.pt')
```

Place `best.pt` in `weights/` folder.

## ⚙️ Configuration

### Adjust Detection Threshold
```python
CONF_THRESH = 0.3  # Lower = more detections, Higher = fewer false positives
```

### Change Overlay Size
```python
overlay_h, overlay_w = 100, 300  # Height, Width in pixels
```

### Different Plate Format
```python
# For ABC-1234 format:
plate_pattern = re.compile(r"^[A-Z]{3}-[0-9]{4}$")
```

### Output as AVI
```python
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = 'output.avi'
```

## 📁 Project Structure

```
License-Plate-Recognition/
├── easyocryoloLIVE.py         # Main detection script
├── calculate_metrics.py       # FPS & performance evaluation
├── weights/
│   └── best.pt                # Trained YOLOv8 model
├── vehicle_video.mp4          # Input video
├── output_with_licensev4.mp4  # Annotated output
└── README.md
```
## 📊 Performance Metrics

| Metric                    | Value        | Notes                                  |
| ------------------------- | ------------ | -------------------------------------- |
| **Speed (4K)**            | ~43 FPS      | Measured on GPU; faster than real-time |
| **Detection FPS (1080p)** | ~60 FPS      | On GPU; ~10 FPS on CPU                 |
| **OCR Correction**        | ↓ 25% errors | Using custom mapping logic             |
| **Stabilization Buffer**  | 10 frames    | Majority voting for consistent text    |
| **Format Supported**      | XX00XXX      | Easily customizable                    |


## 🐛 Common Issues

| Issue                | Solution                                   |
| -------------------- | ------------------------------------------ |
| `Model not found`    | Ensure `weights/best.pt` exists            |
| `Video not found`    | Place `vehicle_video.mp4` in root          |
| `Output file = 0KB`  | Try `mp4v` or `XVID` codec                 |
| `CUDA not available` | Runs on CPU (slower)                       |
| `No plates detected` | Lower `CONF_THRESH`                        |
| `Wrong OCR output`   | Adjust mapping in `correct_plate_format()` |


## 🏁 Results

Processed 4K traffic video at ~43 FPS, achieving real-time inference speed.

Reduced OCR misreads by 25% through character correction and validation.

Delivered smooth detection and stable text overlay using 10-frame majority voting.
