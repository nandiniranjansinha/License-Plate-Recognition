# 🚗 License Plate Recognition System

Automated license plate detection and recognition using **YOLOv8** + **EasyOCR**. Detects plates in video, reads text with OCR, and outputs annotated results.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple.svg)

## ✨ Features

- Real-time license plate detection using custom YOLOv8 model
- OCR text recognition with intelligent character correction
- Text stabilization across frames using majority voting
- Visual overlay with zoomed plate and recognized text
- Supports video files and live webcam

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
EasyOCR → Character Correction → Validation → Stabilization → Output
```

**Key Components**:
- **YOLOv8**: Detects license plate bounding boxes
- **Preprocessing**: Grayscale + Otsu threshold + 2x resize
- **EasyOCR**: Reads text (A-Z, 0-9 only)
- **Correction**: Maps common OCR errors (`0→O`, `I→1`, etc.)
- **Validation**: Regex pattern `^[A-Z]{2}[0-9]{2}[A-Z]{3}$`
- **Stabilization**: Majority voting over last 10 detections per plate

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
├── easyocryoloLIVE.py          # Main script
├── weights/
│   └── best.pt                  # Trained model
├── vehicle_video.mp4            # Input video
├── output_with_licensev4.mp4   # Generated output
└── README.md
```

## 🐛 Common Issues

| Issue | Solution |
|-------|----------|
| Model not found | Ensure `weights/best.pt` exists |
| Video not found | Place `vehicle_video.mp4` in project root |
| Output is 0KB | Verify input video is valid, try different codec |
| "CUDA not available" warning | Normal - runs on CPU (slower but works) |
| No plates detected | Lower `CONF_THRESH` to 0.2 or improve video quality |
| Wrong OCR text | Adjust character mappings in `correct_plate_format()` |

## 📊 Performance

- **Speed**: 30+ FPS with GPU, 5-10 FPS on CPU
- **Accuracy**: Depends on training dataset and video quality
- **Format**: Currently optimized for XX00XXX format
