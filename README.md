# Face Recognition & Blink Detection System

## Overview
This project implements a real-time Face Recognition and Blink Detection (Liveness Detection) system using Python.
The system supports:
- Face comparison using images
- Real-time face recognition using webcam
- Blink detection using Eye Aspect Ratio (EAR)
- Multiple face detection and tracking
- Automatic camera index detection
---
# Technologies Used
- Python
- OpenCV
- face_recognition (dlib-based)
- NumPy
---
# Concepts Used
- 128-D Face Encoding
- Euclidean Distance
- Eye Aspect Ratio (EAR)
- Facial Landmark Detection
- Real-Time Face Tracking
---
# Features
## 1. Face Comparison
- Accepts two image paths
- Detects faces from both images
- Generates 128-D facial encodings
- Compares faces using Euclidean Distance
- Displays:
  - Same Person
  - Different Person
---
## 2. Real-Time Face Recognition + Blink Detection
### Features Included
- Real-time webcam detection
- Multiple face detection
- Individual face tracking with IDs
- Blink detection using EAR
- Person count display
- Individual blink count display
- Automatic camera detection
- Real-time window display
---
# Blink Detection Logic
Blink detection is implemented using:
- Facial landmarks
- Left eye and right eye detection
- Eye Aspect Ratio (EAR)
- State transition:
  - OPEN → CLOSED → OPEN
A blink is counted only when valid eye closure is detected.
---
# Folder Structure

```bash
face-recognition-blink-detection/
│
├── images/
│   ├── billgates1.jpg
│   ├── stevejobs1.jpg
│   └── ...
│
├── main.py
├── requirements.txt
└── README.md
```
---
# Installation
## Install Dependencies
```bash
pip install -r requirements.txt
```
---
# How to Run
```bash
python main.py
```
---
# Menu Options
## Option 1
Compare two images
## Option 2
Real-time face recognition and blink detection
---
# Output
The system displays:
- Face bounding boxes
- Person IDs
- Person count
- Blink status
- Blink count per person
Terminal output also displays:
- Detected person count
- Blink events
---
# Future Improvements
- Advanced face tracking
- Anti-spoofing detection
- Adaptive EAR threshold
- MediaPipe optimization
- Deep learning-based liveness detection
---
# Author
Vaishnavi Garlapati