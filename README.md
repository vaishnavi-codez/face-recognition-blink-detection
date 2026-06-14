# Face Recognition & Blink Detection System

## Overview
This project implements a real-time Face Recognition and Blink Detection (Liveness Detection) system using Python.
The system supports:
- Face comparison using images
- Real-time face recognition using webcam
- Blink detection using Eye Aspect Ratio (EAR)
- Passive anti-spoofing checks for texture, reflection, landmark depth, background motion, and screen artifacts
- Active liveness challenges such as turn left, turn right, smile, or nod
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
- Texture and specular reflection analysis
- Optical flow background-motion analysis
- Frequency-domain screen artifact detection
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
- Rejects flat printed photos and screen replays using anti-spoofing checks
- Requires a random active challenge before accepting a face as live
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
# Anti-Spoofing Logic
The real-time system now combines blink detection with stronger liveness checks:

- Texture analysis rejects overly smooth or uniform face crops.
- Reflection analysis checks for uneven natural face highlights instead of flat glare.
- Landmark depth cues reject faces that look too flat.
- Background motion compares face movement with the surrounding scene.
- Screen artifact detection looks for grid-like display patterns.
- Active challenge detection asks the user to turn, smile, or nod.

The webcam overlay shows `Spoof: ...` when passive checks fail, or `Do: ...` when the user still needs to complete the active challenge.
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
├── anti_spoofing.py
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
- Liveness rejection reason or active challenge instruction
Terminal output also displays:
- Detected person count
- Blink events
---
# Future Improvements
- Advanced face tracking
- Dedicated deep learning anti-spoofing model integration
- Adaptive EAR threshold
- MediaPipe optimization
- Deep learning-based liveness detection
---
# Author
Vaishnavi Garlapati
