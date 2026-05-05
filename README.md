# Face Recognition & Blink Detection System
##  Overview
This project implements a Face Recognition system along with Blink Detection (liveness check) using Python.
It has two main features:
1. Face Comparison using image inputs
2. Real-time Blink Detection using webcam
---
## Technologies Used
- Python
- face_recognition (built on dlib)
- OpenCV
- NumPy
- SciPy
---
## Concepts Used
- 128-Dimensional Face Encoding
- Euclidean Distance
- Cosine Similarity
- Eye Aspect Ratio (EAR)
- Facial Landmarks Detection
---
## Features
### 1. Face Comparison
- Takes two image paths as input
- Detects faces
- Generates face encodings
- Compares using:
  - Euclidean Distance
  - Cosine Similarity
- Outputs:
  - Same Person
  - Maybe Same
  - Different Person
---
### 2. Blink Detection (Liveness Detection)
- Uses webcam
- Detects facial landmarks
- Calculates Eye Aspect Ratio (EAR)
- Displays:
  - "Blink" when eyes are closed
  - "Eyes Open" otherwise
---
##  How to Run
### Option 1 (If libraries already installed)
```bash
python main.py