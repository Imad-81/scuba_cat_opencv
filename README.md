# Scuba Cat

A real-time hand wave detection application using computer vision that triggers a meme video when wave gestures are detected.

## Overview

This project uses MediaPipe for hand and face detection, combined with OpenCV for video capture and processing. When a wave gesture is detected in front of the camera, it automatically plays a meme video file.

## Features

- Real-time hand tracking and gesture detection
- Face mesh detection for reference points
- Wave gesture recognition with configurable sensitivity
- Cooldown mechanism to prevent repeated triggers
- Customizable threshold values for fine-tuning detection

## Requirements

The following libraries are required to run this project:

- **opencv-python** - Computer vision library for video capture and processing
- **mediapipe** - Hand and face detection framework
- **numpy** - Numerical computing library

## Installation

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the required libraries:
```bash
pip install opencv-python mediapipe numpy
```

## Usage

Place your meme video file (`scuba_cat.MP4`) in the project directory, then run:

```bash
python3 main.py
```

## Configuration

You can adjust the following parameters in `main.py` to fine-tune the detection:

- `NOSE_DISTANCE_THRESHOLD` - Distance threshold for face detection (pixels)
- `WAVE_HISTORY` - Number of frames to track for wave detection
- `WAVE_THRESHOLD` - Minimum movement in pixels to register a wave
- `COOLDOWN` - Seconds between triggers

## License

MIT
