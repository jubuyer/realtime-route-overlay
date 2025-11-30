# AR Pipeline Instructions

This document provides a comprehensive guide to the Augmented Reality (AR) Path Projection Pipeline. It details the internal processing steps and provides instructions on how to execute the system.

## 1. Pipeline Overview

The system processes video frames from the KITTI dataset to overlay a navigation path that aligns with the real-world road geometry. The pipeline consists of five sequential stages:

### Step 1: Data Loading
*   **Input**: Raw KITTI dataset files (Images + GPS/IMU data).
*   **Action**: The system loads the image for the current frame and its corresponding sensor data (Latitude, Longitude, Altitude, Yaw, Pitch, Roll).
*   **Context**: Calibration matrices (Camera Intrinsics, Extrinsics) are also loaded at initialization.

### Step 2: Target Determination
The system determines the destination point based on the selected mode:
*   **Default Mode**: Calculates a "Virtual Target" 50 meters directly ahead of the vehicle based on its current heading (Yaw).
*   **Target Mode**: Uses a user-specified coordinate (Latitude, Longitude) as the explicit destination.

### Step 3: Road Geometry Retrieval
*   **Action**: The system sends the `[Current Position, Target Position]` pair to the **Google Maps Roads API**.
*   **Output**: The API returns a list of "snapped" coordinates that follow the actual curvature of the road, correcting for GPS drift and providing a smooth path.

### Step 4: Coordinate Transformation
*   **Action**: The retrieved GPS points are transformed from the Geodetic system to the 2D Image Pixel system.
*   **Chain**: `WGS84 (GPS)` $\to$ `Local Tangent Plane (Meters)` $\to$ `Vehicle Body Frame` $\to$ `Camera Frame` $\to$ `Image Plane (Pixels)`.

### Step 5: AR Rendering
*   **Action**: The transformed pixel coordinates are drawn onto the original video frame.
*   **Visuals**: A smooth yellow path is rendered with a directional arrow at the end, using a semi-transparent overlay for a "Heads-Up Display" (HUD) effect.

---

## 2. Usage Guide

### Prerequisites
Before running the pipeline, ensure your Google Maps API Key is set in the `.env` file:
```bash
GOOGLE_MAPS_API_KEY=your_api_key_here
```

### Basic Execution
Run the pipeline in default mode (projects 50m ahead).
```bash
# Syntax: python scripts/main_pipeline.py [DRIVE_ID]
python scripts/main_pipeline.py 2011_10_03_0042
```

### Target Mode (Navigation)
Run the pipeline with a specific destination coordinate.
```bash
# Syntax: python scripts/main_pipeline.py [DRIVE_ID] --target "LAT,LON"
python scripts/main_pipeline.py 2011_09_30_0016 --target "49.0340,8.3950"
```

### Common Options
*   `-n [NUMBER]`: Number of frames to process (Default: 10).
*   `--output [FILENAME]`: Name of the output video file (Default: `output_video.mp4`).

**Example: Process 200 frames and save to `demo.mp4`**
```bash
python scripts/main_pipeline.py 2011_10_03_0042 -n 200 --output demo.mp4
```
