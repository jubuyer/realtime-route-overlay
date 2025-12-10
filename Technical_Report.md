# Technical Report: AR Path Projection System

## 0. Acknowledgement
I am not a Computer Vision expert, but I am a pretty good Prompt Engineer.

## 1. Introduction
This document details the implementation of the Augmented Reality (AR) Path Projection system for the KITTI autonomous driving dataset. The system projects a navigation path, derived from Google Maps, onto the vehicle's front-facing camera feed.

## 2. New Modules & Implementation Details

We have added three core Python modules to the `scripts/` directory to handle the specific requirements of coordinate transformation, API integration, and visualization.

### 2.1 `scripts/geometry.py` (Coordinate Transformation Engine)
**Purpose**: Handles the complex mathematical chain required to transform a GPS point (Latitude/Longitude) into a 2D pixel coordinate on the image.

**Mathematical Logic**:
The transformation pipeline follows a strict chain of rigid body transformations:

1.  **WGS84 to Local Tangent Plane (LTP)**:
    *   Converts Geodetic coordinates $(\phi, \lambda, h)$ to a local Cartesian system $(x, y, z)$ in meters.
    *   **Method**: Flat Earth Approximation (valid for short distances < 10km).
    *   $x = \Delta \lambda \cdot R_e \cdot \cos(\phi_{ref})$
    *   $y = \Delta \phi \cdot R_e$

2.  **LTP to Vehicle Body Frame**:
    *   Transforms points from the "East-North-Up" frame to the vehicle's "Forward-Left-Up" frame.
    *   **Critical Detail**: Handles KITTI's specific IMU definition where Yaw ($\psi$) is $0$ at East and increases counter-clockwise.
    *   Equation: $P_{body} = (R_z(\psi) R_y(\theta) R_x(\phi))^T \cdot P_{LTP}$

3.  **Body to Camera Frame**:
    *   **Extrinsics**: Applies the rigid transformation from the IMU to the Velodyne LiDAR ($T_{velo}^{imu}$), and then from Velodyne to the Camera ($T_{cam}^{velo}$).
    *   $P_{cam\_unrect} = T_{cam}^{velo} \cdot T_{velo}^{imu} \cdot P_{body}$

4.  **Rectification & Projection**:
    *   **Rectification**: Applies the rotation $R_{rect}^{(0)}$ to align the camera with the stereo rectification plane.
    *   **Intrinsics**: Projects the 3D point onto the 2D image plane using the projection matrix $P_2$.
    *   $\tilde{p} = P_2 \cdot R_{rect}^{(0)} \cdot P_{cam\_unrect}$
    *   Final pixel coordinates: $u = \tilde{p}_x / \tilde{p}_z, v = \tilde{p}_y / \tilde{p}_z$

### 2.2 `scripts/maps_client.py` (Google Maps Integration)
**Purpose**: Fetches the accurate road geometry.

**Method**:
*   Uses the **Google Maps Roads API** (`snapToRoads`).
*   **Logic**: Instead of drawing a straight line to a destination, the system calculates a "virtual target" 50 meters ahead of the vehicle based on its current heading.
*   It sends the segment `[Current Position, Virtual Target]` to the API with `interpolate=true`.
*   The API returns a set of points that perfectly follow the curvature of the actual road, correcting for GPS drift and geometric simplifications.

### 2.3 `scripts/visualizer.py` (AR Rendering)
**Purpose**: Renders the projected points onto the video frame.

**Method**:
*   Receives a list of $(u, v)$ pixel coordinates.
*   Uses OpenCV (`cv2.polylines`) to draw a smooth, anti-aliased curve connecting these points.
*   Draws a directional arrow at the end of the path to indicate heading.
*   Applies a semi-transparent overlay to ensure the AR graphics blend naturally with the scene.

## 3. Current System Pipeline

The `main_pipeline.py` script orchestrates the entire process in a frame-by-frame loop:

1.  **Load Data**: `KittiDataContext` loads the image and raw GPS/IMU data for the current frame $t$.
2.  **Fetch Path**: `GoogleMapsClient` takes the GPS position and Yaw, queries the API (or uses a mock fallback), and retrieves the road geometry.
3.  **Transform**: `GeometryTransformer` processes the road points through the 4-stage coordinate system chain described above, outputting a list of 2D pixel coordinates.
4.  **Render**: `ARVisualizer` draws the path overlay onto the original image.
5.  **Output**: The processed frame is written to `output_video.mp4`.

---
## This process explains how the system combines local vehicle data with cloud-based map data to generate the AR path.

1. **Input (Vehicle State)**

Source: gps_data.json (from the KITTI dataset).

Action: The system reads the vehicle's current Latitude, Longitude, and Heading (Yaw) for the specific video frame being processed.
Example: "Vehicle is at [48.001, 8.001] facing North-East."

2. **Target Estimation**

Source: Internal Calculation (maps_client.py).

Action: Based on the current position and heading, the system mathematically projects a "Virtual Target Point" 50 meters directly ahead of the vehicle.
Example: "Target point is at [48.002, 8.002]."

3. **API Request (Query)**

Action: The system sends a request to the Google Maps Roads API (snapToRoads).

Payload: It sends a path segment consisting of [Current Position, Virtual Target] and requests interpolation (interpolate=true).

Intent: "I am at point A and moving towards point B. Please give me the exact geometry of the road connecting these points."

4. **API Response (Road Geometry)**

Source: Google Maps Platform.

Action: Google returns a list of high-resolution coordinates [P1, P2, P3, ...] that represent the actual center line of the road.

Benefit: These points follow the road's curvature (e.g., turns, winding paths) rather than a straight line, and are corrected for GPS drift.

5. **Projection & Rendering**

Action: The system takes this list of points [P1, P2, P3, ...] returned by Google, transforms them from GPS coordinates into 2D image pixel coordinates using the GeometryTransformer, and draws the yellow AR line on the video frame.



## Problem: 

1. The pipeline that I have designed is not machine learning based. LOL

2. The pipeline is not real-time. LOL

3. The information is sufficient for a "Navigation Guidance" system. We don't really need to use a model to get the road mask in the image. LOL

From a "Navigation Guidance" perspective, the information is sufficient. If we assume high-precision GPS (centimeter-level) and perfect Google Maps data, we indeed do not need to detect the road in the image. The mathematical projection tells us exactly where the road should be based on coordinates. We can simply draw it, and it would align perfectly. This is similar to a racing video game where the map and vehicle position are known ground truths.

However, from an "AR Realism" and "Robustness" perspective, the information is NOT sufficient. In the real world, GPS has drift (meters of error), and map data can be outdated. Without detecting the road in the image (Visual Grounding), we face several issues:

Drift/Misalignment: The AR arrow might be drawn on the sidewalk or grass instead of the road.

Occlusion Errors: If there is a truck in front, the AR arrow might be drawn over the truck instead of being occluded by it.

Elevation Errors: On a steep hill, pure GPS calculation might cause the arrow to appear "floating" in the air.


## Next Steps

1. Apply a Direct Target instead of a Virtual Target.
2. Integration with Road Detection Model (yes or no?)
3. Start writing the latex report.