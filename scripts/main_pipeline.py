import cv2
import sys
import argparse
from pathlib import Path
from dataloader import KittiDataContext, GPS_DATA_DIR


# from yolo_loader import load_model

from geometry import GeometryTransformer
from maps_client import GoogleMapsClient
from visualizer import ARVisualizer

def main():
    parser = argparse.ArgumentParser(description="Main AR Navigation Pipeline")

    parser.add_argument(
        "drive_id",
        type=str,
        help="The ID of the drive to simulate (e.g., '2011_10_03_0042')"
    )

    parser.add_argument(
        "-n", "--num_frames",
        type=int,
        default=10,
        help="Number of frames to process (default: 10)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="output_video.mp4",
        help="Output video path"
    )

    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target GPS coordinates as 'lat,lon' (e.g., '48.123,8.456')"
    )

    args = parser.parse_args()
    DRIVE_TO_RUN = args.drive_id
    FRAMES_TO_PROCESS = args.num_frames
    
    # Parse target if provided
    target_gps = None
    if args.target:
        try:
            lat_str, lon_str = args.target.split(',')
            target_gps = (float(lat_str), float(lon_str))
            print(f"Target Mode Enabled: Destination set to {target_gps}")
        except ValueError:
            print("Error: Invalid target format. Use 'lat,lon'")
            sys.exit(1)

    print("--- Main Pipeline Start ---")

    try:
        loader = KittiDataContext(GPS_DATA_DIR)
    except FileNotFoundError as e:
        print(f"Failed to initialize loader: {e}")
        sys.exit(1)

    if not loader.load_drive_context(DRIVE_TO_RUN):
        print(f"Failed to load {DRIVE_TO_RUN}, exiting.")
        sys.exit(1)
        
    # Initialize Modules
    if not loader.calibration:
        print("Error: No calibration data found. Cannot proceed.")
        sys.exit(1)
        
    geometry = GeometryTransformer(loader.calibration)
    maps_client = GoogleMapsClient() # Will use env var or mock
    visualizer = ARVisualizer()
    
    # Video Writer Setup
    video_writer = None
    
    for frame_index in range(FRAMES_TO_PROCESS):

        print(f"\n[Main Pipeline] Processing Frame: {frame_index}")

        image_path, gps_data = loader.get_frame_data(frame_index)

        if not image_path or not gps_data:
            print(f"  -> Data not found for Frame {frame_index}, skipping.")
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  -> Failed to read image: {image_path.name}")
            continue
            
        # Initialize video writer once we know image size
        if video_writer is None:
            h, w = image.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(args.output, fourcc, 10.0, (w, h))

        # 1. Get Path from Maps API (or Mock)
        # Yaw in KITTI is 0=East, CCW.
        yaw = gps_data['yaw']
        
        # Lookahead 50m OR Explicit Target
        path_gps = maps_client.get_snapped_path(
            gps_data, 
            yaw, 
            lookahead_dist=50.0, 
            target_gps=target_gps
        )
        
        # 2. Transform to Image Coordinates
        projected_points = geometry.transform_path(path_gps, gps_data)
        
        # 3. Render
        result_image = visualizer.draw_ar_path(image, projected_points)
        result_image = visualizer.draw_info(result_image, f"Frame: {frame_index} | Yaw: {yaw:.2f}")
        
        # 4. Write to video
        video_writer.write(result_image)
        print(f"  -> Processed and saved frame {frame_index}")

    if video_writer:
        video_writer.release()
        print(f"\nSaved output to {args.output}")

    print("\n--- Main Pipeline Finished ---")


if __name__ == "__main__":
    main()