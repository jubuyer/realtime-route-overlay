"""
Convert a folder of KITTI raw images (image_02) into a .mp4 video.
"""

import cv2
from pathlib import Path

def frames_to_video(frame_dir, output_path, fps=10):
    frame_dir = Path(frame_dir)
    images = sorted(frame_dir.glob("*.png"))  # KITTI frames are PNGs

    if not images:
        raise ValueError(f"No frames found in {frame_dir}")

    # Read first frame to get size
    first_frame = cv2.imread(str(images[0]))
    height, width, _ = first_frame.shape

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    print(f"Writing video {output_path} from {len(images)} frames...")
    for img_path in images:
        frame = cv2.imread(str(img_path))
        out.write(frame)

    out.release()
    print("Done!")

if __name__ == "__main__":
    # Example usage:
    # frames_to_video("2011_09_26_drive_0052_sync/image_02", "drive_0052.mp4", fps=10)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Path to KITTI image folder')
    parser.add_argument('--output', type=str, required=True, help='Output .mp4 path')
    parser.add_argument('--fps', type=int, default=10, help='FPS of output video')
    args = parser.parse_args()

    frames_to_video(args.input_dir, args.output, args.fps)
