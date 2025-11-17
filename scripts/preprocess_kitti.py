import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import pykitti

"""
KITTI Dataset Parser for AR Navigation
Processes KITTI Raw and Road datasets for road segmentation and GPS alignment
"""

ROOT = Path(__file__).resolve().parent.parent
DATASETS_ROOT = ROOT / "datasets"
KITTI_RAW_DIR = DATASETS_ROOT / "kitti_raw"
KITTI_ROAD_DIR = DATASETS_ROOT / "kitti_road" / "data_road"
PROCESSED_DIR = DATASETS_ROOT / "processed" / "kitti"

# Output directories
OUT_TRAIN_IMG = PROCESSED_DIR / "images" / "train"
OUT_VAL_IMG = PROCESSED_DIR / "images" / "val"
OUT_TRAIN_LABEL = PROCESSED_DIR / "labels" / "train"
OUT_VAL_LABEL = PROCESSED_DIR / "labels" / "val"
GPS_DATA_DIR = PROCESSED_DIR / "gps_data"

# Create directories
for dir_path in [OUT_TRAIN_IMG, OUT_VAL_IMG, OUT_TRAIN_LABEL, OUT_VAL_LABEL, GPS_DATA_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


class KITTIRoadParser:
    """
    Parser for KITTI Road dataset
    Converts road segmentation masks to YOLO polygon format
    """
    
    def __init__(self, kitti_road_dir):
        self.kitti_road_dir = Path(kitti_road_dir)
        self.train_dir = self.kitti_road_dir / "training"
        self.test_dir = self.kitti_road_dir / "testing"
        
        # Road categories in KITTI
        self.road_colors = {
            'road': (255, 0, 255),      # Magenta - road
            'lane': (255, 0, 0),        # Red - lane markings
        }
        
    def parse_road_mask(self, mask_path):
        """
        Parse KITTI road segmentation mask
        Returns binary mask for road area
        """
        mask = cv2.imread(str(mask_path))
        if mask is None:
            print(f"Warning: Unable to read mask {mask_path}")
            return None
        
        # Convert to binary road mask
        # KITTI uses magenta (255, 0, 255) for road
        road_mask = np.all(mask == [255, 0, 255], axis=2).astype(np.uint8) * 255
        
        return road_mask
    
    def mask_to_yolo_polygons(self, mask, img_width, img_height):
        """
        Convert binary mask to YOLO polygon format
        Returns list of normalized polygon coordinates
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        polygons = []
        for cnt in contours:
            # Simplify contour
            epsilon = 0.002 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            if len(approx) < 3:
                continue
            
            # Normalize coordinates
            poly = []
            for point in approx.reshape(-1, 2):
                x_norm = point[0] / img_width
                y_norm = point[1] / img_height
                poly.extend([x_norm, y_norm])
            
            polygons.append(poly)
        
        return polygons
    
    def process_dataset(self, split='train', val_split=0.15):
        """
        Process KITTI Road dataset and convert to YOLO format
        
        Args:
            split: 'train' or 'test'
            val_split: fraction of training data to use for validation
        """
        if split == 'train':
            img_dir = self.train_dir / "image_2"
            mask_dir = self.train_dir / "gt_image_2"
        else:
            img_dir = self.test_dir / "image_2"
            mask_dir = None
        
        if not img_dir.exists():
            print(f"Warning: {img_dir} does not exist")
            return
        
        if split == 'train' and (mask_dir is None or not mask_dir.exists()):
            print(f"Warning: {mask_dir} does not exist")
            return
        
        # Get all images
        img_files = sorted(list(img_dir.glob("*.png")))
        
        if split == 'train' and mask_dir:
            # Split into train/val
            np.random.seed(42)
            indices = np.random.permutation(len(img_files))
            val_size = int(len(img_files) * val_split)
            val_indices = set(indices[:val_size])
            
            print(f"\nProcessing KITTI Road dataset:")
            print(f"  Total images: {len(img_files)}")
            print(f"  Train: {len(img_files) - val_size}")
            print(f"  Val: {val_size}")
        
        stats = {'train': 0, 'val': 0, 'no_mask': 0}
        
        for idx, img_path in enumerate(tqdm(img_files, desc=f"Processing {split}")):
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            img_height, img_width = img.shape[:2]
            
            # Determine if train or val
            is_val = (split == 'train') and (idx in val_indices)
            out_img_dir = OUT_VAL_IMG if is_val else OUT_TRAIN_IMG
            out_label_dir = OUT_VAL_LABEL if is_val else OUT_TRAIN_LABEL
            
            # Save image
            img_name = img_path.name
            out_img_path = out_img_dir / img_name
            cv2.imwrite(str(out_img_path), img)
                        
            # Process mask if available
            if mask_dir:
                parts = img_path.stem.split("_")

                if len(parts) == 2:
                    category, frame = parts
                    road_mask_name = f"{category}_road_{frame}.png"
                    mask_path = mask_dir / road_mask_name
                else:
                    mask_path = None

                if mask_path and mask_path.exists():
                    road_mask = self.parse_road_mask(mask_path)
                    
                    if road_mask is not None:
                        # Convert to YOLO polygons
                        polygons = self.mask_to_yolo_polygons(road_mask, img_width, img_height)
                        
                        # Save label file
                        label_path = out_label_dir / f"{img_path.stem}.txt"
                        if polygons:
                            with open(label_path, 'w') as f:
                                for poly in polygons:
                                    # Compute bounding box from polygon
                                    xs = poly[::2]
                                    ys = poly[1::2]
                                    x_min, x_max = min(xs), max(xs)
                                    y_min, y_max = min(ys), max(ys)
                                    x_center = (x_min + x_max) / 2
                                    y_center = (y_min + y_max) / 2
                                    w = x_max - x_min
                                    h = y_max - y_min

                                    # Class 0 = road
                                    line = f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f} " + " ".join([f"{v:.6f}" for v in poly])
                                    f.write(line + "\n")
                        else:
                            open(label_path, 'w').close()  # Empty label file
                        
                        if is_val:
                            stats['val'] += 1
                        else:
                            stats['train'] += 1
                    else:
                        stats['no_mask'] += 1
                else:
                    stats['no_mask'] += 1
        
        print(f"\nProcessing complete:")
        print(f"  Train samples: {stats['train']}")
        print(f"  Val samples: {stats['val']}")
        print(f"  Skipped (no mask): {stats['no_mask']}")
        
        return stats


class KITTIRawParser:
    """
    Parser for KITTI Raw dataset
    Extracts GPS data and camera calibration for map overlay
    """
    
    def __init__(self, kitti_raw_dir):
        self.kitti_raw_dir = Path(kitti_raw_dir)
        
    def get_available_drives(self):
        """Get list of available drive sequences"""
        drives = []
        if not self.kitti_raw_dir.exists():
            print(f"Warning: {self.kitti_raw_dir} does not exist")
            return drives
        
        for date_dir in self.kitti_raw_dir.iterdir():
            if date_dir.is_dir() and date_dir.name.startswith('2011'):
                for drive_dir in date_dir.iterdir():
                    if drive_dir.is_dir() and 'sync' in drive_dir.name:
                        drive_number = drive_dir.name.split('_')[4]
                        drives.append((date_dir.name, drive_number))
        
        return drives
    
    def load_drive_data(self, date, drive):
        """
        Load KITTI Raw drive data using pykitti
        
        Args:
            date: date string (e.g., '2011_09_26')
            drive: drive number (e.g., '0001')
        
        Returns:
            data: pykitti dataset object with GPS, images, calibration
        """
        try:
            data = pykitti.raw(str(self.kitti_raw_dir), date, drive)
            return data
        except Exception as e:
            print(f"Error loading drive {date}/{drive}: {e}")
            return None
    
    def extract_gps_data(self, data, output_path):
        """
        Extract GPS coordinates and timestamps from KITTI Raw drive
        
        Args:
            data: pykitti dataset object
            output_path: path to save GPS data JSON
        """
        gps_data = []
        
        # Get oxts data (GPS/IMU)
        for i, oxts in enumerate(data.oxts):
            gps_entry = {
                'frame': i,
                'timestamp': data.timestamps[i].strftime('%Y-%m-%d %H:%M:%S.%f'),
                'lat': oxts.packet.lat,
                'lon': oxts.packet.lon,
                'alt': oxts.packet.alt,
                'roll': oxts.packet.roll,
                'pitch': oxts.packet.pitch,
                'yaw': oxts.packet.yaw,
                'vf': oxts.packet.vf,  # forward velocity
                'vl': oxts.packet.vl,  # lateral velocity
                'vu': oxts.packet.vu,  # upward velocity
            }
            gps_data.append(gps_entry)
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(gps_data, f, indent=2)
        
        print(f"Saved GPS data: {len(gps_data)} frames -> {output_path}")
        return gps_data
    
    def extract_calibration(self, data, output_path):
        """
        Extract camera calibration matrices
        Essential for projecting GPS coordinates to image coordinates
        """
        calib = {
            'P2': data.calib.P_rect_20.tolist(),  # Camera 2 projection matrix
            'R_rect': data.calib.R_rect_00.tolist(),  # Rectification matrix
            'Tr_velo_to_cam': data.calib.T_cam0_velo.tolist(),  # Velodyne to camera
        }
        
        with open(output_path, 'w') as f:
            json.dump(calib, f, indent=2)
        
        print(f"Saved calibration data -> {output_path}")
        return calib
    
    def process_drive(self, date, drive, save_images=True):
        """
        Process a complete KITTI Raw drive sequence
        Extracts images, GPS data, and calibration
        """
        print(f"\nProcessing drive: {date}/{drive}")
        
        # Load data
        data = self.load_drive_data(date, drive)
        if data is None:
            return None
        
        # Create output directory
        drive_output = GPS_DATA_DIR / f"{date}_{drive}"
        drive_output.mkdir(parents=True, exist_ok=True)
        
        # Extract GPS data
        gps_path = drive_output / "gps_data.json"
        gps_data = self.extract_gps_data(data, gps_path)
        
        # Extract calibration
        calib_path = drive_output / "calibration.json"
        calib = self.extract_calibration(data, calib_path)
        
        # Optionally save images
        if save_images:
            img_output = drive_output / "images"
            img_output.mkdir(exist_ok=True)
            
            for i, img in enumerate(tqdm(data.cam2, desc="Saving images")):
                img_path = img_output / f"{i:06d}.png"
                cv2.imwrite(str(img_path), cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
        
        print(f"Drive processing complete: {len(gps_data)} frames")
        
        return {
            'gps_data': gps_data,
            'calibration': calib,
            'output_dir': drive_output
        }


def create_yolo_config():
    """Create YAML config file for YOLOv8 training on KITTI"""
    config = {
        'path': str(PROCESSED_DIR),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,
        'names': {0: 'road'}
    }
    
    config_path = PROCESSED_DIR / "kitti_road.yaml"
    with open(config_path, 'w') as f:
        import yaml
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\nCreated YOLO config: {config_path}")
    return config_path


def main():
    """Main processing pipeline"""
    
    print("="*60)
    print("KITTI Dataset Processing Pipeline")
    print("="*60)
    
    # Step 1: Process KITTI Road dataset for segmentation training
    print("\n[1/2] Processing KITTI Road Dataset...")
    if KITTI_ROAD_DIR.exists():
        road_parser = KITTIRoadParser(KITTI_ROAD_DIR)
        road_parser.process_dataset(split='train', val_split=0.15)
        
        # Create YOLO config
        config_path = create_yolo_config()
    else:
        print(f"KITTI Road dataset not found at {KITTI_ROAD_DIR}")
        print("Download from: http://www.cvlibs.net/datasets/kitti/eval_road.php")
    
    # Step 2: Process KITTI Raw dataset for GPS data
    print("\n[2/2] Processing KITTI Raw Dataset (GPS + Calibration)...")
    if KITTI_RAW_DIR.exists():
        raw_parser = KITTIRawParser(KITTI_RAW_DIR)
        
        # Get available drives
        drives = raw_parser.get_available_drives()
        print(f"Found {len(drives)} drive sequences")
        
        # Process first few drives as examples
        for i, (date, drive) in enumerate(drives[:3]):  # Process first 3 drives
            result = raw_parser.process_drive(date, drive, save_images=True)
    else:
        print(f"KITTI Raw dataset not found at {KITTI_RAW_DIR}")
        print("Download from: http://www.cvlibs.net/datasets/kitti/raw_data.php")
    
    print("\n" + "="*60)
    print("Processing Complete!")
    print("="*60)
    print(f"\nProcessed data location: {PROCESSED_DIR}")

if __name__ == "__main__":
    main()