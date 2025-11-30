import json
from pathlib import Path
import sys

try:
    ROOT = Path(__file__).resolve().parent.parent
except NameError:
    ROOT = Path.cwd()

PROCESSED_DIR = ROOT / "datasets" / "processed" / "kitti"
GPS_DATA_DIR = PROCESSED_DIR / "gps_data"


class KittiDataContext:

    def __init__(self, processed_gps_dir=GPS_DATA_DIR):
        self.base_dir = Path(processed_gps_dir)
        self.current_drive_id = None
        self.current_image_dir = None
        self.gps_data_map = {}

        if not self.base_dir.exists():
            print(f"[DataLoader Error] Base directory not found: {self.base_dir}")
            raise FileNotFoundError(f"Base data directory not found: {self.base_dir}")

    def load_drive_context(self, drive_id: str):
        if drive_id == self.current_drive_id:
            return True

        print(f"\n[DataLoader Context] Loading Drive: {drive_id}...")
        self.current_drive_id = drive_id

        drive_data_path = self.base_dir / self.current_drive_id
        if not drive_data_path.exists():
            print(f"[DataLoader Error] Drive folder not found: {drive_data_path}")
            self.current_drive_id = None
            return False

        self.current_image_dir = drive_data_path / "images"
        if not self.current_image_dir.exists():
            print(f"[DataLoader Error] Image folder not found: {self.current_image_dir}")
            self.current_drive_id = None
            return False

        gps_file_path = drive_data_path / "gps_data.json"
        try:
            with open(gps_file_path, 'r') as f:
                gps_list = json.load(f)
        except FileNotFoundError:
            print(f"[DataLoader Error] GPS JSON file not found: {gps_file_path}")
            self.current_drive_id = None
            return False

        self.gps_data_map = {}
        for frame_data in gps_list:
            try:
                self.gps_data_map[frame_data['frame']] = frame_data
            except KeyError:
                print(f"[DataLoader Warning] Found an item without a 'frame' key in {gps_file_path}.")

        print(f"[DataLoader OK] Successfully loaded {drive_id} ({len(self.gps_data_map)} frames)")
        
        # Load Calibration
        calib_path = drive_data_path / "calibration.json"
        try:
            with open(calib_path, 'r') as f:
                self.calibration = json.load(f)
            print(f"[DataLoader OK] Loaded calibration data.")
        except FileNotFoundError:
            print(f"[DataLoader Warning] Calibration file not found: {calib_path}")
            self.calibration = {}
            
        return True

    def get_frame_data(self, frame_index: int):
        if not self.current_drive_id:
            print("[DataLoader Error] You must call load_drive_context() first.")
            return None, None

        gps_info = self.gps_data_map.get(frame_index)

        if not gps_info:
            return None, None

        image_name = f"{frame_index:06d}.png"
        image_path = self.current_image_dir / image_name

        if not image_path.exists():
            return None, gps_info

        return image_path, gps_info