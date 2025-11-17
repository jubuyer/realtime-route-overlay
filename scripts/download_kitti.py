import os
import requests
from pathlib import Path
from tqdm import tqdm
import zipfile
import sys

"""
KITTI Dataset Downloader
Downloads and extracts KITTI Raw and Road datasets
"""

ROOT = Path(__file__).resolve().parent.parent
DATASETS_ROOT = ROOT / "datasets"
DOWNLOADS_DIR = DATASETS_ROOT / "downloads"
KITTI_RAW_DIR = DATASETS_ROOT / "kitti_raw"
KITTI_ROAD_DIR = DATASETS_ROOT / "kitti_road"

# Create directories
for dir_path in [DOWNLOADS_DIR, KITTI_RAW_DIR, KITTI_ROAD_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


class Downloader:
    """Downloader utility for KITTI datasets"""

    def __init__(self, urls, output_dir):
        """
        Args:
            urls: list of URLs to download
            output_dir: Path to save downloaded zips
        """
        self.urls = urls
        self.output_dir = Path(output_dir)

    def download_file(self, url):
        """
        Helper function to download files
        """
        local_filename = self.output_dir / url.split("/")[-1]
        if local_filename.exists():
            print(f"[SKIP] File already exists: {local_filename}")
            return local_filename

        print(f"\nDownloading: {url}")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            block_size = 1024
            with open(local_filename, 'wb') as f, tqdm(
                total=total_size, unit='B', unit_scale=True, desc=local_filename.name
            ) as t:
                for data in r.iter_content(block_size):
                    f.write(data)
                    t.update(len(data))
        return local_filename

    def extract_zip(self, zip_path, extract_to):
        """Extract zip file"""
        extract_to = Path(extract_to)
        print(f"\nExtracting {zip_path} -> {extract_to}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Extraction completed. Removing zip file.")
        zip_path.unlink()

    def run(self):
        """Download and extract all files"""
        for url in self.urls:
            zip_file = self.download_file(url)

            # Determine extraction directory
            if "raw" in url.lower():
                extract_to = KITTI_RAW_DIR
            elif "road" in url.lower():
                extract_to = KITTI_ROAD_DIR
            else:
                continue

            self.extract_zip(zip_file, extract_to)

        print("\nAll downloads and extractions completed.")

def main():
    """Main downloader pipeline"""

    kitti_urls = [
        "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0015/2011_09_26_drive_0015_sync.zip",
        "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0027/2011_09_26_drive_0027_sync.zip",
        "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0028/2011_09_26_drive_0028_sync.zip",
        "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0029/2011_09_26_drive_0029_sync.zip",
        "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0032/2011_09_26_drive_0032_sync.zip",
        "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0052/2011_09_26_drive_0052_sync.zip",
        "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0070/2011_09_26_drive_0070_sync.zip",
        "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0101/2011_09_26_drive_0101_sync.zip",
        "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_29_drive_0004/2011_09_29_drive_0004_sync.zip",
        "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_drive_0016/2011_09_30_drive_0016_sync.zip",
        "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_10_03_drive_0042/2011_10_03_drive_0042_sync.zip",
        "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_10_03_drive_0047/2011_10_03_drive_0047_sync.zip",
        "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_road.zip"
    ]

    downloader = Downloader(kitti_urls, DOWNLOADS_DIR)
    downloader.run()

    print(f"\nDownloaded datasets saved to: {DATASETS_ROOT}")

if __name__ == "__main__":
    main()
