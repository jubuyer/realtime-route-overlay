import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
import math

DAVIS_ROOT = "datasets/davis/DAVIS"
OUTPUT_ROOT = "datasets/processed/davis"

IMG_INPUT = f"{DAVIS_ROOT}/JPEGImages/480p"
MASK_INPUT = f"{DAVIS_ROOT}/Annotations/480p"
TRAIN_SPLIT = f"{DAVIS_ROOT}/ImageSets/2017/train.txt"
VAL_SPLIT = f"{DAVIS_ROOT}/ImageSets/2017/val.txt"

# Output directories
OUT_TRAIN_IMG = f"{OUTPUT_ROOT}/images/train"
OUT_VAL_IMG = f"{OUTPUT_ROOT}/images/val"
OUT_TRAIN_LABEL = f"{OUTPUT_ROOT}/labels/train"
OUT_VAL_LABEL = f"{OUTPUT_ROOT}/labels/val"

os.makedirs(OUT_TRAIN_IMG, exist_ok=True)
os.makedirs(OUT_VAL_IMG, exist_ok=True)
os.makedirs(OUT_TRAIN_LABEL, exist_ok=True)
os.makedirs(OUT_VAL_LABEL, exist_ok=True)

# Load train/val splits
def load_split(file_path):
    with open(file_path, "r") as f:
        return [line.strip() for line in f]

# Letterbox function to resize and pad images
def letterbox(img, new_size=640):
    h, w = img.shape[:2]
    scale = new_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    img_resized = cv2.resize(img, (new_w, new_h))

    # Padding
    dw = (new_size - new_w) / 2
    dh = (new_size - new_h) / 2

    top = int(math.floor(dh))
    bottom = int(math.ceil(dh))
    left = int(math.floor(dw))
    right = int(math.ceil(dw))

    img_padded = cv2.copyMakeBorder(
        img_resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    return img_padded, scale, left, top, new_w, new_h

def mask_letterbox(mask, new_w, new_h, left, top, new_size=640):
    mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    mask_padded = cv2.copyMakeBorder(
        mask_resized,
        top,
        new_size - new_h - top,
        left,
        new_size - new_w - left,
        cv2.BORDER_CONSTANT,
        value=0
    )
    return mask_padded

def mask_to_yolo_polygons(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for cnt in contours:
        try:
            cnt = cnt.reshape(-1, 2)
        except ValueError:
            continue
        if cnt.shape[0] < 3:
            continue
        polys.append(cnt)
    return polys

def process_sequence(seq_list, img_out_dir, label_out_dir):
    for seq in tqdm(seq_list, desc=f"Processing {img_out_dir}"):
        img_seq_path = f"{IMG_INPUT}/{seq}"
        mask_seq_path = f"{MASK_INPUT}/{seq}"

        if not os.path.isdir(img_seq_path):
            print(f"Warning: missing image sequence {img_seq_path}, skipping")
            continue
        if not os.path.isdir(mask_seq_path):
            print(f"Warning: missing mask sequence {mask_seq_path}, skipping")
            continue

        frame_ids = sorted(os.listdir(img_seq_path))

        for frame in frame_ids:
            img_path = f"{img_seq_path}/{frame}"
            mask_path = f"{mask_seq_path}/{frame.replace('.jpg', '.png')}"

            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: could not read image {img_path}, skipping")
                continue

            mask = cv2.imread(mask_path, 0)
            if mask is None:
                mask = np.zeros(img.shape[:2], dtype=np.uint8)

            # Image letterbox
            img_resized, scale, left, top, new_w, new_h = letterbox(img)

            # Mask transform
            mask_resized = mask_letterbox(mask, new_w, new_h, left, top)

            # Polygons
            polys = mask_to_yolo_polygons((mask_resized > 128).astype(np.uint8))

            # Save image
            out_img_name = f"{seq}_{frame}"
            out_img_path = f"{img_out_dir}/{out_img_name}"
            cv2.imwrite(out_img_path, img_resized)

            # Save label
            out_label = f"{label_out_dir}/{seq}_{frame.replace('.jpg', '.txt')}"
            h, w = img_resized.shape[:2]

            with open(out_label, "w") as f:
                for poly in polys:
                    poly_norm = []
                    for x, y in poly:
                        poly_norm.append(x / w)
                        poly_norm.append(y / h)

                    poly_str = " ".join([f"{p:.6f}" for p in poly_norm])
                    f.write(f"0 {poly_str}\n")

if __name__ == "__main__":
    train_seqs = load_split(TRAIN_SPLIT)
    val_seqs = load_split(VAL_SPLIT)

    process_sequence(train_seqs, OUT_TRAIN_IMG, OUT_TRAIN_LABEL)
    process_sequence(val_seqs, OUT_VAL_IMG, OUT_VAL_LABEL)

    # Write davis.yaml for YOLOv8 training
    with open(f"{OUTPUT_ROOT}/davis.yaml", "w") as f:
        f.write(
            "path: datasets/processed/davis\n"
            "train: images/train\n"
            "val: images/val\n"
            "nc: 1\n"
            "names:\n"
            "  0: foreground\n"
        )

    print("DAVIS preprocessing complete.")
