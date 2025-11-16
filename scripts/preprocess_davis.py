import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

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


def load_split(file_path):
    with open(file_path, "r") as f:
        return [line.strip() for line in f]

# Resizes DAVIS images to 640x640 with letterboxing for YOLOv8 compatibility
def resize_letterbox(img, new_size=640):
    h, w = img.shape[:2]
    scale = new_size / max(h, w)
    resized = cv2.resize(img, (int(w * scale), int(h * scale)))

    pad_h = new_size - resized.shape[0]
    pad_w = new_size - resized.shape[1]

    padded = cv2.copyMakeBorder(
        resized,
        0, pad_h,
        0, pad_w,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )
    return padded, scale, pad_w, pad_h

# Converts binary mask to YOLO polygon segmentation format
# Requires polygons to be in the format: [x1, y1, x2, y2, ..., xn, yn]
def mask_to_yolo_polygons(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []

    for cnt in contours:
        if len(cnt) < 3:
            continue
        cnt = cnt.squeeze(1)
        polys.append(cnt)

    return polys


def process_sequence(seq_list, img_out_dir, label_out_dir):
    for seq in tqdm(seq_list, desc=f"Processing {img_out_dir}"):

        img_seq_path = f"{IMG_INPUT}/{seq}"
        mask_seq_path = f"{MASK_INPUT}/{seq}"

        frame_ids = sorted(os.listdir(img_seq_path))

        for frame in frame_ids:
            img_path = f"{img_seq_path}/{frame}"
            mask_path = f"{mask_seq_path}/{frame.replace('.jpg', '.png')}"

            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, 0)

            # Resize
            img_resized, scale, pad_w, pad_h = resize_letterbox(img)
            mask_resized, _, _, _ = resize_letterbox(mask)

            # Convert mask to polygons
            polys = mask_to_yolo_polygons((mask_resized > 128).astype(np.uint8))

            # Write image
            out_img_path = f"{img_out_dir}/{seq}_{frame}"
            cv2.imwrite(out_img_path, img_resized)

            # Write label
            label_path = f"{label_out_dir}/{seq}_{frame.replace('.jpg', '.txt')}"
            h, w = img_resized.shape[:2]

            with open(label_path, "w") as f:
                for poly in polys:
                    # Normalize polygon coordinates
                    poly_norm = []
                    for x, y in poly:
                        poly_norm.append(x / w)
                        poly_norm.append(y / h)

                    # Write class ID 0
                    poly_str = " ".join([str(p) for p in poly_norm])
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
            "names:\n"
            "  0: foreground\n"
        )

    print("DAVIS preprocessing complete.")
