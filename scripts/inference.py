import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# set root to repo of lane detection
PROJECT_ROOT = Path(__file__).resolve().parent.parent
UFLD_PATH = PROJECT_ROOT / "models" / "Ultra-Fast-Lane-Detection-v2"

# Make repo importable
sys.path.insert(0, str(UFLD_PATH))

# importing model and config files over 
from model.model_culane import parsingNet
import configs.tusimple_res18 as cfg

# Anchors (same as in repo docs/issues)
# Normalized row/col anchors (0–1) like official pred2coords
ROW_ANCHOR = np.linspace(160, 710, cfg.num_row) / 720.0
COL_ANCHOR = np.linspace(0.0, 1.0, cfg.num_col)


# Build model (same args as model_tusimple.get_model)
def build_model(device: torch.device):
    model = parsingNet(
        pretrained=True,
        backbone=cfg.backbone,           # '18'
        num_grid_row=cfg.num_cell_row,   # 100
        num_cls_row=cfg.num_row,         # 56
        num_grid_col=cfg.num_cell_col,   # 100
        num_cls_col=cfg.num_col,         # 41
        num_lane_on_row=cfg.num_lanes,   # 4
        num_lane_on_col=cfg.num_lanes,   # 4
        use_aux=cfg.use_aux,             # False
        input_height=cfg.train_height,   # 320
        input_width=cfg.train_width,     # 800
        fc_norm=cfg.fc_norm,             # False
    ).to(device)

    return model


# Load checkpoint
def load_model(ckpt_path: Path, device: torch.device):
    print(f"✓ Loading model from: {ckpt_path}")
    model = build_model(device)

    state = torch.load(str(ckpt_path), map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    new_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            new_state[k[7:]] = v  # strip "module."
        else:
            new_state[k] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print("! Missing keys:", missing)
    if unexpected:
        print("! Unexpected keys:", unexpected)

    model.eval()
    print("✓ Model loaded & set to eval()")
    return model

# Post-processing: pred → lane coordinates
def pred2coords(pred, img_w, img_h, local_width=1):
    """
    pred: dict with keys 'loc_row', 'loc_col', 'exist_row', 'exist_col'
    returns: list[list[(x, y)]], one list per lane
    """
    batch_size, num_grid_row, num_cls_row, num_lane_row = pred["loc_row"].shape
    batch_size, num_grid_col, num_cls_col, num_lane_col = pred["loc_col"].shape

    max_indices_row = pred["loc_row"].argmax(1).cpu()  # [B, num_cls_row, num_lane_row]
    valid_row = pred["exist_row"].argmax(1).cpu()      # [B, num_cls_row, num_lane_row]

    max_indices_col = pred["loc_col"].argmax(1).cpu()
    valid_col = pred["exist_col"].argmax(1).cpu()

    pred["loc_row"] = pred["loc_row"].cpu()
    pred["loc_col"] = pred["loc_col"].cpu()

    coords = []

    # In the official code, middle two lanes are row-based, outer two are col-based
    row_lane_idx = [1, 2]
    col_lane_idx = [0, 3]

    # ----- row-based lanes -----
    for i in row_lane_idx:
        tmp = []
        if valid_row[0, :, i].sum() > num_cls_row / 2:
            for k in range(valid_row.shape[1]):  # over cls_row (row anchors)
                if valid_row[0, k, i]:
                    all_ind = torch.tensor(
                        list(
                            range(
                                max(0, max_indices_row[0, k, i] - local_width),
                                min(
                                    num_grid_row - 1,
                                    max_indices_row[0, k, i] + local_width,
                                )
                                + 1,
                            )
                        )
                    )

                    out_tmp = (
                        pred["loc_row"][0, all_ind, k, i].softmax(0)
                        * all_ind.float()
                    ).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_row - 1) * img_w
                    x = int(out_tmp)
                    y = int(ROW_ANCHOR[k] * img_h)
                    tmp.append((x, y))
            if tmp:
                coords.append(tmp)

    # ----- col-based lanes -----
    for i in col_lane_idx:
        tmp = []
        if valid_col[0, :, i].sum() > num_cls_col / 4:
            for k in range(valid_col.shape[1]):  # over cls_col (col anchors)
                if valid_col[0, k, i]:
                    all_ind = torch.tensor(
                        list(
                            range(
                                max(0, max_indices_col[0, k, i] - local_width),
                                min(
                                    num_grid_col - 1,
                                    max_indices_col[0, k, i] + local_width,
                                )
                                + 1,
                            )
                        )
                    )

                    out_tmp = (
                        pred["loc_col"][0, all_ind, k, i].softmax(0)
                        * all_ind.float()
                    ).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_col - 1) * img_h
                    x = int(COL_ANCHOR[k] * img_w)
                    y = int(out_tmp)
                    tmp.append((x, y))
            if tmp:
                coords.append(tmp)

    return coords


# Preprocess & forward
def run_on_image(model, img_path: Path, save_path: Path, device: torch.device):
    print(f"\n=== Inference on {img_path} ===")
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(img_path)

    ori_h, ori_w = img.shape[:2]

    # Same crop strategy as training/onnx examples
    cut_height = int(cfg.train_height * (1 - cfg.crop_ratio))  # 320 * 0.2 = 64
    img_crop = img[cut_height:, :, :]  # crop off top band
    img_resized = cv2.resize(
        img_crop, (cfg.train_width, cfg.train_height), interpolation=cv2.INTER_CUBIC
    )

    # BGR → RGB, normalize
    img_norm = img_resized[:, :, ::-1].astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_norm - mean) / std

    # HWC → CHW
    img_chw = np.transpose(img_norm, (2, 0, 1))
    inp = torch.from_numpy(img_chw).unsqueeze(0).to(device)  # [1,3,H,W]

    with torch.no_grad():
        pred = model(inp)
        # pred is a dict with loc_row/loc_col/exist_row/exist_col
        if not isinstance(pred, dict):
            raise TypeError(f"Expected dict output, got {type(pred)}")

    lanes = pred2coords(pred, ori_w, ori_h)

    # Draw lanes on original image
    vis = img.copy()
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)]

    for li, lane in enumerate(lanes):
        color = colors[li % len(colors)]
        for x, y in lane:
            cv2.circle(vis, (x, y), 3, color, -1)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), vis)
    print(f"✓ Saved visualization to {save_path}")
    print(f"✓ Detected {len(lanes)} lane polylines")
    return vis, lanes


# Main
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt_path = UFLD_PATH / "weights" / "tusimple_res18.pth"
    img_path = (
        PROJECT_ROOT
        / "datasets"
        / "TUSimple"
        / "test_set"
        / "clips"
        / "0530"
        / "1492626047222176976_0"
        / "1.jpg"
    )
    out_path = PROJECT_ROOT / "results" / "tusimple_test_output.jpg"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    model = load_model(ckpt_path, device)
    run_on_image(model, img_path, out_path, device)
