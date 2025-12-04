"""
Baseline Video Pipeline for UFLDv2 Lane Detection

CPU-only sequential video processing pipeline that loads frames one by one,
converts to PyTorch tensors, and feeds them to UFLDv2 for inference.
Includes comprehensive benchmarking with FPS, latency, and p95 metrics.
"""
import json
import os
import sys
import time
import argparse
import numpy as np
import psutil
import cv2
import torch
from collections import deque
from pathlib import Path
import socket
import platform

# set root to repo of lane detection
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
UFLD_PATH = PROJECT_ROOT / "models" / "Ultra-Fast-Lane-Detection-v2"

# Make repo importable
sys.path.insert(0, str(UFLD_PATH))

# importing model and config files over 
from model.model_culane import parsingNet
import configs.tusimple_res18 as cfg

# pred2coords static variables
ROW_ANCHOR = np.linspace(160, 710, cfg.num_row) / 720.0
COL_ANCHOR = np.linspace(0.0, 1.0, cfg.num_col)

class VideoFrameLoader:
    def __init__(self, video_path, target_height=None, target_width=None):
        if target_height is None:
            target_height = cfg.train_height
        if target_width is None:
            target_width = cfg.train_width
        self.video_path = video_path
        self.target_height = int(target_height)
        self.target_width = int(target_width)
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.total_frames = total if total > 0 else None
        fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.fps = fps if fps > 0.0 else 0.0
        self.current_frame = 0
        
        self.metrics = {
            'load_times': [],
            'preprocess_times': [],
            'total_times': []
        }

    def _process_frame(self, frame, t_start):
        t_load = time.perf_counter()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cut_height = int(cfg.train_height * (1 - cfg.crop_ratio))
        frame_crop = frame_rgb[cut_height:, :, :]
        frame_resized = cv2.resize(frame_crop, (self.target_width, self.target_height))
        frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=frame_tensor.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=frame_tensor.dtype).view(3, 1, 1)
        frame_tensor = (frame_tensor - mean) / std
        frame_tensor = frame_tensor.unsqueeze(0)
        t_preprocess = time.perf_counter()
        self.metrics['load_times'].append(t_load - t_start)
        self.metrics['preprocess_times'].append(t_preprocess - t_load)
        self.metrics['total_times'].append(t_preprocess - t_start)
        return frame_tensor, frame_rgb
    
    def __iter__(self):
        self.current_frame = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return self
    
    def __next__(self):
        t_start = time.perf_counter()
        ret, frame = self.cap.read()
        if not ret:
            raise StopIteration
        frame_tensor, frame_rgb = self._process_frame(frame, t_start)
        self.current_frame += 1
        return frame_tensor, frame_rgb
    
    def __getitem__(self, idx):
        if self.total_frames is not None and (idx < 0 or idx >= self.total_frames):
            raise IndexError("Frame index out of range")
        t_start = time.perf_counter()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = self.cap.read()
        if not ret:
            raise IndexError(f"Frame {idx} could not be read")
        frame_tensor, frame_rgb = self._process_frame(frame, t_start)
        self.current_frame = int(idx) + 1
        return frame_tensor, frame_rgb
    
    def __len__(self):
        return self.total_frames if self.total_frames is not None else 0
    
    def get_metrics(self):
        return self.metrics
    
    def release(self):
        self.cap.release()

    def get_summary(self):
        summary = {}
        def stats(arr):
            if not arr:
                return 0.0, 0.0, 0.0, 0.0
            a = np.array(arr)
            return float(a.mean()), float(a.std()), float(np.percentile(a, 95)), float(np.percentile(a, 99))
        l_mean, l_std, l_p95, l_p99 = stats(self.metrics['load_times'])
        p_mean, p_std, p_p95, p_p99 = stats(self.metrics['preprocess_times'])
        t_mean, t_std, t_p95, t_p99 = stats(self.metrics['total_times'])
        summary['load'] = {'mean': l_mean, 'std': l_std, 'p95': l_p95, 'p99': l_p99}
        summary['preprocess'] = {'mean': p_mean, 'std': p_std, 'p95': p_p95, 'p99': p_p99}
        summary['total'] = {'mean': t_mean, 'std': t_std, 'p95': t_p95, 'p99': t_p99}
        return summary 

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
def load_ufldv2_model(ckpt_path: Path, device: torch.device):
    print(f"Loading model from: {ckpt_path}")
    print(f"Using device: {device}")
    model = build_model(device)

    state = torch.load(str(ckpt_path), map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    new_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            new_state[k[7:]] = v
        else:
            new_state[k] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)

    model.eval()
    print("Model loaded & set to eval()")
    return model

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

def overlay_lanes_on_frame(frame, output, img_height=cfg.train_height, img_width=cfg.train_width):
    frame_overlay = frame.copy()
    try:
        lanes = pred2coords(output, img_width, img_height)
        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)]
        for li, lane in enumerate(lanes):
            color = colors[li % len(colors)]
            for x, y in lane:
                cv2.circle(frame_overlay, (x, y), 3, color, -1)
    except Exception as e:
        pass
    return frame_overlay


def calculate_statistics(times):
    if not times:
        return 0, 0, 0, 0, 0
    
    times_array = np.array(times)
    mean_time = np.mean(times_array)
    std_time = np.std(times_array)
    p95_time = np.percentile(times_array, 95)
    p99_time = np.percentile(times_array, 99)
    fps = 1.0 / mean_time if mean_time > 0 else 0
    
    return mean_time, std_time, p95_time, p99_time, fps


def save_benchmark_results(results, output_path):
    with open(output_path, 'w') as f:
        f.write("Baseline Video Pipeline Benchmark Results\n")
        f.write("=" * 60 + "\n\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    json_path = str(Path(output_path).with_suffix('.json'))
    meta = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
        'hostname': socket.gethostname(),
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
    }
    results_with_meta = {'metadata': meta, 'results': results}
    with open(json_path, 'w') as jf:
        json.dump(results_with_meta, jf, indent=2)

def run_benchmark(args, model, loader, video_writer=None):
    """
    Run the benchmark loop. Returns the results dict.
    """
    def _resolve_path(p):
        p = os.path.expanduser(p)
        p = Path(p)
        if not p.is_absolute():
            p = (Path(__file__).resolve().parent / p).resolve()
        return str(p)

    inference_times = []
    postprocess_times = []
    frame_times = []
    memory_samples = []
    try:
        process = psutil.Process(os.getpid())
    except Exception:
        process = None

    print("Processing frames...")

    max_frames_arg = args.max_frames if args.max_frames else None
    warmup = max(0, args.warmup_frames)

    frames_read = 0
    frames_timed = 0

    for frame_tensor, original_frame in loader:
        frames_read += 1

        if max_frames_arg is not None and frames_timed >= max_frames_arg:
            break

        if frames_read <= warmup:
            with torch.inference_mode():
                _ = model(frame_tensor)
            if process is not None:
                memory_samples.append(process.memory_info().rss / 1024.0 / 1024.0)
            continue

        t_frame_start = time.perf_counter()

        # inference
        t_inference_start = time.perf_counter()
        with torch.inference_mode():
            output = model(frame_tensor)
        t_inference_end = time.perf_counter()
        inference_times.append(t_inference_end - t_inference_start)

        # postprocess / write overlay if requested
        t_postprocess_start = time.perf_counter()
        if video_writer:
            frame_bgr = cv2.cvtColor(original_frame, cv2.COLOR_RGB2BGR)
            frame_resized = cv2.resize(frame_bgr, (loader.target_width, loader.target_height))
            frame_overlay = overlay_lanes_on_frame(
                frame_resized, output,
                img_height=loader.target_height, img_width=loader.target_width
            )
            video_writer.write(frame_overlay)
        t_postprocess_end = time.perf_counter()
        postprocess_times.append(t_postprocess_end - t_postprocess_start)

        t_frame_end = time.perf_counter()
        frame_times.append(t_frame_end - t_frame_start)

        if process is not None:
            memory_samples.append(process.memory_info().rss / 1024.0 / 1024.0)

        frames_timed += 1

    # cleanup
    loader.release()
    if video_writer:
        video_writer.release()

    # collect loader metrics and compute statistics
    loader_metrics = loader.get_metrics()
    load_mean, load_std, load_p95, load_p99, _ = calculate_statistics(loader_metrics['load_times'])
    preprocess_mean, preprocess_std, preprocess_p95, preprocess_p99, _ = calculate_statistics(loader_metrics['preprocess_times'])
    inference_mean, inference_std, inference_p95, inference_p99, _ = calculate_statistics(inference_times)
    postprocess_mean, postprocess_std, postprocess_p95, postprocess_p99, _ = calculate_statistics(postprocess_times)
    frame_mean, frame_std, frame_p95, frame_p99, frame_fps = calculate_statistics(frame_times)

    results = {
        'Total Frames Processed': frames_timed,
        'Video FPS': f"{loader.fps:.2f}",
        'Data Loading Time (mean)': f"{load_mean*1000:.2f} ms",
        'Data Loading Time (std)': f"{load_std*1000:.2f} ms",
        'Data Loading Time (p95)': f"{load_p95*1000:.2f} ms",
        'Data Loading Time (p99)': f"{load_p99*1000:.2f} ms",
        'Preprocessing Time (mean)': f"{preprocess_mean*1000:.2f} ms",
        'Preprocessing Time (std)': f"{preprocess_std*1000:.2f} ms",
        'Preprocessing Time (p95)': f"{preprocess_p95*1000:.2f} ms",
        'Preprocessing Time (p99)': f"{preprocess_p99*1000:.2f} ms",
        'Inference Time (mean)': f"{inference_mean*1000:.2f} ms",
        'Inference Time (std)': f"{inference_std*1000:.2f} ms",
        'Inference Time (p95)': f"{inference_p95*1000:.2f} ms",
        'Inference Time (p99)': f"{inference_p99*1000:.2f} ms",
        'Postprocessing Time (mean)': f"{postprocess_mean*1000:.2f} ms",
        'Postprocessing Time (std)': f"{postprocess_std*1000:.2f} ms",
        'Postprocessing Time (p95)': f"{postprocess_p95*1000:.2f} ms",
        'Postprocessing Time (p99)': f"{postprocess_p99*1000:.2f} ms",
        'Total Time per Frame (mean)': f"{frame_mean*1000:.2f} ms",
        'Total Time per Frame (std)': f"{frame_std*1000:.2f} ms",
        'Total Time per Frame (p95)': f"{frame_p95*1000:.2f} ms",
        'Total Time per Frame (p99)': f"{frame_p99*1000:.2f} ms",
        'Processing FPS': f"{frame_fps:.2f}",
    }

    if memory_samples:
        m_arr = np.array(memory_samples)
        results['CPU Memory (mean MB)'] = f"{m_arr.mean():.2f}"
        results['CPU Memory (p95 MB)'] = f"{np.percentile(m_arr,95):.2f}"

    print("\n" + "=" * 60)
    print("Baseline Video Pipeline Benchmark Results")
    print("=" * 60)
    for key, value in results.items():
        if key.strip():
            print(f"{key}: {value}")

    benchmark_output_path = _resolve_path(args.benchmark_output)
    save_benchmark_results(results, benchmark_output_path)
    print(f"\nBenchmark results saved to: {benchmark_output_path}")

    if args.output_video:
        print(f"Output video saved to: {args.output_video}")

    return results

def main():
    UFLD_PATH = PROJECT_ROOT / "models" / "Ultra-Fast-Lane-Detection-v2"
    
    parser = argparse.ArgumentParser(description='Baseline Video Pipeline for UFLDv2')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--checkpoint', type=str,
                        default=str(UFLD_PATH / "weights" / "tusimple_res18.pth"),
                        help='Path to model checkpoint')
    parser.add_argument('--output-video', type=str, default=None,
                        help='Path to save output video with lane overlays')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum number of frames to process')
    parser.add_argument('--benchmark-output', type=str, default='baseline_benchmark_results.txt',
                        help='Path to save benchmark results')
    parser.add_argument('--warmup-frames', type=int, default=3,
                        help='Number of warm-up frames to skip from timing')
    
    args = parser.parse_args()
    
    def _resolve_path(p):
        p = os.path.expanduser(p)
        p = Path(p)
        if not p.is_absolute():
            p = (Path(__file__).resolve().parent / p).resolve()
        return str(p)
    
    checkpoint_path = _resolve_path(args.checkpoint)
    
    device = torch.device('cpu') # CPU-only pipeline
    model = load_ufldv2_model(Path(checkpoint_path), device=device)
    
    print(f"Loading video: {args.video}")
    loader = VideoFrameLoader(args.video, target_height=cfg.train_height, target_width=cfg.train_width)
    print(f"Total frames: {len(loader)}, FPS: {loader.fps}")
    
    video_writer = None
    if args.output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.output_video, fourcc, loader.fps, (800, 320))
    
    run_benchmark(args, model, loader, video_writer)


if __name__ == '__main__':
    main()
