"""
GPU-accelerated video processing pipeline, DataLoader-style batching,
multi-worker parallel loading, optional caching, FP16 inference (CUDA-only),
and WandB logging. Includes comprehensive benchmarking with FPS, latency, and GPU metrics.
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import psutil
import platform
import json
try:
    import wandb
except Exception:
    wandb = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
UFLD_PATH = PROJECT_ROOT / "models" / "Ultra-Fast-Lane-Detection-v2"
sys.path.insert(0, str(UFLD_PATH))

from model.model_culane import parsingNet
import configs.tusimple_res18 as cfg

ROW_ANCHOR = np.linspace(160, 710, cfg.num_row) / 720.0
COL_ANCHOR = np.linspace(0.0, 1.0, cfg.num_col)


class VideoDataset(Dataset):
    def __init__(self, video_path, target_height=None, target_width=None, cache_mode="preload",
                 cache_dir=None, rebuild_cache=False):
        self.video_path = str(video_path)
        self.target_height = int(target_height if target_height is not None else cfg.train_height)
        self.target_width = int(target_width if target_width is not None else cfg.train_width)
        self.cache_mode = cache_mode
        self.cache_dir = str(cache_dir) if cache_dir else None
        self.rebuild_cache = rebuild_cache

        self.frames = []
        self.tensor_cache_index = []
        self.metrics = {
            'preload_total_time': 0.0,
            'cache_build_time': 0.0,
            'cache_load_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'load_times': [],
            'preprocess_times': [],
        }

        if self.cache_mode == "tensor_cache":
            self._prepare_tensor_cache()
        else:
            self._load_video_preload()

    def _prepare_tensor_cache(self):
        t0 = time.perf_counter()
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()

        cache_base = Path(self.cache_dir) if self.cache_dir else (Path(self.video_path).with_suffix('') )
        cache_base = cache_base if cache_base.is_absolute() else (Path(__file__).resolve().parent / cache_base)
        cache_base.mkdir(parents=True, exist_ok=True)
        cache_file = cache_base / (Path(self.video_path).stem + f"_{self.target_height}x{self.target_width}.pt")

        if self.rebuild_cache or not cache_file.exists():
            t_build0 = time.perf_counter()
            tensors = []
            cap = cv2.VideoCapture(self.video_path)
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                t_load = time.perf_counter()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cut_height = int(cfg.train_height * (1 - cfg.crop_ratio))
                frame_crop = frame_rgb[cut_height:, :, :]
                frame_resized = cv2.resize(frame_crop, (self.target_width, self.target_height))
                tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
                mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1)
                tensor = (tensor - mean) / std
                tensors.append(tensor)
                t_pre = time.perf_counter()
                self.metrics['load_times'].append(t_load - t_build0)
                self.metrics['preprocess_times'].append(t_pre - t_load)
                idx += 1
            cap.release()
            torch.save({'tensors': tensors, 'total': idx}, cache_file)
            self.metrics['cache_build_time'] = time.perf_counter() - t_build0
        t_load_cache0 = time.perf_counter()
        data = torch.load(cache_file, map_location='cpu')
        self.tensor_cache_index = list(range(int(data.get('total', len(data['tensors'])))))
        self.tensor_cache = data['tensors']
        self.metrics['cache_load_time'] = time.perf_counter() - t_load_cache0
        self.metrics['preload_total_time'] = time.perf_counter() - t0

    def _load_video_preload(self):
        t0 = time.perf_counter()
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.frames.append(frame)
        cap.release()
        self.metrics['preload_total_time'] = time.perf_counter() - t0

    def __len__(self):
        if self.cache_mode == "tensor_cache":
            return len(self.tensor_cache_index)
        return len(self.frames)

    def __getitem__(self, idx):
        if self.cache_mode == "tensor_cache":
            self.metrics['cache_hits'] += 1
            return self.tensor_cache[idx], idx

        frame = self.frames[idx]
        t_load = time.perf_counter()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cut_height = int(cfg.train_height * (1 - cfg.crop_ratio))
        frame_crop = frame_rgb[cut_height:, :, :]
        frame_resized = cv2.resize(frame_crop, (self.target_width, self.target_height))
        tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1)
        tensor = (tensor - mean) / std
        t_pre = time.perf_counter()
        self.metrics['load_times'].append(t_load - t_load)  # ~0 for preload access
        self.metrics['preprocess_times'].append(t_pre - t_load)
        return tensor, idx


class OptimizedVideoLoader:
    def __init__(self, video_path, batch_size=8, num_workers=4,
                 target_height=None, target_width=None, use_fp16=True, device='cuda',
                 cache_mode="preload", cache_dir=None, rebuild_cache=False):
        self.video_path = video_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_height = int(target_height if target_height is not None else cfg.train_height)
        self.target_width = int(target_width if target_width is not None else cfg.train_width)
        self.use_fp16 = use_fp16
        self.device = device
        
        self.dataset = VideoDataset(video_path, self.target_height, self.target_width,
                                    cache_mode=cache_mode, cache_dir=cache_dir, rebuild_cache=rebuild_cache)
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False
        )
        
        cap = cv2.VideoCapture(video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        self.metrics = {
            'batch_load_times': [],
            'gpu_transfer_times': [],
            'total_batch_times': []
        }
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)
    
    def get_total_frames(self):
        return len(self.dataset)
    
    def get_metrics(self):
        return self.metrics
    
    def process_batch(self, batch_tensors):
        t_start = time.perf_counter()
        batch_tensors = batch_tensors.to(self.device, non_blocking=True)
        if self.use_fp16 and self.device == 'cuda':
            batch_tensors = batch_tensors.half()
        if self.device == 'cuda':
            torch.cuda.synchronize()
        t_transfer = time.perf_counter()
        self.metrics['gpu_transfer_times'].append(t_transfer - t_start)
        return batch_tensors


def build_model(device: torch.device):
    model = parsingNet(
        pretrained=True,
        backbone=cfg.backbone,
        num_grid_row=cfg.num_cell_row,
        num_cls_row=cfg.num_row,
        num_grid_col=cfg.num_cell_col,
        num_cls_col=cfg.num_col,
        num_lane_on_row=cfg.num_lanes,
        num_lane_on_col=cfg.num_lanes,
        use_aux=cfg.use_aux,
        input_height=cfg.train_height,
        input_width=cfg.train_width,
        fc_norm=cfg.fc_norm,
    ).to(device)
    return model

def load_ufldv2_model(ckpt_path: Path, device: torch.device, use_fp16: bool=False):
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
    if use_fp16 and str(device) == 'cuda':
        model.half()
    model.eval()
    print("Model loaded & set to eval()")
    return model


def pred2coords(pred, img_w, img_h, local_width=1):
    batch_size, num_grid_row, num_cls_row, num_lane_row = pred["loc_row"].shape
    batch_size, num_grid_col, num_cls_col, num_lane_col = pred["loc_col"].shape
    max_indices_row = pred["loc_row"].argmax(1).cpu()
    valid_row = pred["exist_row"].argmax(1).cpu()
    max_indices_col = pred["loc_col"].argmax(1).cpu()
    valid_col = pred["exist_col"].argmax(1).cpu()
    pred["loc_row"] = pred["loc_row"].cpu()
    pred["loc_col"] = pred["loc_col"].cpu()
    coords = []
    row_lane_idx = [1, 2]
    col_lane_idx = [0, 3]
    for i in row_lane_idx:
        tmp = []
        if valid_row[0, :, i].sum() > num_cls_row / 2:
            for k in range(valid_row.shape[1]):
                if valid_row[0, k, i]:
                    all_ind = torch.tensor(list(range(max(0, max_indices_row[0, k, i] - local_width),
                                                      min(num_grid_row - 1, max_indices_row[0, k, i] + local_width) + 1)))
                    out_tmp = (pred["loc_row"][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_row - 1) * img_w
                    x = int(out_tmp)
                    y = int(ROW_ANCHOR[k] * img_h)
                    tmp.append((x, y))
            if tmp:
                coords.append(tmp)
    for i in col_lane_idx:
        tmp = []
        if valid_col[0, :, i].sum() > num_cls_col / 4:
            for k in range(valid_col.shape[1]):
                if valid_col[0, k, i]:
                    all_ind = torch.tensor(list(range(max(0, max_indices_col[0, k, i] - local_width),
                                                      min(num_grid_col - 1, max_indices_col[0, k, i] + local_width) + 1)))
                    out_tmp = (pred["loc_col"][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
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
    except Exception:
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
        f.write("Optimized Video Pipeline Benchmark Results\n")
        f.write("=" * 60 + "\n\n")
        
        for key, value in results.items():
            f.write(f"{key}: {value}\n")


def main():
    parser = argparse.ArgumentParser(description='Optimized Video Pipeline for UFLDv2')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--checkpoint', type=str,
                        default=str(UFLD_PATH / 'weights' / 'tusimple_res18.pth'),
                        help='Path to model checkpoint')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for processing')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--fp16', action='store_true', help='Enable FP16 inference (CUDA only)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--output-video', type=str, default=None,
                        help='Path to save output video with lane overlays')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum number of frames to process')
    parser.add_argument('--benchmark-output', type=str, default='optimized_benchmark_results.txt',
                        help='Path to save benchmark results')
    parser.add_argument('--target-height', type=int, default=cfg.train_height, help='Model input height')
    parser.add_argument('--target-width', type=int, default=cfg.train_width, help='Model input width')
    parser.add_argument('--warmup-frames', type=int, default=3, help='Number of warm-up frames to skip from timing')
    parser.add_argument('--cache-mode', type=str, default='preload', choices=['preload', 'tensor_cache'], help='Dataset cache mode')
    parser.add_argument('--cache-dir', type=str, default=None, help='Cache directory for tensor_cache mode')
    parser.add_argument('--rebuild-cache', action='store_true', help='Force rebuild of tensor cache')
    parser.add_argument('--use-wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='route-overlay-optimization', help='WandB project name')
    parser.add_argument('--run-name', type=str, default=None, help='WandB run name')
    parser.add_argument('--wandb-log-interval', type=int, default=10, help='Log every N batches')
    
    args = parser.parse_args()
    
    def _resolve_path(p):
        p = os.path.expanduser(p)
        p = Path(p)
        if not p.is_absolute():
            p = (Path(__file__).resolve().parent / p).resolve()
        return str(p)

    use_fp16 = bool(args.fp16 and args.device == 'cuda')
    checkpoint_path = _resolve_path(args.checkpoint)
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
        use_fp16 = False
    
    print("Loading UFLDv2 model...")
    device = torch.device(args.device)
    model = load_ufldv2_model(Path(checkpoint_path), device=device, use_fp16=use_fp16)
    print(f"Model loaded successfully on {args.device}" + (" with FP16" if use_fp16 else ""))
    
    print(f"Loading video: {args.video}")
    loader = OptimizedVideoLoader(
        args.video,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_height=args.target_height,
        target_width=args.target_width,
        use_fp16=use_fp16,
        device=args.device,
        cache_mode=args.cache_mode,
        cache_dir=args.cache_dir,
        rebuild_cache=args.rebuild_cache
    )
    print(f"Total frames: {loader.get_total_frames()}, FPS: {loader.fps}")
    print(f"Total batches: {len(loader)}, Batch size: {args.batch_size}")
    
    video_writer = None
    if args.output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.output_video, fourcc, loader.fps, (loader.target_width, loader.target_height))
    
    dataloader_times = []
    gpu_transfer_times = []
    inference_times = []
    postprocess_times = []
    batch_times = []
    per_frame_times = []
    memory_samples = []
    process = None
    try:
        process = psutil.Process(os.getpid())
    except Exception:
        process = None
    
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated() / 1024 / 1024
    
    # WandB init
    if args.use_wandb and wandb is not None:
        run_name = args.run_name
        if not run_name:
            base = Path(args.video).stem
            run_name = f"gpu-optimized-{base}-{args.target_height}x{args.target_width}-workers{args.num_workers}-bs{args.batch_size}-{time.strftime('%Y%m%d_%H%M%S', time.localtime())}"
        wandb.init(
            project=args.wandb_project,
            group=f"gpu-optimized-{Path(args.video).stem}",
            name=run_name,
            config={
                "device": args.device,
                "fp16": use_fp16,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "video_path": args.video,
                "video_fps": loader.fps,
                "cache_mode": args.cache_mode,
                "target_height": args.target_height,
                "target_width": args.target_width,
                "model_train_height": cfg.train_height,
                "model_train_width": cfg.train_width,
                "backbone": cfg.backbone,
                "crop_ratio": cfg.crop_ratio,
                "num_row": cfg.num_row,
                "num_col": cfg.num_col,
                "num_lanes": cfg.num_lanes,
            },
        )
    
    print("Processing batches...")
    frame_count = 0
    max_frames = args.max_frames if args.max_frames else loader.get_total_frames()
    batch_count = 0
    warmup = max(0, args.warmup_frames)
    
    for batch_tensors, indices in loader:
        if frame_count >= max_frames:
            break
        
        t_batch_start = time.time()
        
        t_transfer_start = time.perf_counter()
        batch_tensors = batch_tensors.to(args.device, non_blocking=True)
        if use_fp16 and args.device == 'cuda':
            batch_tensors = batch_tensors.half()
        if args.device == 'cuda':
            torch.cuda.synchronize()
        t_transfer_end = time.perf_counter()
        gpu_transfer_times.append(t_transfer_end - t_transfer_start)
        
        # warmup skip timing
        if frame_count < warmup:
            with torch.inference_mode():
                _ = model(batch_tensors)
            current_batch_size = len(indices)
            frame_count += current_batch_size
            batch_count += 1
            continue

        t_inference_start = time.perf_counter()
        with torch.inference_mode():
            outputs = model(batch_tensors)
        if args.device == 'cuda':
            torch.cuda.synchronize()
        t_inference_end = time.perf_counter()
        inference_times.append(t_inference_end - t_inference_start)
        
        t_postprocess_start = time.perf_counter()
        if video_writer:
            for i, idx in enumerate(indices):
                idx_val = idx.item()
                if idx_val >= max_frames:
                    continue
                canvas = np.zeros((loader.target_height, loader.target_width, 3), dtype=np.uint8)
                if isinstance(outputs, dict):
                    output_single = {k: v[i:i+1] for k, v in outputs.items()}
                else:
                    output_single = outputs[i:i+1]
                frame_overlay = overlay_lanes_on_frame(canvas, output_single,
                                                       img_height=loader.target_height, img_width=loader.target_width)
                video_writer.write(frame_overlay)
        t_postprocess_end = time.perf_counter()
        postprocess_times.append(t_postprocess_end - t_postprocess_start)
        
        t_batch_end = time.perf_counter()
        batch_time = t_batch_end - t_batch_start
        batch_times.append(batch_time)
        
        current_batch_size = len(indices)
        frame_count += current_batch_size
        
        for _ in range(current_batch_size):
            per_frame_times.append(batch_time / current_batch_size)
        
        batch_count += 1
        
        if batch_count % 10 == 0:
            print(f"Processed {frame_count}/{max_frames} frames ({batch_count} batches)")

        if args.use_wandb and wandb is not None and batch_count % max(1, args.wandb_log_interval) == 0:
            wandb.log({
                "batch_idx": batch_count,
                "frames_processed": frame_count,
                "gpu_transfer_time_ms": gpu_transfer_times[-1] * 1000,
                "inference_time_ms": inference_times[-1] * 1000,
                "postprocess_time_ms": postprocess_times[-1] * 1000,
                "total_batch_time_ms": batch_time * 1000,
                "batch_fps": (current_batch_size / batch_time) if batch_time > 0 else 0.0,
                "dataset_preprocess_mean_ms": (np.mean(loader.dataset.metrics['preprocess_times']) * 1000) if loader.dataset.metrics['preprocess_times'] else 0.0,
            })
    
    if video_writer:
        video_writer.release()
    
    if args.device == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        memory_used = peak_memory - initial_memory
    
    loader_metrics = loader.get_metrics()
    
    transfer_mean, transfer_std, transfer_p95, transfer_p99, _ = calculate_statistics(gpu_transfer_times)
    inference_mean, inference_std, inference_p95, inference_p99, _ = calculate_statistics(inference_times)
    postprocess_mean, postprocess_std, postprocess_p95, postprocess_p99, _ = calculate_statistics(postprocess_times)
    batch_mean, batch_std, batch_p95, batch_p99, batch_fps = calculate_statistics(batch_times)
    frame_mean, frame_std, frame_p95, frame_p99, frame_fps = calculate_statistics(per_frame_times)
    
    results = {
        'Total Frames Processed': frame_count,
        'Total Batches Processed': batch_count,
        'Batch Size': args.batch_size,
        'Num Workers': args.num_workers,
        'Video FPS': f"{loader.fps:.2f}",
        'Device': args.device,
        'FP16 Enabled': use_fp16,
        'Target Height': args.target_height,
        'Target Width': args.target_width,
        'Cache Mode': args.cache_mode,
        'GPU Transfer Time per Batch (mean)': f"{transfer_mean*1000:.2f} ms",
        'GPU Transfer Time per Batch (std)': f"{transfer_std*1000:.2f} ms",
        'GPU Transfer Time per Batch (p95)': f"{transfer_p95*1000:.2f} ms",
        'GPU Transfer Time per Batch (p99)': f"{transfer_p99*1000:.2f} ms",
        'Inference Time per Batch (mean)': f"{inference_mean*1000:.2f} ms",
        'Inference Time per Batch (std)': f"{inference_std*1000:.2f} ms",
        'Inference Time per Batch (p95)': f"{inference_p95*1000:.2f} ms",
        'Inference Time per Batch (p99)': f"{inference_p99*1000:.2f} ms",
        'Postprocessing Time per Batch (mean)': f"{postprocess_mean*1000:.2f} ms",
        'Postprocessing Time per Batch (std)': f"{postprocess_std*1000:.2f} ms",
        'Postprocessing Time per Batch (p95)': f"{postprocess_p95*1000:.2f} ms",
        'Postprocessing Time per Batch (p99)': f"{postprocess_p99*1000:.2f} ms",
        'Total Time per Batch (mean)': f"{batch_mean*1000:.2f} ms",
        'Total Time per Batch (std)': f"{batch_std*1000:.2f} ms",
        'Total Time per Batch (p95)': f"{batch_p95*1000:.2f} ms",
        'Total Time per Batch (p99)': f"{batch_p99*1000:.2f} ms",
        'Batch Processing FPS': f"{batch_fps:.2f}",
        'Time per Frame (mean)': f"{frame_mean*1000:.2f} ms",
        'Time per Frame (std)': f"{frame_std*1000:.2f} ms",
        'Time per Frame (p95)': f"{frame_p95*1000:.2f} ms",
        'Time per Frame (p99)': f"{frame_p99*1000:.2f} ms",
        'Frame Processing FPS': f"{frame_fps:.2f}",
    }
    
    if args.device == 'cuda':
        results['GPU Memory Used (MB)'] = f"{memory_used:.2f}"
        results['Peak GPU Memory (MB)'] = f"{peak_memory:.2f}"
    
    # include dataset/cache metrics
    results['Dataset Preload Time (s)'] = f"{loader.dataset.metrics.get('preload_total_time', 0.0):.2f}"
    results['Cache Build Time (s)'] = f"{loader.dataset.metrics.get('cache_build_time', 0.0):.2f}"
    results['Cache Load Time (s)'] = f"{loader.dataset.metrics.get('cache_load_time', 0.0):.2f}"
    results['Cache Hits'] = loader.dataset.metrics.get('cache_hits', 0)
    results['Cache Misses'] = loader.dataset.metrics.get('cache_misses', 0)

    print("\n" + "=" * 60)
    print("Optimized Video Pipeline Benchmark Results")
    print("=" * 60)
    for key, value in results.items():
        print(f"{key}: {value}")
    
    # add date stamp to avoid overwrite
    date_stamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    out_path = Path(_resolve_path(args.benchmark_output))
    benchmark_output_path = str(out_path.with_name(f"{out_path.stem}_{date_stamp}{out_path.suffix}"))
    save_benchmark_results(results, benchmark_output_path)
    print(f"\nBenchmark results saved to: {benchmark_output_path}")
    
    if args.output_video:
        print(f"Output video saved to: {args.output_video}")

    if args.use_wandb and wandb is not None:
        wandb.log({
            "summary/total_frames": frame_count,
            "summary/video_fps": loader.fps,
            "summary/transfer_mean_ms": transfer_mean * 1000,
            "summary/inference_mean_ms": inference_mean * 1000,
            "summary/postprocess_mean_ms": postprocess_mean * 1000,
            "summary/batch_mean_ms": batch_mean * 1000,
            "summary/frame_mean_ms": frame_mean * 1000,
            "summary/fps": frame_fps,
        })
        wandb.finish()


if __name__ == '__main__':
    main()
