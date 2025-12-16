"""
Optimization Technique 2: Mixed Precision Inference (FP16)
Uses automatic mixed precision to reduce memory usage and improve inference speed
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import json
import platform

import cv2
import numpy as np
import torch
import torch.profiler
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
import wandb

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
UFLD_PATH = PROJECT_ROOT / "models" / "Ultra-Fast-Lane-Detection-v2"

sys.path.insert(0, str(UFLD_PATH))

from model.model_culane import parsingNet
import configs.tusimple_res18 as cfg

ROW_ANCHOR = np.linspace(160, 710, cfg.num_row) / 720.0
COL_ANCHOR = np.linspace(0.0, 1.0, cfg.num_col)


class LaneDataset(Dataset):
    """Dataset for loading images for inference benchmarking"""
    
    def __init__(self, image_paths: List[Path], transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(str(img_path))
        
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {img_path}")
        
        ori_h, ori_w = img.shape[:2]
        
        # Crop and resize
        cut_height = int(cfg.train_height * (1 - cfg.crop_ratio))
        img_crop = img[cut_height:, :, :]
        img_resized = cv2.resize(
            img_crop, (cfg.train_width, cfg.train_height), 
            interpolation=cv2.INTER_CUBIC
        )
        
        # Normalize
        img_norm = img_resized[:, :, ::-1].astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (img_norm - mean) / std
        
        # HWC → CHW
        img_chw = np.transpose(img_norm, (2, 0, 1))
        
        return torch.from_numpy(img_chw).float(), ori_w, ori_h, str(img_path)


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


def load_model(ckpt_path: Path, device: torch.device, use_fp16: bool = False):
    print(f"Loading model from: {ckpt_path}")
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
    
    model.load_state_dict(new_state, strict=False)
    model.eval()
    
    if use_fp16:
        print("Converting model to FP16...")
        model = model.half()
        print("Model converted to FP16")
    
    print("Model loaded successfully\n")
    return model


def get_image_paths(dataset_dir: Path, max_images: int = None) -> List[Path]:
    """Collect all image paths from dataset directory"""
    image_paths = []
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    for ext in extensions:
        image_paths.extend(dataset_dir.rglob(f'*{ext}'))
    
    image_paths = sorted(image_paths)
    
    if max_images:
        image_paths = image_paths[:max_images]
    
    print(f"Found {len(image_paths)} images")
    return image_paths


def get_gpu_info():
    """Get GPU information for logging"""
    if not torch.cuda.is_available():
        return {"gpu_available": False}
    
    gpu_info = {
        "gpu_available": True,
        "gpu_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "tensor_cores_available": torch.cuda.get_device_capability(0)[0] >= 7,  # Volta and newer
    }
    
    return gpu_info


def get_system_info():
    """Get system information for logging"""
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cpu_count": os.cpu_count(),
        "hostname": platform.node(),
    }


class BenchmarkMetrics:
    """Track and compute benchmark metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.data_loading_times = []
        self.inference_times = []
        self.total_times = []
        self.gpu_memory_used = []
        self.gpu_memory_allocated = []
        self.batch_sizes = []
    
    def add_batch_metrics(self, data_load_time, inference_time, 
                         gpu_memory, gpu_memory_alloc, batch_size):
        self.data_loading_times.append(data_load_time)
        self.inference_times.append(inference_time)
        self.total_times.append(data_load_time + inference_time)
        self.gpu_memory_used.append(gpu_memory)
        self.gpu_memory_allocated.append(gpu_memory_alloc)
        self.batch_sizes.append(batch_size)
    
    def compute_statistics(self) -> Dict:
        """Compute summary statistics"""
        total_images = sum(self.batch_sizes)
        total_time = sum(self.inference_times)
        
        return {
            "data_loading": {
                "mean_ms": np.mean(self.data_loading_times) * 1000,
                "std_ms": np.std(self.data_loading_times) * 1000,
                "median_ms": np.median(self.data_loading_times) * 1000,
            },
            "inference": {
                "mean_ms": np.mean(self.inference_times) * 1000,
                "std_ms": np.std(self.inference_times) * 1000,
                "min_ms": np.min(self.inference_times) * 1000,
                "max_ms": np.max(self.inference_times) * 1000,
                "median_ms": np.median(self.inference_times) * 1000,
                "p95_ms": np.percentile(self.inference_times, 95) * 1000,
                "p99_ms": np.percentile(self.inference_times, 99) * 1000,
            },
            "total_pipeline": {
                "mean_ms": np.mean(self.total_times) * 1000,
                "std_ms": np.std(self.total_times) * 1000,
                "median_ms": np.median(self.total_times) * 1000,
            },
            "throughput": {
                "images_per_second": total_images / total_time if total_time > 0 else 0,
                "batches_per_second": len(self.inference_times) / total_time if total_time > 0 else 0,
                "mean_batch_fps": 1.0 / np.mean(self.inference_times),
            },
            "gpu_memory": {
                "mean_allocated_mb": np.mean(self.gpu_memory_allocated),
                "max_allocated_mb": np.max(self.gpu_memory_allocated),
                "mean_reserved_mb": np.mean(self.gpu_memory_used),
                "max_reserved_mb": np.max(self.gpu_memory_used),
            }
        }


def benchmark_inference(model, dataloader, device, precision_mode, batch_size,
                       num_warmup=10, log_frequency=10):
    """Run inference benchmark with specified precision"""
    metrics = BenchmarkMetrics()
    
    print(f"\n{'='*60}")
    print(f"PRECISION MODE: {precision_mode.upper()}")
    print(f"Batch Size: {batch_size}")
    print(f"{'='*60}\n")
    
    use_amp = (precision_mode == "amp")
    use_fp16_input = (precision_mode == "fp16")
    
    print(f"Running warmup ({num_warmup} iterations)...")
    with torch.no_grad():
        for i, (imgs, _, _, _) in enumerate(dataloader):
            if i >= num_warmup:
                break
            imgs = imgs.to(device)
            if use_fp16_input:
                imgs = imgs.half()
            
            if use_amp:
                with autocast():
                    _ = model(imgs)
            else:
                _ = model(imgs)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    print("Warmup complete. Starting benchmark...\n")

    total_images = 0
    batch_count = 0
    
    with torch.no_grad():
        for batch_idx, (imgs, ori_ws, ori_hs, paths) in enumerate(dataloader):
            # data loading time
            data_load_start = time.perf_counter()
            imgs = imgs.to(device)
            if use_fp16_input:
                imgs = imgs.half()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            data_load_time = time.perf_counter() - data_load_start
            
            # inference time
            inference_start = time.perf_counter()
            if use_amp:
                with autocast():
                    pred = model(imgs)
            else:
                pred = model(imgs)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            inference_time = time.perf_counter() - inference_start
            
            # GPU memory usage
            if device.type == 'cuda':
                gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**2  # MB
                gpu_mem_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            else:
                gpu_mem_reserved = 0
                gpu_mem_allocated = 0
            
            current_batch_size = imgs.shape[0]
            metrics.add_batch_metrics(
                data_load_time, inference_time, 
                gpu_mem_reserved, gpu_mem_allocated, current_batch_size
            )
            
            total_images += current_batch_size
            batch_count += 1
  
            if batch_count % log_frequency == 0:
                images_per_sec = current_batch_size / inference_time if inference_time > 0 else 0
                wandb.log({
                    f"{precision_mode}/batch_inference_time_ms": inference_time * 1000,
                    f"{precision_mode}/batch_data_loading_time_ms": data_load_time * 1000,
                    f"{precision_mode}/batch_images_per_second": images_per_sec,
                    f"{precision_mode}/batch_gpu_memory_allocated_mb": gpu_mem_allocated,
                    f"{precision_mode}/batch_gpu_memory_reserved_mb": gpu_mem_reserved,
                    f"{precision_mode}/batch_latency_per_image_ms": (inference_time * 1000) / current_batch_size,
                    "progress/total_images_processed": total_images,
                })
                
                print(f"[{precision_mode.upper()}] Processed {total_images} images... "
                      f"Inference: {inference_time*1000:.2f}ms, "
                      f"Images/sec: {images_per_sec:.1f}")
    
    print(f"\n[{precision_mode.upper()}] Completed benchmark on {total_images} images\n")
    
    return metrics


def print_results(precision_mode: str, metrics: BenchmarkMetrics, 
                 baseline_stats: Dict = None, save_path: Path = None):
    """Print and save benchmark results"""
    stats = metrics.compute_statistics()
    
    print(f"\n{'='*60}")
    print(f"{precision_mode.upper()} RESULTS")
    print(f"{'='*60}\n")
    
    print("INFERENCE:")
    print(f"  Mean:   {stats['inference']['mean_ms']:.2f} ms")
    print(f"  Median: {stats['inference']['median_ms']:.2f} ms")
    print(f"  Std:    {stats['inference']['std_ms']:.2f} ms")
    print(f"  P95:    {stats['inference']['p95_ms']:.2f} ms")
    print(f"  P99:    {stats['inference']['p99_ms']:.2f} ms\n")
    
    print("THROUGHPUT:")
    print(f"  Images/second: {stats['throughput']['images_per_second']:.2f}\n")
    
    print("GPU MEMORY:")
    print(f"  Peak Allocated: {stats['gpu_memory']['max_allocated_mb']:.2f} MB")
    print(f"  Peak Reserved:  {stats['gpu_memory']['max_reserved_mb']:.2f} MB\n")
    
    if baseline_stats:
        baseline_fps = baseline_stats['throughput']['images_per_second']
        current_fps = stats['throughput']['images_per_second']
        speedup = current_fps / baseline_fps
        
        baseline_mem = baseline_stats['gpu_memory']['max_allocated_mb']
        current_mem = stats['gpu_memory']['max_allocated_mb']
        memory_reduction = (baseline_mem - current_mem) / baseline_mem * 100
        
        baseline_latency = baseline_stats['inference']['mean_ms']
        current_latency = stats['inference']['mean_ms']
        latency_improvement = baseline_latency / current_latency
        
        print("COMPARISON TO FP32 BASELINE:")
        print(f"  Throughput Speedup: {speedup:.2f}x ({baseline_fps:.1f} → {current_fps:.1f} images/s)")
        print(f"  Latency Improvement: {latency_improvement:.2f}x ({baseline_latency:.2f}ms → {current_latency:.2f}ms)")
        print(f"  Memory Reduction: {memory_reduction:.1f}% ({baseline_mem:.1f}MB → {current_mem:.1f}MB)")
        print(f"  Memory Saved: {baseline_mem - current_mem:.2f} MB\n")
    
    print(f"{'='*60}\n")
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Results saved to: {save_path}\n")
    
    return stats


def create_comparison_plots(fp32_stats: Dict, fp16_stats: Dict, amp_stats: Dict):
    """Create comparison visualizations"""
    
    modes = ['FP32', 'FP16', 'AMP']
    throughputs = [
        fp32_stats['throughput']['images_per_second'],
        fp16_stats['throughput']['images_per_second'],
        amp_stats['throughput']['images_per_second']
    ]
    latencies = [
        fp32_stats['inference']['mean_ms'],
        fp16_stats['inference']['mean_ms'],
        amp_stats['inference']['mean_ms']
    ]
    memories = [
        fp32_stats['gpu_memory']['max_allocated_mb'],
        fp16_stats['gpu_memory']['max_allocated_mb'],
        amp_stats['gpu_memory']['max_allocated_mb']
    ]
    
    comparison_table = wandb.Table(
        columns=["Precision Mode", "Throughput (img/s)", "Latency (ms)", 
                 "Memory (MB)", "Speedup vs FP32", "Memory Reduction"],
        data=[
            [
                mode,
                f"{throughput:.2f}",
                f"{latency:.2f}",
                f"{memory:.2f}",
                f"{throughput / throughputs[0]:.2f}x",
                f"{(memories[0] - memory) / memories[0] * 100:.1f}%"
            ]
            for mode, throughput, latency, memory in zip(modes, throughputs, latencies, memories)
        ]
    )
    wandb.log({"comparison/precision_comparison_table": comparison_table})
    
    # best configuration
    best_throughput_idx = np.argmax(throughputs)
    best_memory_idx = np.argmin(memories)
    
    print(f"\n{'='*60}")
    print("PRECISION OPTIMIZATION ANALYSIS")
    print(f"{'='*60}\n")
    print(f"Best Throughput: {modes[best_throughput_idx]} ({throughputs[best_throughput_idx]:.2f} img/s)")
    print(f"  Speedup: {throughputs[best_throughput_idx] / throughputs[0]:.2f}x over FP32")
    print(f"  Latency: {latencies[best_throughput_idx]:.2f} ms")
    print(f"\nBest Memory Efficiency: {modes[best_memory_idx]} ({memories[best_memory_idx]:.2f} MB)")
    print(f"  Memory Saved: {memories[0] - memories[best_memory_idx]:.2f} MB")
    print(f"  Reduction: {(memories[0] - memories[best_memory_idx]) / memories[0] * 100:.1f}%")
    print(f"\n{'='*60}\n")
    
    wandb.log({
        "comparison/best_throughput_mode": modes[best_throughput_idx],
        "comparison/best_throughput": throughputs[best_throughput_idx],
        "comparison/max_speedup": throughputs[best_throughput_idx] / throughputs[0],
        "comparison/best_memory_mode": modes[best_memory_idx],
        "comparison/min_memory_mb": memories[best_memory_idx],
        "comparison/max_memory_reduction_pct": (memories[0] - memories[best_memory_idx]) / memories[0] * 100,
    })


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16
    num_workers = 4
    max_images = 500
    experiment_name = "ufldv2-mixed-precision-optimization-NVidia-GPU-4070"
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Mixed precision optimization requires GPU.")
        return
    
    ckpt_path = UFLD_PATH / "weights" / "tusimple_res18.pth"
    dataset_dir = PROJECT_ROOT / "datasets" / "TUSimple" / "test_set"
    results_dir = PROJECT_ROOT / "results" / "mixed_precision"
    results_dir.mkdir(parents=True, exist_ok=True)
    gpu_info = get_gpu_info()
    system_info = get_system_info()
    
    print(f"GPU: {gpu_info['gpu_name']}")
    print(f"Tensor Cores Available: {gpu_info['tensor_cores_available']}")
    print(f"CUDA Version: {gpu_info['cuda_version']}\n")
    
    # load once to get params
    temp_model = build_model(device)
    param_count = sum(p.numel() for p in temp_model.parameters())
    param_size_mb = param_count * 4 / 1024**2
    del temp_model
    torch.cuda.empty_cache()
    
    model_info = {
        "model_name": "UFLDv2",
        "backbone": cfg.backbone,
        "total_parameters": param_count,
        "model_size_fp32_mb": param_size_mb,
        "model_size_fp16_mb": param_size_mb / 2,
        "input_height": cfg.train_height,
        "input_width": cfg.train_width,
    }
    
    wandb.init(
        project="ufldv2-optimization",
        name=experiment_name,
        group="simple-optimizations",
        config={
            **model_info,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "max_images": max_images,
            "device": str(device),
            "optimization_technique": "mixed_precision_fp16",
            "precision_modes_tested": ["fp32", "fp16", "amp"],
            **system_info,
            **gpu_info,
            "dataset": "TuSimple",
        },
        tags=["optimization", "mixed-precision", "fp16", "lane-detection"],
        notes="Optimization Technique 2: Mixed Precision Inference - Comparing FP32, FP16, and AMP"
    )
    
    image_paths = get_image_paths(dataset_dir, max_images)
    dataset = LaneDataset(image_paths)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    all_results = {}
    
    # Test 1: FP32 Baseline
    print(f"\n{'#'*60}")
    print("# TEST 1: FP32 BASELINE")
    print(f"{'#'*60}\n")
    
    model_fp32 = load_model(ckpt_path, device, use_fp16=False)
    metrics_fp32 = benchmark_inference(
        model_fp32, dataloader, device, "fp32", batch_size,
        num_warmup=10, log_frequency=10
    )
    stats_fp32 = print_results("fp32", metrics_fp32, 
                               save_path=results_dir / "fp32_metrics.json")
    all_results['fp32'] = stats_fp32
    
    wandb.log({
        "summary/fp32_throughput": stats_fp32['throughput']['images_per_second'],
        "summary/fp32_latency_ms": stats_fp32['inference']['mean_ms'],
        "summary/fp32_memory_mb": stats_fp32['gpu_memory']['max_allocated_mb'],
    })
    
    del model_fp32
    torch.cuda.empty_cache()
    
    # Test 2: FP16 (model.half())
    print(f"\n{'#'*60}")
    print("# TEST 2: FP16 (NATIVE)")
    print(f"{'#'*60}\n")
    
    model_fp16 = load_model(ckpt_path, device, use_fp16=True)
    metrics_fp16 = benchmark_inference(
        model_fp16, dataloader, device, "fp16", batch_size,
        num_warmup=10, log_frequency=10
    )
    stats_fp16 = print_results("fp16", metrics_fp16, baseline_stats=stats_fp32,
                               save_path=results_dir / "fp16_metrics.json")
    all_results['fp16'] = stats_fp16
    
    wandb.log({
        "summary/fp16_throughput": stats_fp16['throughput']['images_per_second'],
        "summary/fp16_latency_ms": stats_fp16['inference']['mean_ms'],
        "summary/fp16_memory_mb": stats_fp16['gpu_memory']['max_allocated_mb'],
        "summary/fp16_speedup": stats_fp16['throughput']['images_per_second'] / stats_fp32['throughput']['images_per_second'],
    })
    
    del model_fp16
    torch.cuda.empty_cache()
    
    # Test 3: Automatic Mixed Precision (AMP)
    print(f"\n{'#'*60}")
    print("# TEST 3: AUTOMATIC MIXED PRECISION (AMP)")
    print(f"{'#'*60}\n")
    
    model_amp = load_model(ckpt_path, device, use_fp16=False)  # Keep model in FP32
    metrics_amp = benchmark_inference(
        model_amp, dataloader, device, "amp", batch_size,
        num_warmup=10, log_frequency=10
    )
    stats_amp = print_results("amp", metrics_amp, baseline_stats=stats_fp32,
                             save_path=results_dir / "amp_metrics.json")
    all_results['amp'] = stats_amp

    wandb.log({
        "summary/amp_throughput": stats_amp['throughput']['images_per_second'],
        "summary/amp_latency_ms": stats_amp['inference']['mean_ms'],
        "summary/amp_memory_mb": stats_amp['gpu_memory']['max_allocated_mb'],
        "summary/amp_speedup": stats_amp['throughput']['images_per_second'] / stats_fp32['throughput']['images_per_second'],
    })
    
    del model_amp
    torch.cuda.empty_cache()

    print("\nCreating comparison visualizations...")
    create_comparison_plots(stats_fp32, stats_fp16, stats_amp)
    
    all_results_path = results_dir / "precision_comparison.json"
    with open(all_results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"All results saved to: {all_results_path}\n")

    results_artifact = wandb.Artifact(
        name=f"{experiment_name}-results",
        type="results",
        description="Mixed precision optimization results comparing FP32, FP16, and AMP"
    )
    results_artifact.add_file(str(all_results_path))
    wandb.log_artifact(results_artifact)
    
    print(f"\n{'='*60}")
    print(f"WandB Dashboard: {wandb.run.url}")
    print(f"{'='*60}\n")
    
    wandb.finish()


if __name__ == "__main__":
    main()