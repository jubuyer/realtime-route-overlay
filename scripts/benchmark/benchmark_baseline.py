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
import wandb

# Set root to repo of lane detection
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
UFLD_PATH = PROJECT_ROOT / "models" / "Ultra-Fast-Lane-Detection-v2"

sys.path.insert(0, str(UFLD_PATH))

from model.model_culane import parsingNet
import configs.tusimple_res18 as cfg

# Anchors
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
        
        # Crop and resize (same as inference script)
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


def load_model(ckpt_path: Path, device: torch.device):
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
    print("Model loaded successfully\n")
    return model


def get_image_paths(dataset_dir: Path, max_images: int = None) -> List[Path]:
    """Collect all image paths from dataset directory"""
    image_paths = []
    
    # Look for common image extensions
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
    }
    
    # Get GPU memory info
    if torch.cuda.is_available():
        gpu_info["gpu_memory_total_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
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
        self.preprocessing_times = []
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
        return {
            "data_loading": {
                "mean_ms": np.mean(self.data_loading_times) * 1000,
                "std_ms": np.std(self.data_loading_times) * 1000,
                "min_ms": np.min(self.data_loading_times) * 1000,
                "max_ms": np.max(self.data_loading_times) * 1000,
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
                "inference_fps": 1.0 / np.mean(self.inference_times),
                "total_fps": 1.0 / np.mean(self.total_times),
                "peak_fps": 1.0 / np.min(self.inference_times),
            },
            "gpu_memory": {
                "mean_allocated_mb": np.mean(self.gpu_memory_allocated),
                "max_allocated_mb": np.max(self.gpu_memory_allocated),
                "mean_reserved_mb": np.mean(self.gpu_memory_used),
                "max_reserved_mb": np.max(self.gpu_memory_used),
            }
        }


def benchmark_inference(model, dataloader, device, num_warmup=10, 
                       use_profiler=False, log_frequency=10):
    """Run inference benchmark with WandB logging"""
    metrics = BenchmarkMetrics()
    
    print(f"\n{'='*60}")
    print("BASELINE PERFORMANCE BENCHMARK")
    print(f"{'='*60}\n")
    
    print(f"Device: {device}")
    print(f"Batch size: {dataloader.batch_size}")
    print(f"Number of images: {len(dataloader.dataset)}")
    print(f"Warmup iterations: {num_warmup}\n")
    
    # Warmup
    print("Running warmup...")
    with torch.no_grad():
        for i, (imgs, _, _, _) in enumerate(dataloader):
            if i >= num_warmup:
                break
            imgs = imgs.to(device)
            _ = model(imgs)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    print("Warmup complete. Starting benchmark...\n")
    
    # Benchmark
    total_images = 0
    batch_count = 0
    
    profiler_context = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) if use_profiler else None
    
    if profiler_context:
        profiler_context.__enter__()
    
    # Track real-time metrics for WandB
    step = 0
    
    with torch.no_grad():
        for batch_idx, (imgs, ori_ws, ori_hs, paths) in enumerate(dataloader):
            # Measure data loading time (transfer to GPU)
            data_load_start = time.perf_counter()
            imgs = imgs.to(device)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            data_load_time = time.perf_counter() - data_load_start
            
            # Measure inference time
            inference_start = time.perf_counter()
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
            
            batch_size = imgs.shape[0]
            metrics.add_batch_metrics(
                data_load_time, inference_time, 
                gpu_mem_reserved, gpu_mem_allocated, batch_size
            )
            
            total_images += batch_size
            batch_count += 1
            step += 1
            
            # Log to WandB at specified frequency
            if batch_count % log_frequency == 0:
                wandb.log({
                    "batch/inference_time_ms": inference_time * 1000,
                    "batch/data_loading_time_ms": data_load_time * 1000,
                    "batch/total_time_ms": (data_load_time + inference_time) * 1000,
                    "batch/fps": 1.0 / inference_time if inference_time > 0 else 0,
                    "batch/gpu_memory_allocated_mb": gpu_mem_allocated,
                    "batch/gpu_memory_reserved_mb": gpu_mem_reserved,
                    "progress/images_processed": total_images,
                    "progress/batch_number": batch_count,
                }, step=step)
                
                print(f"Processed {total_images} images... "
                      f"Inference: {inference_time*1000:.2f}ms, "
                      f"FPS: {1.0/inference_time:.1f}")
            
            if profiler_context and batch_count == 5:
                profiler_context.step()
    
    if profiler_context:
        profiler_context.__exit__(None, None, None)
    
    print(f"\nCompleted benchmark on {total_images} images\n")
    
    return metrics, profiler_context


def create_wandb_plots(metrics: BenchmarkMetrics):
    """Create custom visualizations for WandB"""
    
    # Create distribution plots
    inference_times_ms = np.array(metrics.inference_times) * 1000
    data_loading_times_ms = np.array(metrics.data_loading_times) * 1000
    
    # Log histograms
    wandb.log({
        "distributions/inference_time_histogram": wandb.Histogram(inference_times_ms),
        "distributions/data_loading_histogram": wandb.Histogram(data_loading_times_ms),
    })
    
    # Create a table with per-batch metrics
    batch_table = wandb.Table(
        columns=["batch_idx", "inference_ms", "data_loading_ms", "fps", "gpu_memory_mb"],
        data=[
            [i, inf*1000, load*1000, 1.0/inf, mem] 
            for i, (inf, load, mem) in enumerate(
                zip(metrics.inference_times, 
                    metrics.data_loading_times, 
                    metrics.gpu_memory_allocated)
            )
        ][:100]  # Log first 100 batches to avoid too large tables
    )
    wandb.log({"tables/batch_metrics": batch_table})
    
    # Create time series data for line plots
    batch_indices = list(range(len(metrics.inference_times)))
    
    # Log as line plot data
    for i, (inf_time, data_time) in enumerate(zip(metrics.inference_times, 
                                                    metrics.data_loading_times)):
        if i % 5 == 0:  # Sample every 5th point
            wandb.log({
                "timeseries/inference_time_ms": inf_time * 1000,
                "timeseries/data_loading_time_ms": data_time * 1000,
            })


def print_results(metrics: BenchmarkMetrics, save_path: Path = None):
    """Print and save benchmark results"""
    stats = metrics.compute_statistics()
    
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}\n")
    
    print("DATA LOADING:")
    print(f"  Mean:   {stats['data_loading']['mean_ms']:.2f} ms")
    print(f"  Median: {stats['data_loading']['median_ms']:.2f} ms")
    print(f"  Std:    {stats['data_loading']['std_ms']:.2f} ms")
    print(f"  Min:    {stats['data_loading']['min_ms']:.2f} ms")
    print(f"  Max:    {stats['data_loading']['max_ms']:.2f} ms\n")
    
    print("INFERENCE:")
    print(f"  Mean:   {stats['inference']['mean_ms']:.2f} ms")
    print(f"  Median: {stats['inference']['median_ms']:.2f} ms")
    print(f"  Std:    {stats['inference']['std_ms']:.2f} ms")
    print(f"  Min:    {stats['inference']['min_ms']:.2f} ms")
    print(f"  Max:    {stats['inference']['max_ms']:.2f} ms")
    print(f"  P95:    {stats['inference']['p95_ms']:.2f} ms")
    print(f"  P99:    {stats['inference']['p99_ms']:.2f} ms\n")
    
    print("TOTAL PIPELINE:")
    print(f"  Mean:   {stats['total_pipeline']['mean_ms']:.2f} ms")
    print(f"  Median: {stats['total_pipeline']['median_ms']:.2f} ms")
    print(f"  Std:    {stats['total_pipeline']['std_ms']:.2f} ms\n")
    
    print("THROUGHPUT:")
    print(f"  Inference Only (Mean): {stats['throughput']['inference_fps']:.2f} FPS")
    print(f"  Inference Only (Peak): {stats['throughput']['peak_fps']:.2f} FPS")
    print(f"  Total Pipeline:        {stats['throughput']['total_fps']:.2f} FPS\n")
    
    print("GPU MEMORY:")
    print(f"  Mean Allocated: {stats['gpu_memory']['mean_allocated_mb']:.2f} MB")
    print(f"  Peak Allocated: {stats['gpu_memory']['max_allocated_mb']:.2f} MB")
    print(f"  Mean Reserved:  {stats['gpu_memory']['mean_reserved_mb']:.2f} MB")
    print(f"  Peak Reserved:  {stats['gpu_memory']['max_reserved_mb']:.2f} MB\n")
    
    print(f"{'='*60}\n")
    
    # Save results
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Results saved to: {save_path}\n")
    
    return stats


def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1  # Start with batch_size=1 for baseline
    num_workers = 4
    max_images = 500  # Adjust based on your dataset size
    experiment_name = "ufldv2-baseline"
    
    # Paths
    ckpt_path = UFLD_PATH / "weights" / "tusimple_res18.pth"
    dataset_dir = PROJECT_ROOT / "datasets" / "TUSimple" / "test_set"
    results_dir = PROJECT_ROOT / "results" / "baseline"
    
    # Get system info
    gpu_info = get_gpu_info()
    system_info = get_system_info()
    
    # Load model
    model = load_model(ckpt_path, device)
    
    # Get model info
    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size_mb = param_count * 4 / 1024**2
    
    model_info = {
        "model_name": "UFLDv2",
        "backbone": cfg.backbone,
        "total_parameters": param_count,
        "trainable_parameters": trainable_params,
        "model_size_mb": param_size_mb,
        "input_height": cfg.train_height,
        "input_width": cfg.train_width,
        "num_lanes": cfg.num_lanes,
    }
    
    print(f"Model parameters: {param_count:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {param_size_mb:.2f} MB\n")
    
    # Initialize WandB
    wandb.init(
        project="ufldv2-optimization",
        name=experiment_name,
        config={
            # Model configuration
            **model_info,
            
            # Training/Inference configuration
            "batch_size": batch_size,
            "num_workers": num_workers,
            "max_images": max_images,
            "device": str(device),
            "precision": "fp32",
            "optimization_technique": "baseline",
            
            # System information
            **system_info,
            **gpu_info,
            
            # Dataset info
            "dataset": "TuSimple",
            "dataset_split": "test",
        },
        tags=["baseline", "inference", "lane-detection"],
        notes="Baseline performance benchmark without any optimizations"
    )
    
    # Log model architecture summary
    wandb.config.update({"model_architecture": str(model)[:1000]})  # First 1000 chars
    
    # Prepare dataset
    image_paths = get_image_paths(dataset_dir, max_images)
    dataset = LaneDataset(image_paths)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Run benchmark
    metrics, profiler = benchmark_inference(
        model, dataloader, device, 
        num_warmup=10,
        use_profiler=True,
        log_frequency=10
    )
    
    # Compute final statistics
    stats = metrics.compute_statistics()
    
    # Log summary metrics to WandB
    wandb.log({
        # Data loading metrics
        "summary/data_loading_mean_ms": stats['data_loading']['mean_ms'],
        "summary/data_loading_std_ms": stats['data_loading']['std_ms'],
        "summary/data_loading_median_ms": stats['data_loading']['median_ms'],
        
        # Inference metrics
        "summary/inference_mean_ms": stats['inference']['mean_ms'],
        "summary/inference_median_ms": stats['inference']['median_ms'],
        "summary/inference_std_ms": stats['inference']['std_ms'],
        "summary/inference_min_ms": stats['inference']['min_ms'],
        "summary/inference_max_ms": stats['inference']['max_ms'],
        "summary/inference_p95_ms": stats['inference']['p95_ms'],
        "summary/inference_p99_ms": stats['inference']['p99_ms'],
        
        # Pipeline metrics
        "summary/total_pipeline_mean_ms": stats['total_pipeline']['mean_ms'],
        "summary/total_pipeline_median_ms": stats['total_pipeline']['median_ms'],
        
        # Throughput metrics
        "summary/inference_fps": stats['throughput']['inference_fps'],
        "summary/peak_fps": stats['throughput']['peak_fps'],
        "summary/total_fps": stats['throughput']['total_fps'],
        
        # Memory metrics
        "summary/gpu_memory_allocated_mb": stats['gpu_memory']['max_allocated_mb'],
        "summary/gpu_memory_reserved_mb": stats['gpu_memory']['max_reserved_mb'],
        
        # Efficiency metrics
        "summary/images_per_second": stats['throughput']['inference_fps'],
        "summary/ms_per_image": stats['inference']['mean_ms'],
    })
    
    # Create visualizations
    print("Creating visualizations for WandB...")
    create_wandb_plots(metrics)
    
    # Print and save results
    results_path = results_dir / "baseline_metrics.json"
    print_results(metrics, results_path)
    
    # Save profiler results
    if profiler:
        profiler_path = results_dir / "profiler_trace.json"
        profiler.export_chrome_trace(str(profiler_path))
        print(f"Profiler trace saved to: {profiler_path}")
        print("View it at: chrome://tracing\n")
        
        # Print profiler summary
        print("Top 10 operations by CPU time:")
        profiler_summary = profiler.key_averages().table(
            sort_by="cpu_time_total", row_limit=10
        )
        print(profiler_summary)
        
        # Log profiler trace as artifact
        artifact = wandb.Artifact(
            name=f"{experiment_name}-profiler-trace",
            type="profiler_trace",
            description="PyTorch profiler trace for baseline inference"
        )
        artifact.add_file(str(profiler_path))
        wandb.log_artifact(artifact)
    
    # Save results JSON as artifact
    results_artifact = wandb.Artifact(
        name=f"{experiment_name}-metrics",
        type="results",
        description="Detailed benchmark metrics"
    )
    results_artifact.add_file(str(results_path))
    wandb.log_artifact(results_artifact)
    
    # Create summary table for comparison
    summary_table = wandb.Table(
        columns=["Metric", "Value", "Unit"],
        data=[
            ["Mean Inference Time", f"{stats['inference']['mean_ms']:.2f}", "ms"],
            ["Median Inference Time", f"{stats['inference']['median_ms']:.2f}", "ms"],
            ["P95 Inference Time", f"{stats['inference']['p95_ms']:.2f}", "ms"],
            ["P99 Inference Time", f"{stats['inference']['p99_ms']:.2f}", "ms"],
            ["Inference FPS", f"{stats['throughput']['inference_fps']:.2f}", "FPS"],
            ["Peak FPS", f"{stats['throughput']['peak_fps']:.2f}", "FPS"],
            ["GPU Memory (Peak)", f"{stats['gpu_memory']['max_allocated_mb']:.2f}", "MB"],
            ["Model Size", f"{param_size_mb:.2f}", "MB"],
            ["Total Parameters", f"{param_count:,}", "count"],
        ]
    )
    wandb.log({"summary/metrics_table": summary_table})
    
    print(f"\n{'='*60}")
    print(f"WandB Dashboard: {wandb.run.get_url()}")
    print(f"{'='*60}\n")
 
    wandb.finish()


if __name__ == "__main__":
    main()