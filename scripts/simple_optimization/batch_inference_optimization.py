"""
Optimization Technique: Batch Inference Optimization
Tests different batch sizes to maximize throughput and GPU utilization
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
        self.preprocessing_times = []
        self.inference_times = []
        self.total_times = []
        self.gpu_memory_used = []
        self.gpu_memory_allocated = []
        self.batch_sizes = []
        self.gpu_utilization = []
    
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
                "images_per_second": len(self.inference_times) * np.mean(self.batch_sizes) / np.sum(self.inference_times),
                "batches_per_second": len(self.inference_times) / np.sum(self.inference_times),
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


def benchmark_batch_size(model, dataloader, device, batch_size, num_warmup=10, log_frequency=10):
    """Run inference benchmark for a specific batch size"""
    metrics = BenchmarkMetrics()
    
    print(f"\n{'='*60}")
    print(f"BATCH SIZE: {batch_size}")
    print(f"{'='*60}\n")

    print(f"Running warmup ({num_warmup} iterations)...")
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
    
    # benchmark
    total_images = 0
    batch_count = 0
    
    with torch.no_grad():
        for batch_idx, (imgs, ori_ws, ori_hs, paths) in enumerate(dataloader):
            # data loading time (transfer to GPU)
            data_load_start = time.perf_counter()
            imgs = imgs.to(device)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            data_load_time = time.perf_counter() - data_load_start
            
            # inference time
            inference_start = time.perf_counter()
            pred = model(imgs)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            inference_time = time.perf_counter() - inference_start

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
            
            # log to WandB at specified frequency
            if batch_count % log_frequency == 0:
                images_per_sec = current_batch_size / inference_time if inference_time > 0 else 0
                wandb.log({
                    f"bs{batch_size}/batch_inference_time_ms": inference_time * 1000,
                    f"bs{batch_size}/batch_data_loading_time_ms": data_load_time * 1000,
                    f"bs{batch_size}/batch_total_time_ms": (data_load_time + inference_time) * 1000,
                    f"bs{batch_size}/batch_images_per_second": images_per_sec,
                    f"bs{batch_size}/batch_gpu_memory_allocated_mb": gpu_mem_allocated,
                    f"bs{batch_size}/batch_gpu_memory_reserved_mb": gpu_mem_reserved,
                    f"bs{batch_size}/batch_latency_per_image_ms": (inference_time * 1000) / current_batch_size,
                    "progress/total_images_processed": total_images,
                    "progress/batch_number": batch_count,
                })
                
                print(f"[BS={batch_size}] Processed {total_images} images... "
                      f"Inference: {inference_time*1000:.2f}ms, "
                      f"Images/sec: {images_per_sec:.1f}, "
                      f"Latency/img: {(inference_time*1000)/current_batch_size:.2f}ms")
    
    print(f"\n[BS={batch_size}] Completed benchmark on {total_images} images\n")
    
    return metrics


def print_batch_results(batch_size: int, metrics: BenchmarkMetrics, baseline_stats: Dict = None):
    """Print results for a specific batch size with comparison to baseline"""
    stats = metrics.compute_statistics()
    
    print(f"\n{'='*60}")
    print(f"BATCH SIZE {batch_size} RESULTS")
    print(f"{'='*60}\n")
    
    print("INFERENCE:")
    print(f"  Mean:   {stats['inference']['mean_ms']:.2f} ms")
    print(f"  Median: {stats['inference']['median_ms']:.2f} ms")
    print(f"  Std:    {stats['inference']['std_ms']:.2f} ms")
    print(f"  P95:    {stats['inference']['p95_ms']:.2f} ms")
    print(f"  P99:    {stats['inference']['p99_ms']:.2f} ms\n")
    
    print("THROUGHPUT:")
    print(f"  Images/second: {stats['throughput']['images_per_second']:.2f}")
    print(f"  Batches/second: {stats['throughput']['batches_per_second']:.2f}")
    print(f"  Latency per image: {stats['inference']['mean_ms']/batch_size:.2f} ms\n")
    
    print("GPU MEMORY:")
    print(f"  Peak Allocated: {stats['gpu_memory']['max_allocated_mb']:.2f} MB")
    print(f"  Peak Reserved:  {stats['gpu_memory']['max_reserved_mb']:.2f} MB\n")
    
    # comparison with baseline
    if baseline_stats:
        baseline_fps = baseline_stats['throughput']['images_per_second']
        current_fps = stats['throughput']['images_per_second']
        speedup = current_fps / baseline_fps
        
        baseline_latency = baseline_stats['inference']['mean_ms']
        current_latency = stats['inference']['mean_ms'] / batch_size
        latency_improvement = baseline_latency / current_latency
        
        print("COMPARISON TO BASELINE (Batch Size 1):")
        print(f"  Throughput Speedup: {speedup:.2f}x ({baseline_fps:.1f} → {current_fps:.1f} images/s)")
        print(f"  Latency per Image Improvement: {latency_improvement:.2f}x ({baseline_latency:.2f}ms → {current_latency:.2f}ms)")
        print(f"  Memory Overhead: {stats['gpu_memory']['max_allocated_mb'] - baseline_stats['gpu_memory']['max_allocated_mb']:.2f} MB\n")
    
    print(f"{'='*60}\n")
    
    return stats


def create_comparison_plots(all_results: Dict):
    """Create comparison plots across all batch sizes"""
    
    batch_sizes = sorted(all_results.keys())
    
    # prepare data for plots
    throughputs = [all_results[bs]['throughput']['images_per_second'] for bs in batch_sizes]
    latencies = [all_results[bs]['inference']['mean_ms'] / bs for bs in batch_sizes]  # per-image latency
    memories = [all_results[bs]['gpu_memory']['max_allocated_mb'] for bs in batch_sizes]
    p95_latencies = [all_results[bs]['inference']['p95_ms'] / bs for bs in batch_sizes]
    
    # create comparison table
    comparison_table = wandb.Table(
        columns=["Batch Size", "Throughput (img/s)", "Latency per Image (ms)", 
                 "P95 Latency (ms)", "Peak Memory (MB)", "Speedup vs BS=1"],
        data=[
            [
                bs, 
                f"{all_results[bs]['throughput']['images_per_second']:.2f}",
                f"{all_results[bs]['inference']['mean_ms'] / bs:.2f}",
                f"{all_results[bs]['inference']['p95_ms'] / bs:.2f}",
                f"{all_results[bs]['gpu_memory']['max_allocated_mb']:.2f}",
                f"{all_results[bs]['throughput']['images_per_second'] / all_results[1]['throughput']['images_per_second']:.2f}x"
            ]
            for bs in batch_sizes
        ]
    )
    wandb.log({"comparison/batch_size_comparison_table": comparison_table})
    
    # log individual metrics for plotting
    for bs in batch_sizes:
        wandb.log({
            "comparison/throughput_vs_batch_size": throughputs[batch_sizes.index(bs)],
            "comparison/latency_vs_batch_size": latencies[batch_sizes.index(bs)],
            "comparison/memory_vs_batch_size": memories[batch_sizes.index(bs)],
            "comparison/batch_size": bs,
        })
    
    # find optimal batch size (best throughput)
    optimal_bs = batch_sizes[np.argmax(throughputs)]
    optimal_throughput = max(throughputs)
    
    wandb.log({
        "comparison/optimal_batch_size": optimal_bs,
        "comparison/optimal_throughput": optimal_throughput,
        "comparison/max_speedup": optimal_throughput / throughputs[0],
    })
    
    print(f"\n{'='*60}")
    print("OPTIMAL BATCH SIZE ANALYSIS")
    print(f"{'='*60}\n")
    print(f"Optimal Batch Size: {optimal_bs}")
    print(f"Maximum Throughput: {optimal_throughput:.2f} images/second")
    print(f"Speedup over BS=1: {optimal_throughput / throughputs[0]:.2f}x")
    print(f"Latency per Image: {all_results[optimal_bs]['inference']['mean_ms'] / optimal_bs:.2f} ms")
    print(f"Memory Usage: {all_results[optimal_bs]['gpu_memory']['max_allocated_mb']:.2f} MB")
    print(f"\n{'='*60}\n")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_sizes_to_test = [1, 4, 8, 16, 32]
    num_workers = 4
    max_images = 500
    experiment_name = "ufldv2-batch-optimization-NVidia-GPU-4070"

    ckpt_path = UFLD_PATH / "weights" / "tusimple_res18.pth"
    dataset_dir = PROJECT_ROOT / "datasets" / "TUSimple" / "test_set"
    results_dir = PROJECT_ROOT / "results" / "batch_optimization"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    gpu_info = get_gpu_info()
    system_info = get_system_info()
    model = load_model(ckpt_path, device)
    
    # get model info
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
    print(f"Model size: {param_size_mb:.2f} MB\n")
    
    # initialize WandB
    wandb.init(
        project="ufldv2-optimization",
        name=experiment_name,
        group = 'simple-optimizations',
        config={
            **model_info,
            "batch_sizes_tested": batch_sizes_to_test,
            "num_workers": num_workers,
            "max_images": max_images,
            "device": str(device),
            "precision": "fp32",
            "optimization_technique": "batch_inference",
            **system_info,
            **gpu_info,
            "dataset": "TuSimple",
            "dataset_split": "test",
        },
        tags=["optimization", "batch-inference", "lane-detection"],
        notes="Optimization Technique 1: Batch Inference - Testing different batch sizes for optimal throughput"
    )
    
    # prepare dataset
    image_paths = get_image_paths(dataset_dir, max_images)
    
    # store results for all batch sizes
    all_results = {}
    baseline_stats = None
    
    # test each batch size
    for batch_size in batch_sizes_to_test:
        print(f"\n\n{'#'*60}")
        print(f"# TESTING BATCH SIZE: {batch_size}")
        print(f"{'#'*60}\n")
        
        # create dataset and dataloader for this batch size
        dataset = LaneDataset(image_paths)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if device.type == 'cuda' else False,
            drop_last=False
        )
        
        metrics = benchmark_batch_size(
            model, dataloader, device, batch_size,
            num_warmup=10,
            log_frequency=10
        )
        
        stats = print_batch_results(batch_size, metrics, baseline_stats)
        all_results[batch_size] = stats
        
        # save baseline
        if batch_size == 1:
            baseline_stats = stats
        
        wandb.log({
            f"summary/bs{batch_size}_throughput_imgs_per_sec": stats['throughput']['images_per_second'],
            f"summary/bs{batch_size}_latency_per_image_ms": stats['inference']['mean_ms'] / batch_size,
            f"summary/bs{batch_size}_inference_time_ms": stats['inference']['mean_ms'],
            f"summary/bs{batch_size}_p95_latency_ms": stats['inference']['p95_ms'] / batch_size,
            f"summary/bs{batch_size}_memory_mb": stats['gpu_memory']['max_allocated_mb'],
        })
        
        # save individual results
        result_path = results_dir / f"batch_size_{batch_size}_metrics.json"
        with open(result_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Results saved to: {result_path}\n")
    
    print("\nCreating comparison visualizations...")
    create_comparison_plots(all_results)

    all_results_path = results_dir / "all_batch_sizes_comparison.json"
    with open(all_results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"All results saved to: {all_results_path}\n")
    
    # final summary artifact
    results_artifact = wandb.Artifact(
        name=f"{experiment_name}-results",
        type="results",
        description="Batch inference optimization results across all batch sizes"
    )
    results_artifact.add_file(str(all_results_path))
    wandb.log_artifact(results_artifact)
    
    print(f"\n{'='*60}")
    print(f"WandB Dashboard: {wandb.run.url}")
    print(f"{'='*60}\n")
    
    wandb.finish()


if __name__ == "__main__":
    main()