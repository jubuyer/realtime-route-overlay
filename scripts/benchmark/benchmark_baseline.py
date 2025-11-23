"""
Baseline Benchmark Script for UFLDv2 Lane Detection
Performs detailed profiling of model inference performance
"""

import torch
import torch.nn as nn
import time
import json
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import sys

from model.model import parsingNet
import cv2
from utils.common import merge_config
from data.dataset import LaneTestDataset
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
import wandb

sys.path.append(str(Path(__file__).parent.parent / 'models' / 'Ultra-Fast-Lane-Detection-v2'))

class BenchmarkProfiler:
    def __init__(self, config, use_wandb=True):
        self.config = config
        self.use_wandb = use_wandb
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.metrics = {
            'data_loading_times': [],
            'preprocessing_times': [],
            'inference_times': [],
            'postprocessing_times': [],
            'total_times': [],
            'gpu_memory_allocated': [],
            'gpu_memory_reserved': [],
        }
        
    def setup_wandb(self, project_name="route-overlay-optimization", run_name="baseline"):
        """Initialize Weights & Biases logging"""
        if self.use_wandb:
            wandb.init(
                project=project_name,
                name=run_name,
                config={
                    "model": self.config['model'],
                    "backbone": self.config['backbone'],
                    "griding_num": self.config['griding_num'],
                    "num_lanes": self.config['num_lanes'],
                    "device": str(self.device),
                    "batch_size": self.config.get('batch_size', 1),
                }
            )
            print("WandB initialized")
    
    def load_model(self, weight_path):
        """Load pretrained UFLDv2 model"""
        print(f"Loading model from {weight_path}...")
        
        # Initialize model
        net = parsingNet(
            pretrained=False,
            backbone=self.config['backbone'],
            cls_dim=(self.config['griding_num'] + 1, 
                     self.config['cls_num_per_lane'], 
                     self.config['num_lanes']),
            use_aux=False
        ).to(self.device)
        
        # weights
        state_dict = torch.load(weight_path, map_location=self.device)
        compatible_state_dict = {}
        for k, v in state_dict.items():
            if 'module.' in k:
                compatible_state_dict[k[7:]] = v
            else:
                compatible_state_dict[k] = v
        
        net.load_state_dict(compatible_state_dict, strict=False)
        net.eval()
        
        print("Model loaded successfully")
        return net
    
    def prepare_dataloader(self, data_root, test_json):
        """Prepare test dataloader"""
        print("Preparing dataloader...")
        
        dataset = LaneTestDataset(
            data_root,
            test_json,
            img_height=self.config['img_height'],
            img_width=self.config['img_width']
        )
        
        loader = DataLoader(
            dataset,
            batch_size=self.config.get('batch_size', 1),
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        print(f"Dataloader ready with {len(dataset)} samples")
        return loader
    
    def warmup(self, model, num_iterations=10):
        """Warmup GPU for accurate benchmarking"""
        print(f"Warming up GPU ({num_iterations} iterations)...")
        
        dummy_input = torch.randn(
            1, 3, 
            self.config['img_height'], 
            self.config['img_width']
        ).to(self.device)
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(dummy_input)
        
        torch.cuda.synchronize()
        print("Warmup complete")
    
    def benchmark_inference(self, model, dataloader, num_batches=100):
        """Benchmark of inference pipeline"""
        print(f"\nRunning benchmark on {num_batches} batches...")
        
        model.eval()
        batch_count = 0
        
        with torch.no_grad():
            pbar = tqdm(dataloader, total=min(num_batches, len(dataloader)))
            
            for batch_idx, data in enumerate(pbar):
                if batch_count >= num_batches:
                    break
                
                # Time: Data Loading (measured by iterator)
                data_load_start = time.time()
                
                # Time: Preprocessing (already done in dataset, measure transfer)
                preprocess_start = time.time()
                imgs = data['img'].to(self.device, non_blocking=True)
                torch.cuda.synchronize()
                preprocess_time = time.time() - preprocess_start
                
                # Time: Inference
                inference_start = time.time()
                torch.cuda.synchronize()
                
                outputs = model(imgs)
                
                torch.cuda.synchronize()
                inference_time = time.time() - inference_start
                
                # Time: Postprocessing (minimal for this model)
                postprocess_start = time.time()
                # Simulate basic postprocessing
                _ = outputs.cpu()
                postprocess_time = time.time() - postprocess_start
                
                # Total time
                total_time = preprocess_time + inference_time + postprocess_time
                
                # Memory metrics
                memory_allocated = torch.cuda.memory_allocated() / 1e6  # MB
                memory_reserved = torch.cuda.memory_reserved() / 1e6  # MB
                
                self.metrics['preprocessing_times'].append(preprocess_time * 1000)
                self.metrics['inference_times'].append(inference_time * 1000)
                self.metrics['postprocessing_times'].append(postprocess_time * 1000)
                self.metrics['total_times'].append(total_time * 1000)
                self.metrics['gpu_memory_allocated'].append(memory_allocated)
                self.metrics['gpu_memory_reserved'].append(memory_reserved)

                if self.use_wandb and batch_idx % 10 == 0:
                    wandb.log({
                        'batch_idx': batch_idx,
                        'preprocess_time_ms': preprocess_time * 1000,
                        'inference_time_ms': inference_time * 1000,
                        'postprocess_time_ms': postprocess_time * 1000,
                        'total_time_ms': total_time * 1000,
                        'fps': 1.0 / total_time,
                        'gpu_memory_allocated_mb': memory_allocated,
                        'gpu_memory_reserved_mb': memory_reserved,
                    })
                
                # Update progress bar
                pbar.set_postfix({
                    'FPS': f"{1.0/total_time:.1f}",
                    'Latency': f"{total_time*1000:.1f}ms",
                    'GPU_Mem': f"{memory_allocated:.0f}MB"
                })
                
                batch_count += 1
        
        print("Benchmark complete")
    
    def profile_with_pytorch_profiler(self, model, dataloader, num_batches=10):
        """Profiling using PyTorch Profiler"""
        print("\nRunning PyTorch Profiler...")
        
        model.eval()
        
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            with torch.no_grad():
                for batch_idx, data in enumerate(dataloader):
                    if batch_idx >= num_batches:
                        break
                    
                    imgs = data['img'].to(self.device)
                    
                    with record_function("model_inference"):
                        outputs = model(imgs)

        prof_output_dir = Path('results/profiling')
        prof_output_dir.mkdir(parents=True, exist_ok=True)

        trace_path = prof_output_dir / 'trace.json'
        prof.export_chrome_trace(str(trace_path))
        print(f"Profiler trace saved to {trace_path}")

        print("\nTop operations by CUDA time:")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

        if self.use_wandb:
            wandb.save(str(trace_path))
    
    def compute_statistics(self):
        """Compute and display statistics"""
        print("\n" + "="*60)
        print("BENCHMARK RESULTS")
        print("="*60)
        
        stats = {}
        
        for metric_name, values in self.metrics.items():
            if len(values) > 0:
                stats[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'p95': np.percentile(values, 95),
                    'p99': np.percentile(values, 99),
                }

        print(f"\nInference Performance:")
        print(f"  Mean Latency: {stats['total_times']['mean']:.2f} ms")
        print(f"  Std Latency:  {stats['total_times']['std']:.2f} ms")
        print(f"  P95 Latency:  {stats['total_times']['p95']:.2f} ms")
        print(f"  P99 Latency:  {stats['total_times']['p99']:.2f} ms")
        print(f"  Mean FPS:     {1000/stats['total_times']['mean']:.1f}")
        print(f"  Min FPS:      {1000/stats['total_times']['max']:.1f}")
        
        print(f"\nTime Breakdown:")
        print(f"  Preprocessing:  {stats['preprocessing_times']['mean']:.2f} ms ({stats['preprocessing_times']['mean']/stats['total_times']['mean']*100:.1f}%)")
        print(f"  Inference:      {stats['inference_times']['mean']:.2f} ms ({stats['inference_times']['mean']/stats['total_times']['mean']*100:.1f}%)")
        print(f"  Postprocessing: {stats['postprocessing_times']['mean']:.2f} ms ({stats['postprocessing_times']['mean']/stats['total_times']['mean']*100:.1f}%)")
        
        print(f"\nGPU Memory:")
        print(f"  Mean Allocated: {stats['gpu_memory_allocated']['mean']:.1f} MB")
        print(f"  Peak Allocated: {stats['gpu_memory_allocated']['max']:.1f} MB")
        print(f"  Mean Reserved:  {stats['gpu_memory_reserved']['mean']:.1f} MB")
        
        print("="*60)

        if self.use_wandb:
            wandb.log({
                'summary/mean_latency_ms': stats['total_times']['mean'],
                'summary/p95_latency_ms': stats['total_times']['p95'],
                'summary/p99_latency_ms': stats['total_times']['p99'],
                'summary/mean_fps': 1000/stats['total_times']['mean'],
                'summary/inference_time_ms': stats['inference_times']['mean'],
                'summary/peak_memory_mb': stats['gpu_memory_allocated']['max'],
            })

        results_dir = Path('results/benchmarks')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        with open(results_dir / 'baseline_stats.json', 'w') as f:
            json.dump(stats, f, indent=2, default=float)
        
        print(f"\nResults saved to {results_dir / 'baseline_stats.json'}")
        
        return stats


def main():
    parser = argparse.ArgumentParser(description='Benchmark UFLDv2 Model')
    parser.add_argument('--config', type=str, 
                        default='models/Ultra-Fast-Lane-Detection-v2/configs/tusimple_res18.py',
                        help='Path to model config file')
    parser.add_argument('--weight', type=str,
                        default='models/Ultra-Fast-Lane-Detection-v2/weights/tusimple_res18.pth',
                        help='Path to model weights')
    parser.add_argument('--data_root', type=str,
                        default='datasets/TUSimple',
                        help='Path to dataset root')
    parser.add_argument('--test_json', type=str,
                        default='datasets/TUSimple/test_label.json',
                        help='Path to test label JSON')
    parser.add_argument('--num_batches', type=int, default=100,
                        help='Number of batches to benchmark')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable WandB logging')
    parser.add_argument('--run_name', type=str, default='baseline-res18',
                        help='WandB run name')
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return
    
    config = merge_config(args.config)
    config['batch_size'] = args.batch_size
    
    profiler = BenchmarkProfiler(config, use_wandb=not args.no_wandb)

    if not args.no_wandb:
        profiler.setup_wandb(run_name=args.run_name)

    model = profiler.load_model(args.weight)
    dataloader = profiler.prepare_dataloader(args.data_root, args.test_json)
    profiler.warmup(model)
    profiler.benchmark_inference(model, dataloader, num_batches=args.num_batches)
    profiler.profile_with_pytorch_profiler(model, dataloader, num_batches=10)
    stats = profiler.compute_statistics()

    if not args.no_wandb:
        wandb.finish()
    
    print("\n Benchmark complete")


if __name__ == '__main__':
    main()