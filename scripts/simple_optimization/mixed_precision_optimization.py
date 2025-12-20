"""
Optimization Technique 2: Mixed Precision Inference (FP16)
Uses automatic mixed precision to reduce memory usage and improve inference speed
while measuring impact on lane detection accuracy
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
    
    def __init__(self, image_paths: List[Path], labels_dict: Dict = None, transform=None):
        self.image_paths = image_paths
        self.labels_dict = labels_dict or {}
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(str(img_path))
        
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {img_path}")
        
        ori_h, ori_w = img.shape[:2]
        
        # get label if available - match TuSimple format
        try:
            # the label set has clips in path
            path_parts = img_path.parts
            if 'clips' in path_parts:
                clips_idx = path_parts.index('clips')
                relative_path = '/'.join(path_parts[clips_idx:])
            else:
                # fallback: get relative path from test_set
                relative_path = img_path.relative_to(img_path.parents[2])
                relative_path = str(relative_path).replace('\\', '/')
        except:
            relative_path = img_path.name
        
        label = self.labels_dict.get(relative_path, None)
        
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
        
        return torch.from_numpy(img_chw).float(), ori_w, ori_h, str(img_path), label


def custom_collate(batch):
    """Custom collate function to handle None labels"""
    imgs = torch.stack([item[0] for item in batch])
    ori_ws = torch.tensor([item[1] for item in batch])
    ori_hs = torch.tensor([item[2] for item in batch])
    paths = [item[3] for item in batch]
    labels = [item[4] for item in batch]
    
    return imgs, ori_ws, ori_hs, paths, labels


def load_tusimple_labels(label_path: Path) -> Dict:
    """Load TuSimple test labels"""
    print(f"Loading labels from: {label_path}")
    labels_dict = {}
    
    with open(label_path, 'r') as f:
        for line in f:
            label = json.loads(line.strip())
            raw_file = label['raw_file']
            labels_dict[raw_file] = label
    
    print(f"Loaded {len(labels_dict)} labels\n")
    return labels_dict


def compute_tusimple_accuracy(predictions: List[Dict], labels_dict: Dict) -> Dict:
    """
    Compute TuSimple accuracy metrics
    predictions: list of dicts with 'raw_file' and 'lanes' (list of lane coordinates)
    labels_dict: ground truth labels
    """
    total_pred_lanes = sum(len(p['lanes']) for p in predictions)
    
    if total_pred_lanes == 0:
        print("Warning: No valid predictions found. Returning zero accuracy.")
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'total_gt_lanes': 0,
            'correct_predictions': 0,
            'false_positives': 0,
            'false_negatives': 0,
        }
    
    print(f"\nDEBUG - Path matching:")
    print(f"Sample prediction paths (first 3):")
    for i, pred in enumerate(predictions[:3]):
        print(f"  {i}: {pred['raw_file']}")
    print(f"\nSample label paths (first 3):")
    for i, key in enumerate(list(labels_dict.keys())[:3]):
        print(f"  {i}: {key}")
    
    total = 0
    correct = 0
    fp = 0  # False positives
    fn = 0  # False negatives
    
    matched_files = 0
    
    for pred in predictions:
        raw_file = pred['raw_file']
        if raw_file not in labels_dict:
            continue
        
        matched_files += 1
        gt = labels_dict[raw_file]
        pred_lanes = pred['lanes']
        gt_lanes = gt['lanes']
        
        # IoU-based matching
        matched_gt = set()
        
        for pred_lane in pred_lanes:
            pred_lane_valid = [p for p in pred_lane if p >= 0]  # rm invalid points
            if len(pred_lane_valid) == 0:
                fp += 1
                continue
                
            best_iou = 0
            best_match = -1
            
            for gt_idx, gt_lane in enumerate(gt_lanes):
                gt_lane_valid = [p for p in gt_lane if p >= 0]
                if len(gt_lane_valid) == 0:
                    continue
                
                iou = compute_lane_iou(pred_lane, gt_lane)
                if iou > best_iou:
                    best_iou = iou
                    best_match = gt_idx
            
            if best_iou > 0.5 and best_match not in matched_gt:
                correct += 1
                matched_gt.add(best_match)
            else:
                fp += 1

        fn += len(gt_lanes) - len(matched_gt)
        total += len(gt_lanes)
    
    print(f"\nMatched {matched_files}/{len(predictions)} files with ground truth")
    print(f"Total GT lanes: {total}, Correct: {correct}, FP: {fp}, FN: {fn}")
    
    accuracy = correct / total if total > 0 else 0
    precision = correct / (correct + fp) if (correct + fp) > 0 else 0
    recall = correct / (correct + fn) if (correct + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'total_gt_lanes': total,
        'correct_predictions': correct,
        'false_positives': fp,
        'false_negatives': fn,
    }


def compute_lane_iou(lane1: List, lane2: List) -> float:
    """Compute IoU between two lanes (simplified version)"""
    # overlap ratio based on x-coordinates
    
    if len(lane1) == 0 or len(lane2) == 0:
        return 0.0
    
    # count overlapping points within threshold
    threshold = 40
    overlap = 0
    
    for p1 in lane1:
        for p2 in lane2:
            if abs(p1 - p2) < threshold:
                overlap += 1
                break
    
    union = len(lane1) + len(lane2) - overlap
    return overlap / union if union > 0 else 0.0


def decode_predictions(outputs, ori_w, ori_h) -> List[List]:
    """Decode UFLDv2 model outputs to TuSimple lane format
    
    Output shapes:
    - loc_row: (batch, num_grid=100, num_cls_row=56, num_lanes=4)
    - exist_row: (batch, 2, num_cls_row=56, num_lanes=4)
    
    Returns list of lanes for each image, where each lane is a list of x-coordinates
    corresponding to the h_samples heights in TuSimple format
    """
    if not isinstance(outputs, dict):
        batch_size = outputs.shape[0]
        return [[] for _ in range(batch_size)]
    
    loc_row = outputs['loc_row']  # (batch, 100, 56, 4)
    exist_row = outputs['exist_row']  # (batch, 2, 56, 4)
    
    batch_size = loc_row.shape[0]
    num_grid = loc_row.shape[1]  # 100
    num_cls_row = loc_row.shape[2]  # 56 (same as cfg.num_row)
    num_lanes = loc_row.shape[3]  # 4
    
    lanes_batch = []
    
    for b in range(batch_size):
        lanes = []
        
        for lane_idx in range(num_lanes):
            exist_pred = exist_row[b, :, :, lane_idx]
            # avg across row anchors and apply softmax
            exist_mean = exist_pred.mean(dim=1)  # (2,)
            exist_prob = torch.softmax(exist_mean, dim=0)[1]  # prob of existence
            
            if exist_prob < 0.5:  # lane DNE
                continue
            
            lane_points = []
            
            for row_idx in range(num_cls_row):
                row_loc = loc_row[b, :, row_idx, lane_idx]  # (100,)
                row_prob = torch.softmax(row_loc, dim=0)
                max_prob, max_grid_idx = torch.max(row_prob, dim=0)
                
                # threshold for valid detection
                if max_prob < 0.3:
                    lane_points.append(-2)
                else:
                    # convert grid index to x-coordinate
                    x_normalized = max_grid_idx.item() / (num_grid - 1)
                    x = int(x_normalized * ori_w)
                    
                    x = max(0, min(x, ori_w - 1))
                    lane_points.append(x)
            
            # add lane if it has enough valid points
            valid_points = sum(1 for p in lane_points if p >= 0)
            if valid_points >= 2:
                lanes.append(lane_points)
        
        lanes_batch.append(lanes)
    
    return lanes_batch


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
        "tensor_cores_available": torch.cuda.get_device_capability(0)[0] >= 7,
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
        self.predictions = []
    
    def add_batch_metrics(self, data_load_time, inference_time, 
                         gpu_memory, gpu_memory_alloc, batch_size, predictions=None):
        self.data_loading_times.append(data_load_time)
        self.inference_times.append(inference_time)
        self.total_times.append(data_load_time + inference_time)
        self.gpu_memory_used.append(gpu_memory)
        self.gpu_memory_allocated.append(gpu_memory_alloc)
        self.batch_sizes.append(batch_size)
        if predictions:
            self.predictions.extend(predictions)
    
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
                       labels_dict, num_warmup=10, log_frequency=10, collect_predictions=True):
    """Run inference benchmark with specified precision and accuracy measurement"""
    metrics = BenchmarkMetrics()
    
    print(f"\n{'='*60}")
    print(f"PRECISION MODE: {precision_mode.upper()}")
    print(f"Batch Size: {batch_size}")
    print(f"{'='*60}\n")
    
    use_amp = (precision_mode == "amp")
    use_fp16_input = (precision_mode == "fp16")
    
    print(f"Running warmup ({num_warmup} iterations)...")
    with torch.no_grad():
        for i, batch_data in enumerate(dataloader):
            if i >= num_warmup:
                break
            imgs = batch_data[0].to(device)
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
    all_predictions = []
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            imgs, ori_ws, ori_hs, paths, labels = batch_data
            
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
            
            if collect_predictions:
                lanes_batch = decode_predictions(
                    pred, ori_ws[0].item(), ori_hs[0].item()
                )

                for i, path in enumerate(paths):
                    try:
                        path_obj = Path(path)
                        path_parts = path_obj.parts
                        if 'clips' in path_parts:
                            clips_idx = path_parts.index('clips')
                            img_name = '/'.join(path_parts[clips_idx:])
                        else:
                            img_name = str(
                                path_obj.relative_to(path_obj.parents[2])
                            ).replace('\\', '/')
                    except:
                        img_name = Path(path).name

                    all_predictions.append({
                        'raw_file': img_name,
                        'lanes': lanes_batch[i]
                    })
                    
            # GPU memory usage
            if device.type == 'cuda':
                gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**2
                gpu_mem_allocated = torch.cuda.memory_allocated() / 1024**2
            else:
                gpu_mem_reserved = 0
                gpu_mem_allocated = 0
            
            current_batch_size = imgs.shape[0]
            metrics.add_batch_metrics(
                data_load_time, inference_time, 
                gpu_mem_reserved, gpu_mem_allocated, current_batch_size,
                predictions=all_predictions[-current_batch_size:]
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
    
    print(f"\n[{precision_mode.upper()}] Completed benchmark on {total_images} images")
    
    # comp accuracy
    print("Computing accuracy metrics...")
    accuracy_metrics = compute_tusimple_accuracy(all_predictions, labels_dict)
    print(f"Accuracy: {accuracy_metrics['accuracy']:.4f}")
    print(f"F1 Score: {accuracy_metrics['f1_score']:.4f}\n")
    
    return metrics, accuracy_metrics


def print_results(precision_mode: str, metrics: BenchmarkMetrics, accuracy_metrics: Dict,
                 baseline_stats: Dict = None, baseline_accuracy: Dict = None, 
                 save_path: Path = None):
    """Print and save benchmark results with accuracy"""
    stats = metrics.compute_statistics()
    stats['accuracy'] = accuracy_metrics
    
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
    
    print("ACCURACY:")
    print(f"  Accuracy: {accuracy_metrics['accuracy']:.4f}")
    print(f"  Precision: {accuracy_metrics['precision']:.4f}")
    print(f"  Recall: {accuracy_metrics['recall']:.4f}")
    print(f"  F1 Score: {accuracy_metrics['f1_score']:.4f}\n")
    
    print("GPU MEMORY:")
    print(f"  Peak Allocated: {stats['gpu_memory']['max_allocated_mb']:.2f} MB")
    print(f"  Peak Reserved:  {stats['gpu_memory']['max_reserved_mb']:.2f} MB\n")
    
    if baseline_stats and baseline_accuracy:
        baseline_fps = baseline_stats['throughput']['images_per_second']
        current_fps = stats['throughput']['images_per_second']
        speedup = current_fps / baseline_fps
        
        baseline_mem = baseline_stats['gpu_memory']['max_allocated_mb']
        current_mem = stats['gpu_memory']['max_allocated_mb']
        memory_reduction = (baseline_mem - current_mem) / baseline_mem * 100
        
        baseline_acc = baseline_accuracy['accuracy']
        current_acc = accuracy_metrics['accuracy']

        if baseline_acc > 0:
            accuracy_change = (current_acc - baseline_acc) / baseline_acc * 100
            print("COMPARISON TO FP32 BASELINE:")
            print(f"  Throughput Speedup: {speedup:.2f}x ({baseline_fps:.1f} → {current_fps:.1f} images/s)")
            print(f"  Memory Reduction: {memory_reduction:.1f}% ({baseline_mem:.1f}MB → {current_mem:.1f}MB)")
            print(f"  Accuracy Change: {accuracy_change:+.2f}% ({baseline_acc:.4f} → {current_acc:.4f})")
            print(f"  F1 Score Change: {(accuracy_metrics['f1_score'] - baseline_accuracy['f1_score']):.4f}\n")
        else:
            print("COMPARISON TO FP32 BASELINE:")
            print(f"  Throughput Speedup: {speedup:.2f}x ({baseline_fps:.1f} → {current_fps:.1f} images/s)")
            print(f"  Memory Reduction: {memory_reduction:.1f}% ({baseline_mem:.1f}MB → {current_mem:.1f}MB)")
            print(f"  Accuracy: Not computed (decoder not implemented)\n")
    
    print(f"{'='*60}\n")
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Results saved to: {save_path}\n")
    
    return stats


def create_comparison_plots(results: Dict):
    """Create comparison visualizations including accuracy"""
    
    modes = ['FP32', 'FP16', 'AMP']
    fp32_stats, fp16_stats, amp_stats = results['fp32'], results['fp16'], results['amp']
    
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
    accuracies = [
        fp32_stats['accuracy']['accuracy'],
        fp16_stats['accuracy']['accuracy'],
        amp_stats['accuracy']['accuracy']
    ]
    f1_scores = [
        fp32_stats['accuracy']['f1_score'],
        fp16_stats['accuracy']['f1_score'],
        amp_stats['accuracy']['f1_score']
    ]
    
    comparison_table = wandb.Table(
        columns=["Precision", "Throughput (img/s)", "Latency (ms)", 
                 "Memory (MB)", "Accuracy", "F1 Score", "Speedup", "Acc Change"],
        data=[
            [
                mode,
                f"{throughput:.2f}",
                f"{latency:.2f}",
                f"{memory:.2f}",
                f"{acc:.4f}",
                f"{f1:.4f}",
                f"{throughput / throughputs[0]:.2f}x",
                f"{(acc - accuracies[0]) / accuracies[0] * 100:+.2f}%"
            ]
            for mode, throughput, latency, memory, acc, f1 in 
            zip(modes, throughputs, latencies, memories, accuracies, f1_scores)
        ]
    )
    wandb.log({"comparison/precision_comparison_table": comparison_table})
    
    best_throughput_idx = np.argmax(throughputs)
    best_accuracy_idx = np.argmax(accuracies)
    
    print(f"\n{'='*60}")
    print("PRECISION OPTIMIZATION ANALYSIS")
    print(f"{'='*60}\n")
    print(f"Best Throughput: {modes[best_throughput_idx]} ({throughputs[best_throughput_idx]:.2f} img/s)")
    print(f"  Speedup: {throughputs[best_throughput_idx] / throughputs[0]:.2f}x over FP32")
    print(f"  Accuracy: {accuracies[best_throughput_idx]:.4f}")
    print(f"\nBest Accuracy: {modes[best_accuracy_idx]} ({accuracies[best_accuracy_idx]:.4f})")
    print(f"  Throughput: {throughputs[best_accuracy_idx]:.2f} img/s")
    print(f"  F1 Score: {f1_scores[best_accuracy_idx]:.4f}")
    print(f"\n{'='*60}\n")
    
    wandb.log({
        "comparison/best_throughput_mode": modes[best_throughput_idx],
        "comparison/best_accuracy_mode": modes[best_accuracy_idx],
        "comparison/max_speedup": throughputs[best_throughput_idx] / throughputs[0],
        "comparison/max_accuracy": max(accuracies),
        "comparison/accuracy_drop_fp16": (accuracies[0] - accuracies[1]) / accuracies[0] * 100,
        "comparison/accuracy_drop_amp": (accuracies[0] - accuracies[2]) / accuracies[0] * 100,
    })


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16
    num_workers = 4
    max_images = None
    experiment_name = "ufldv2-mixed-precision-with-accuracy"
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Mixed precision optimization requires GPU.")
        return
    
    ckpt_path = UFLD_PATH / "weights" / "tusimple_res18.pth"
    dataset_dir = PROJECT_ROOT / "datasets" / "TUSimple" / "test_set"
    label_path = PROJECT_ROOT / "datasets" / "TUSimple" / "test_label.json"
    results_dir = PROJECT_ROOT / "results" / "mixed_precision_accuracy"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    labels_dict = load_tusimple_labels(label_path)
    
    gpu_info = get_gpu_info()
    system_info = get_system_info()

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
    }
    
    wandb.init(
        project="ufldv2-optimization",
        name=experiment_name,
        group="simple-optimizations",
        config={
            **model_info,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "optimization_technique": "mixed_precision_with_accuracy",
            "precision_modes_tested": ["fp32", "fp16", "amp"],
            **system_info,
            **gpu_info,
            "dataset": "TuSimple",
            "evaluate_accuracy": True,
        },
        tags=["optimization", "mixed-precision", "accuracy", "lane-detection"],
    )
    
    image_paths = get_image_paths(dataset_dir, max_images)
    dataset = LaneDataset(image_paths, labels_dict)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate
    )
    
    all_results = {}
    
    # Test 1: FP32 Baseline
    print(f"\n{'#'*60}")
    print("# TEST 1: FP32 BASELINE")
    print(f"{'#'*60}\n")
    
    model_fp32 = load_model(ckpt_path, device, use_fp16=False)
    metrics_fp32, accuracy_fp32 = benchmark_inference(
        model_fp32, dataloader, device, "fp32", batch_size, labels_dict,
        num_warmup=10, log_frequency=10
    )
    stats_fp32 = print_results("fp32", metrics_fp32, accuracy_fp32,
                               save_path=results_dir / "fp32_metrics.json")
    all_results['fp32'] = stats_fp32
    
    wandb.log({
        "summary/fp32_throughput": stats_fp32['throughput']['images_per_second'],
        "summary/fp32_accuracy": accuracy_fp32['accuracy'],
        "summary/fp32_f1_score": accuracy_fp32['f1_score'],
    })
    
    del model_fp32
    torch.cuda.empty_cache()
    
    # Test 2: FP16 (model.half())
    print(f"\n{'#'*60}")
    print("# TEST 2: FP16 (NATIVE)")
    print(f"{'#'*60}\n")
    
    model_fp16 = load_model(ckpt_path, device, use_fp16=True)
    metrics_fp16, accuracy_fp16 = benchmark_inference(
        model_fp16, dataloader, device, "fp16", batch_size, labels_dict,
        num_warmup=10, log_frequency=10
    )
    stats_fp16 = print_results("fp16", metrics_fp16, accuracy_fp16, 
                               baseline_stats=stats_fp32, baseline_accuracy=accuracy_fp32,
                               save_path=results_dir / "fp16_metrics.json")
    all_results['fp16'] = stats_fp16
    
    wandb.log({
        "summary/fp16_throughput": stats_fp16['throughput']['images_per_second'],
        "summary/fp16_accuracy": accuracy_fp16['accuracy'],
        "summary/fp16_speedup": stats_fp16['throughput']['images_per_second'] / stats_fp32['throughput']['images_per_second'],
        "summary/fp16_accuracy_drop": (accuracy_fp32['accuracy'] - accuracy_fp16['accuracy']) / accuracy_fp32['accuracy'] * 100 if accuracy_fp32['accuracy'] > 0 else 0,
    })
    
    del model_fp16
    torch.cuda.empty_cache()
    
    # Test 3: Automatic Mixed Precision (AMP)
    print(f"\n{'#'*60}")
    print("# TEST 3: AUTOMATIC MIXED PRECISION (AMP)")
    print(f"{'#'*60}\n")
    
    model_amp = load_model(ckpt_path, device, use_fp16=False)
    metrics_amp, accuracy_amp = benchmark_inference(
        model_amp, dataloader, device, "amp", batch_size, labels_dict,
        num_warmup=10, log_frequency=10
    )
    stats_amp = print_results("amp", metrics_amp, accuracy_amp,
                             baseline_stats=stats_fp32, baseline_accuracy=accuracy_fp32,
                             save_path=results_dir / "amp_metrics.json")
    all_results['amp'] = stats_amp
    
    wandb.log({
        "summary/amp_throughput": stats_amp['throughput']['images_per_second'],
        "summary/amp_accuracy": accuracy_amp['accuracy'],
        "summary/amp_speedup": stats_amp['throughput']['images_per_second'] / stats_fp32['throughput']['images_per_second'],
        "summary/amp_accuracy_drop": (accuracy_fp32['accuracy'] - accuracy_amp['accuracy']) / accuracy_fp32['accuracy'] * 100 if accuracy_fp32['accuracy'] > 0 else 0,
    })
    
    del model_amp
    torch.cuda.empty_cache()
    
    print("\nCreating comparison visualizations...")
    create_comparison_plots(all_results)
    
    all_results_path = results_dir / "precision_comparison_with_accuracy.json"
    with open(all_results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"All results saved to: {all_results_path}\n")
    
    wandb.finish()


if __name__ == "__main__":
    main()