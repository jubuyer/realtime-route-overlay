# HPML Project: Optimizing UFLDv2 For GPS Path Projection

## Team Information
- Jubayer Ahmed (ja3847)
- Kahou Lei (kl3641)
- Phoebe Tang (ft2619)
- Tianshu Chu (tc3396)

**Institution:** Columbia University  

**Wandb Dashboard**

View training and evaluation metrics here: https://wandb.ai/hpmlcoms6998-columbia-university/ufldv2-optimization

---

## 0. Problem Statement
In this work, we frame the idea of GPS Path Project on AR overlay as an optimization
challenge: how to improve the throughput and latency of a
lane-detection-based AR pipeline while preserving geometric
accuracy. Rather than focusing on training novel models,
we target system-level optimizations, including data loading,
batching, and inference execution, to enable scalable, real-time
AR navigation suitable for high-volume video processing.

## 1. Project Description
This repo contains a system-level optimization of Ultra Fast Lane Detection v2 (UFLDv2) for Augmented Reality (AR) navigation pipelines. We implement and evaluate a series of performance optimizations, including batch inference, mixed precision execution, and post-training quantization, to accelerate lane detection on GPU hardware while preserving geometric accuracy. Using high resolution driving data from KITTI dataset, we integrate the AR navigation pipeline that projects routes onto vehicle camera feeds. The system implements a comprehensive coordinate transformation pipeline that converts WGS84 coordinates to the 2D image plane, taking into account vehicle dynamics (Yaw, Pitch, Roll) and sensor calibration.

---

### Base Model: Ultra Fast Lane Detection v2

### What is UFLDv2?
Ultra Fast Lane Detection v2 is a state-of-the-art lane detection model that reformulates lane detection as a row-wise classification problem rather than traditional pixel-wise segmentation. This approach significantly reduces computational cost while maintaining high accuracy.

**Key Features:**
- **Fast**: Achieves 200+ FPS on high-end GPUs
- **Accurate**: Competitive F1 scores on TuSimple and CULane benchmarks
- **Efficient**: Lightweight architecture suitable for real-time applications
- **Row-based Selection**: Uses global image features for efficient lane prediction

**Original Paper**: [Ultra Fast Deep Lane Detection with Hybrid Anchor Driven Ordinal Classification](https://arxiv.org/abs/2206.07389)

**Original Repository**: https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2

### Model Architecture
- **Backbone**: ResNet-18 (for speed) or ResNet-34 (for accuracy)
- **Input Size**: 1640 × 590 pixels (TuSimple dataset)
- **Output**: Lane coordinates represented as row-wise anchor points
- **Parameters**: ~11M (ResNet-18), ~21M (ResNet-34)

---
## 2. Repository outline

Below is an expanded description of the repository layout and the purpose of the most relevant files and folders. Use this as a quick reference to find code, data helpers, experiments, and results.

Realtime-Route-Overlay (top-level)

- configs/
	- env.yml — Conda environment specification used to reproduce the development environment (packages, channels, python version). Used with `conda env create -f configs/env.yml`.

- datasets/
	- README.md — Instructions for acquiring and organizing datasets (KITTI, TuSimple, etc.).
	- (this directory is intended to contain dataset download scripts or dataset pointers; datasets are too large to upload to github so we chose to keep them locally)

- logs/
	- Stores experiment logs, saved benchmark outputs and WandB run exports. Example subfolders include `benchmarks/` (plain-text / JSON benchmark results) and `wandb/` (archived WandB runs).

- models/
	- Ultra-Fast-Lane-Detection-v2/ — Copy of the UFLDv2 model code (architecture and model utilities). This folder is the base lane-detection implementation used by the project and may include model definitions, checkpoints, and conversion scripts.

- notebooks/
	- Jupyter notebooks used for dataset download

- quantization/
	- Contains artifacts and notebooks related to model quantization and hardware-optimized exports. Example files in this repo:
		- `Snapdragon_optimized.ipynb` — notebook demonstrating device-specific optimization steps.
		- `ufldv2_res18_FP32.onnx` — exported ONNX model used as a starting point for quantization/optimization.

- results/
	- Stores processed experiment outputs and figures used for evaluation. Typical subfolders include:
		- `baseline/` — results from baseline runs
		- `batch_optimization/` — results from batch-sizing experiments
		- `mixed_precision/` and `mixed_precision_accuracy/` — experiments and accuracy comparisons for AMP/mixed-precision runs

- scripts/
	- Main collection of runnable Python scripts and utilities used by the project. Key scripts:
		- `dataloader.py` — dataset loading utilities and PyTorch dataset/dataloader wrappers.
		- `diagnose_import.py` — quick environment / import checks to validate dependencies.
		- `download_kitti.py` — helper to download KITTI data
		- `geometry.py` — 
		- `inference.py` — inference runner for UFLDv2 (single-image inference harness).
		- `kitti_frames_to_video.py` — converts KITTI frames into a stitched video for visualization/benchmarking.
		- `main_pipeline.py` — 
		- `maps_client.py` — 
		- `preprocess_davis.py` / `preprocess_kitti.py` — dataset-specific preprocessing scripts.
		- `verify_setup.py` — convenience script to validate that environment, devices, and key files are present.
		- `visualizer.py` — 
	- Additional script folders:
		- `scripts/benchmark/` — benchmarking helpers and runners used to collect throughput/latency metrics.
		- `scripts/GPU Provisioning/` — scripts used to provision GPU VMs from edstem
		- `scripts/simple_optimization/` — experimental optimization scripts.

- README.md
	- This file — high-level project overview, setup instructions, and the section you are reading now.

## 3. Set Up Instructions and Commands
### Environment Setup
First, create a VM and download all the requirements. 

```bash
git clone https://github.com/jubuyer/realtime-route-overlay.git
cd realtime-route-overlay

conda env create -f configs/env.yml
conda activate motiondet

# setup wandb
wandb login
```

### Dataset Setup

Please download relevant datasets (KITTI and TuSimple). Since datasets are too large to upload to main, we chose to keep them locally and included instructions on downloading them.

#### TuSimple Dataset
The TuSimple dataset contains highway driving scenarios with lane annotations.

**Download Options:**
1. **Kaggle**: https://www.kaggle.com/datasets/manideep1108/tusimple
2. **Official**: https://github.com/TuSimple/tusimple-benchmark/issues/3

**Dataset Structure:**
The downloaded folder should follow this structure:
```
data/tusimple/
├── clips/
│   ├── 0313-1/
│   ├── 0313-2/
│   ├── 0531/
│   └── 0601/
├── label_data_0313.json
├── label_data_0531.json
├── label_data_0601.json
├── test_tasks_0627.json
└── test_label.json
```

Please drag folder to `datasets/`

#### KITTI Raw Suite
The KITTI Raw Road dataset was used to benchmark inference. Accessing this dataset requires an official account.

Please run
`python scripts/download_kitti.py` to download the KITTI dataset.

### Pretrained Model Weights

Please download the model weights and place them in the correct repository (instructions below)

**Available Models:**
- `tusimple_res18.pth` - ResNet-18 backbone (faster, \~11M parameters)
- `tusimple_res34.pth` - ResNet-34 backbone (more accurate, \~21M parameters)

**Download:**

Model weights are available for download on the [model repo](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2) under trained model section. Please drag the downloaded weights to `models/Ultra-Fast-Lane-Detection-v2/weights/`.

---

### List of Commands

Below are the most common commands to verify the environment and run the benchmarking / optimization experiments used in this project. Most scripts use hard-coded paths near the top of the file for checkpoints, dataset locations and results directories — edit those variables if you want to run different inputs or save to a different location.

Prerequisites
- Activate the conda environment created from `configs/env.yml` (or install requirements):

```bash
conda activate motiondet

# (Optional) Login to Weights & Biases to enable experiment logging
wandb login
```

1) Verify environment and required files

```bash
python scripts/verify_setup.py
```

Description: Runs a series of checks to confirm the repository layout, required directories, model weights, dataset presence, and GPU/PyTorch availability. The script prints a checklist and exits with non-zero status if critical components (datasets or weights) are missing.

When to run: Right after cloning and installing dependencies, before running experiments.

2) Diagnose UFLDv2 import issues

```bash
python scripts/diagnose_import.py
```

Description: Adds the `models/Ultra-Fast-Lane-Detection-v2` folder to `sys.path`, lists key files in the `model/` folder, attempts to import `torch`, then tries to import the UFLDv2 `parsingNet`. Useful for debugging import errors and viewing a traceback if the model fails to import.

When to run: If `verify_setup.py` reports missing model files or you get import errors when running inference.

3) Single-image inference & visualization

```bash
python scripts/inference.py
```

Description: Runs model inference on a single example image (paths and checkpoint are defined near the top of the script) and writes a visualization to `results/`. Use this for a quick functional check of the model and the projection/visualization pipeline.

Notes: The script uses hard-coded `ckpt_path`, `img_path`, and `out_path` variables; edit them in `scripts/inference.py` if you want to run a different image or checkpoint.

4) Baseline benchmark (throughput / latency)

```bash
python scripts/benchmark/benchmark_baseline.py
```

Description: Runs the baseline inference benchmark (default batch_size=1) across images in `datasets/TUSimple/test_set` and logs metrics and profiler traces to Weights & Biases. Results and a JSON summary are saved under `results/baseline/`.

Configuration: Edit variables in the script (e.g., `batch_size`, `num_workers`, `max_images`, `experiment_name`, `ckpt_path`, `dataset_dir`) to tune the run. Ensure `wandb` is configured if you want cloud logging.

5) Batch inference optimization (test multiple batch sizes)

```bash
python scripts/simple_optimization/batch_inference_optimization.py
```

Description: Iterates over a set of batch sizes (default `[1, 4, 8, 16, 32]`), measures throughput, latency and GPU memory, and records results to `results/batch_optimization/` and WandB. The script saves per-batch JSON metrics and a combined comparison file.

Configuration: Modify `batch_sizes_to_test`, `num_workers`, `max_images`, and `ckpt_path` inside the script to control which batch sizes and dataset to benchmark.

6) Mixed-precision optimization (FP16 / AMP)

```bash
python scripts/simple_optimization/mixed_precision_optimization.py
```

Description: Runs precision comparison tests: FP32 baseline, native FP16 (model.half()), and Automatic Mixed Precision (AMP). Measures throughput, memory, and TuSimple-style accuracy (if labels are available). Results and accuracy summaries are saved to `results/` and logged to WandB.

Configuration: The script expects TuSimple labels (for accuracy) and allows configuring `batch_size`, `max_images`, and the label path near the top of the file. Edit those variables if your dataset layout differs.

## 4. Results

We evaluated three complementary optimization strategies for UFLDv2 inference: batch inference on GPU, mixed-precision inference on GPU, and INT8 post-training quantization (PTQ) on a mobile NPU. Across all experiments, we observe clear trade-offs between latency, throughput, memory footprint, and deployment constraints.

### Optimization 1: Batch Inference (GPU)

Batching significantly improves throughput by amortizing kernel launch and memory access overheads. Increasing the batch size from 1 to 4 yields a 2.07× throughput improvement (202.17 → 418.50 img/s) while reducing per-image latency (4.95 ms → 2.39 ms). Throughput peaks at batch sizes 4 and 16 (≈418 img/s), indicating near-optimal GPU utilization in this range. Beyond batch size 16, throughput declines due to increased memory pressure, with batch size 32 showing higher latency, reduced throughput, and substantially higher GPU memory usage.

| Batch Size | Mean (ms) | Median (ms) | Std Dev (ms) | P95 (ms) | Throughput (img/s) | Speedup (×) | GPU Mem (MB) |
|------------|-----------|-------------|--------------|----------|--------------------|-------------|--------------|
| 1  | 4.95 | 3.53 | 12.17 | 5.72 | 202.17 | 1.00 | 380.99 |
| 4  | 9.56 | 6.20 | 23.98 | 10.07 | 418.50 | 2.07 | 390.23 |
| 8  | 21.44 | 12.15 | 48.70 | 16.05 | 366.85 | 1.81 | 403.35 |
| 16 | 37.58 | 25.01 | 36.83 | 96.71 | 418.71 | 2.07 | 427.20 |
| 32 | 75.48 | 49.65 | 80.76 | 205.35 | 390.84 | 1.93 | 478.44 |

### Optimization 2: Mixed-Precision Inference (GPU, Batch = 16)

Mixed precision improves performance without sacrificing accuracy. Native FP16 delivers a 1.28× throughput increase over FP32 (370.66 → 474.43 img/s) while reducing peak GPU memory usage by 48.7%. AMP achieves the highest throughput (1.29× speedup) but provides negligible memory savings, as parameters remain in FP32. All precision modes yield identical accuracy and F1 scores, confirming numerical robustness.

| Metric | FP32 | FP16 | AMP |
|-------|------|------|-----|
| Mean Latency (ms) | 42.45 | 33.16 | 32.93 |
| P95 Latency (ms)  | 112.23 | 97.79 | 95.66 |
| Throughput (img/s) | 370.66 | 474.43 | 477.85 |
| Speedup vs FP32 | 1.00× | 1.28× | 1.29× |
| Peak GPU Mem (MB) | 429.15 | 220.32 | 427.99 |
| Accuracy | 0.4348 | 0.4348 | 0.4348 |
| F1 Score | 0.6061 | 0.6061 | 0.6061 |

### Optimization 3: INT8 PTQ on Snapdragon NPU

INT8 PTQ enables efficient edge deployment. On a Snapdragon-8 Gen 3 NPU, INT8 reduces mean latency by 65% (5.47 → 1.92 ms), yielding a 2.86× speedup and a 185.5% throughput increase. Model size is reduced by 58.5%, improving deployability on mobile devices. Output drift remains minimal (cosine similarity 0.996), indicating that aggressive quantization preserves numerical fidelity.

| Metric | FP32 | INT8 | Change |
|------|------|------|--------|
| Mean Latency (ms) | 5.47 | 1.92 | −65.0% |
| P95 Latency (ms) | 5.55 | 1.97 | −64.5% |
| Throughput (img/s) | 182.7 | 521.7 | +185.5% |
| Model Size (MB) | 367.70 | 152.53 | −58.5% |
| Cosine Similarity (FP32 vs INT8) | — | 0.996 | — |

### Overall Takeaway

Batching and mixed precision are effective for maximizing GPU throughput, with batch sizes 4–16 and native FP16 offering the best balance of speed and memory efficiency. For edge deployment, INT8 PTQ provides the largest gains in latency, throughput, and model size with minimal numerical drift, making it the most suitable configuration for real-time AR lane overlay on mobile hardware.

---
