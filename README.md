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
```bash
git clone https://github.com/jubuyer/realtime-route-overlay.git
cd realtime-route-overlay

conda env create -f configs/env.yml
conda activate motiondet

# or use requirements.txt
pip install -r configs/requirements.txt

# setup wandb
wandb login
```

### Dataset Setup

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
#### KITTI Raw Suite
The KITTI Raw Road dataset was used to benchmark inference. Accessing this dataset requires an official account.

Please run
`python scripts/download_kitti.py` to download the KITTI dataset.

### Pretrained Model Weights

**Available Models:**
- `tusimple_res18.pth` - ResNet-18 backbone (faster, ~11M parameters)
- `tusimple_res34.pth` - ResNet-34 backbone (more accurate, ~21M parameters)

**Download:**
Model weights are available for download on the [model repo](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2) under trained model section.
---
### List of Commands

## 4. Results

---
