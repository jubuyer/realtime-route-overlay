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
## 2. Repository Outline

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
