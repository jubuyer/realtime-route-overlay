# HPML Project: Optimizing UFLDv2 For GPS Path Projection

## Team Information
- Jubayer Ahmed (ja3847)
- Kahou Lei (kl3641)
- Phoebe Tang (ft2619)
- Tianshu Chu (tc3396)

**Institution:** Columbia University  

---

## 1. Problem Statement
In this work, we frame the problem as an optimization
challenge: how to improve the throughput and latency of a
lane-detection-based AR pipeline while preserving geometric
accuracy. Rather than focusing on training novel models,
we target system-level optimizations, including data loading,
batching, and inference execution, to enable scalable, real-time
AR navigation suitable for high-volume video processing.

---

## 2. Model Description
### Base Model: Ultra Fast Lane Detection v2

### What is UFLDv2?
Ultra Fast Lane Detection v2 is a state-of-the-art lane detection model that reformulates lane detection as a row-wise classification problem rather than traditional pixel-wise segmentation. This approach significantly reduces computational cost while maintaining high accuracy.

**Key Features:**
- **Fast**: Achieves 300+ FPS on high-end GPUs
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

## 3. Final Results Summary

---

## 4. Reproducibility Instructions

### A. Requirements

#### Environment Setup
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

#### Dataset Setup

#### TuSimple Dataset
The TuSimple dataset contains highway driving scenarios with lane annotations.

**Download Options:**
1. **Kaggle**: https://www.kaggle.com/datasets/manideep1108/tusimple
2. **Official**: https://github.com/TuSimple/tusimple-benchmark/issues/3

**Dataset Structure:**
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

#### Pretrained Model Weights

**Available Models:**
- `tusimple_res18.pth` - ResNet-18 backbone (faster, ~11M parameters)
- `tusimple_res34.pth` - ResNet-34 backbone (more accurate, ~21M parameters)

**Download:**
Model weights are available for download on the [model repo](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2) under trained model section.
---

### B. Wandb Dashboard

View training and evaluation metrics here: https://wandb.ai/hpmlcoms6998-columbia-university/ufldv2-optimization

---
