# Real-Time Route Overlay for Augmented Visual Navigation

**Team:** Jubayer Ahmed, Kahou Lei, Phoebe Tang, Tianshu Chu

**Institution:** Columbia University  

---

### Overview
This repository implements a real-time augmented reality navigation pipeline that overlays GPS-based routes from OpenStreetMap or recorded GPS logs directly onto live or pre-recorded video feeds (e.g., walking or driving footage). The system aligns the map coordinates with the camera’s visual perspective to create an intuitive visual navigation experience.

Our focus is on real-time inference performance, low-latency overlay rendering, and efficient edge deployment using NVIDIA GPUs and TensorRT optimization.

---

### Core Components
- **Semantic Segmentation:** YOLOv8-seg to detect drivable areas and scene structure.  
- **Geometric Alignment:** SuperPoint + SuperGlue for frame–map homography estimation.  
- **Overlay Rendering:** OpenCV-based augmentation of route paths onto live video streams.  
- **Optimization:** Mixed precision (FP16), INT8 quantization (TensorRT), and efficient batching.  
- **Experiment Tracking:** W&B integration for real-time metric visualization.  

---

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
