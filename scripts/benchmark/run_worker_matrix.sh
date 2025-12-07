#!/bin/bash

VIDEOS=(
  "datasets/kitti_raw/2011_09_26/2011_09_26_drive_0052_sync/drive_0052_sync.mp4"
  "datasets/kitti_raw/2011_09_26/2011_09_26_drive_0015_sync/drive_0015_sync.mp4"
  "datasets/kitti_raw/2011_10_03/2011_10_03_drive_0042_sync/drive_0042_sync.mp4"
)

WORKERS=(0 2 4 8)

PROJECT="ufldv2-optimization"

for VIDEO in "${VIDEOS[@]}"; do
  STEM=$(basename "$VIDEO" .mp4)
  for NW in "${WORKERS[@]}"; do
    RUN_NAME="gpu-opt-${STEM}-workers${NW}-$(date +%Y%m%d_%H%M%S)"
    echo "Running: $VIDEO with --num-workers $NW"
    python3 scripts/benchmark/optimized_video_pipeline.py \
      --video "$VIDEO" \
      --num-workers "$NW" \
      --use-wandb \
      --wandb-project "$PROJECT" \
      --run-name "$RUN_NAME"
  done
done