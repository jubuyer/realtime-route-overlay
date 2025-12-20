# Dataset

## TuSimple Dataset
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
## KITTI Raw Suite
The KITTI Raw Road dataset was used to benchmark inference. Accessing this dataset requires an official account.

Please run
`python scripts/download_kitti.py` to download the KITTI dataset.