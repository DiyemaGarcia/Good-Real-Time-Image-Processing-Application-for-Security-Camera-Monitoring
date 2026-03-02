# Feature Detection Comparison: Classical vs Deep Learning Methods

Replication of the study: *"Exploring Feature Detection: A Comparative Study of Classical and Deep Learning Methods Across Complex Scenes"* by Huichuan Zhou (2024).

## Overview

This project implements a comprehensive comparison of feature detection and matching algorithms across three challenging scenarios:
- **Repetitive Patterns**: Testing robustness to self-similar structures
- **Cluttered Backgrounds**: Evaluating performance in complex visual scenes
- **High Illumination**: Assessing stability under strong lighting variations

### Methods Compared

#### Classical Methods
- **SIFT** (Scale-Invariant Feature Transform)
  - Invariant to scale, rotation, and illumination
  - Robust keypoint detection via DoG pyramid
  
- **SURF** (Speeded-Up Robust Features)
  - Faster alternative to SIFT using box filters
  - Efficient computation via integral images

#### Deep Learning Methods
- **SuperPoint** (MagicLeap, 2018)
  - Self-supervised convolutional network
  - Joint keypoint detection and descriptor computation
  
- **D2-Net** (CVPR 2019)
  - VGG16-based architecture
  - Simultaneous detection and dense description

## Installation

### 1. Clone or download this project

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download pretrained models
```bash
python download_models.py
```

## Dataset Organization

Place your image pairs in the following structure:
```
data/
├── repeating_patterns/
│   ├── pair_001_img1.jpg
│   ├── pair_001_img2.jpg
│   ├── pair_002_img1.jpg
│   ├── pair_002_img2.jpg
│   └── ...
├── cluttered_backgrounds/
│   └── (same structure)
└── high_lighting/
    └── (same structure)
```

## Running Experiments

Open `main.ipynb` in Jupyter:
```bash
jupyter notebook main.ipynb
```

## Evaluation Metrics

Following the paper's methodology:

1. **Number of Features**: Total keypoints detected per image
2. **Number of Matches**: Successfully matched keypoint pairs
3. **Matching Efficiency (%)**: `(matches / min(features1, features2)) × 100`
4. **Average Matching Distance**: Mean descriptor distance for matches
5. **Spatial Distribution Entropy**: Uniformity of feature distribution (8×8 grid)

## Results

Results are saved in `results/` directory:

- **Visualizations**: 
  - Keypoint detection plots
  - Feature matching diagrams
  - Spatial distribution heatmaps
  
- **Data Files**:
  - JSON: Raw experimental data
  - CSV: Statistical summaries
  - TXT: Comparative reports

- **Summary Tables**: Performance comparison across all methods
