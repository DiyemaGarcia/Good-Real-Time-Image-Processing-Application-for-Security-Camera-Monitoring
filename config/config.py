"""
Configuration file for feature detection comparison experiments
"""
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "pretrained_models"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RESULTS_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Image preprocessing configuration
IMAGE_CONFIG = {
    'target_size': (512, 512),
    'grayscale': True,
    'normalize': False
}
'''# Image preprocessing configuration
IMAGE_CONFIG = {
    'target_size': (256, 256),  # ← Réduire de 512 à 256
    'grayscale': True,
    'normalize': False
}'''

# SIFT configuration
SIFT_CONFIG = {
    'nfeatures': 0,
    'nOctaveLayers': 3,
    'contrastThreshold': 0.04,
    'edgeThreshold': 10,
    'sigma': 1.6
}

# SURF configuration
SURF_CONFIG = {
    'hessianThreshold': 400,
    'nOctaves': 4,
    'nOctaveLayers': 3,
    'extended': False,
    'upright': False
}

# SuperPoint configuration
SUPERPOINT_CONFIG = {
    'nms_dist': 4,
    'conf_thresh': 0.015,
    'nn_thresh': 0.7,
    'cuda': True,
    'model_path': MODELS_DIR / 'superpoint_v1.pth'
}

# D2-Net configuration
D2NET_CONFIG = {
    'model_file': MODELS_DIR / 'd2_tf.pth',
    'max_edge': 1600,
    'max_sum_edges': 2800,
    'preprocessing': 'torch',
    'use_relu': True,
    'multiscale': False,
    'cuda': True
}

# Matching configuration
MATCHING_CONFIG = {
    'matcher_type': 'BF',
    'norm_type': 'L2',
    'cross_check': True,
    'ratio_test': 0.75
}

# Evaluation metrics
METRICS_CONFIG = {
    'spatial_entropy_bins': 8,
    'distance_threshold': 50
}

# Dataset paths
DATASETS = {
    'repeating_patterns': DATA_DIR / 'repeating_patterns',
    'cluttered_backgrounds': DATA_DIR / 'cluttered_backgrounds',
    'high_lighting': DATA_DIR / 'high_lighting'
}

# Create dataset directories
for dataset_path in DATASETS.values():
    dataset_path.mkdir(parents=True, exist_ok=True)

# Experiment configuration
EXPERIMENT_CONFIG = {
    'methods': ['SIFT', 'SuperPoint', 'D2-Net'],
    'datasets': list(DATASETS.keys()),
    'save_visualizations': True,
    'save_results': True
}
'''# Experiment configuration
EXPERIMENT_CONFIG = {
    'methods': ['SIFT', 'SuperPoint', 'D2-Net'],
    'datasets': list(DATASETS.keys()),
    'save_visualizations': False,  # ← CHANGEZ ICI
    'save_results': True
}'''

# Visualization configuration
VIS_CONFIG = {
    'figure_size': (15, 10),
    'dpi': 150,
    'feature_color': (0, 255, 0),
    'match_color': (0, 255, 255),
    'line_thickness': 1,
    'marker_size': 5
}

# URLs for pretrained models
MODEL_URLS = {
    'superpoint': 'https://github.com/magicleap/SuperPointPretrainedNetwork/raw/master/superpoint_v1.pth',
    'd2net': 'https://dusmanu.com/files/d2-net/d2_tf.pth'
}