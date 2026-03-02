"""
Experiment runner for feature detection comparison
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import time
import json
from tqdm import tqdm

import gc
import torch

import sys
sys.path.append('..')

from models import SIFTDetector, SURFDetector, SuperPointDetector, D2NetDetector
from utils import ImageLoader, FeatureMatcher, MetricsCalculator
from visualization import ResultsPlotter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Run feature detection and matching experiments"""
    
    def __init__(self, config: dict):
        """Initialize ExperimentRunner"""
        self.config = config
        self.results_dir = Path(config.get('results_dir', 'results'))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.image_loader = ImageLoader(
            target_size=config['image_config']['target_size'],
            grayscale=config['image_config']['grayscale']
        )
        
        self.matcher = FeatureMatcher(
            matcher_type=config['matching_config']['matcher_type'],
            norm_type=config['matching_config']['norm_type'],
            cross_check=config['matching_config']['cross_check'],
            ratio_test=config['matching_config']['ratio_test']
        )
        
        self.metrics_calculator = MetricsCalculator(
            image_size=config['image_config']['target_size'],
            num_bins=config['metrics_config']['spatial_entropy_bins']
        )
        
        self.plotter = ResultsPlotter(
            figure_size=config['vis_config']['figure_size'],
            dpi=config['vis_config']['dpi']
        )
        
        self.detectors = self._initialize_detectors(config)
        
        logger.info(f"Initialized ExperimentRunner with {len(self.detectors)} detectors")
    
    def _initialize_detectors(self, config: dict) -> Dict:
        """Initialize all feature detectors"""
        detectors = {}
        
        if 'SIFT' in config['experiment_config']['methods']:
            try:
                detectors['SIFT'] = SIFTDetector(**config['sift_config'])
                logger.info("SIFT detector initialized")
            except Exception as e:
                logger.error(f"Failed to initialize SIFT: {str(e)}")
        
        if 'SURF' in config['experiment_config']['methods']:
            try:
                detectors['SURF'] = SURFDetector(**config['surf_config'])
                logger.info("SURF detector initialized")
            except Exception as e:
                logger.error(f"Failed to initialize SURF: {str(e)}")
        
        if 'SuperPoint' in config['experiment_config']['methods']:
            try:
                detectors['SuperPoint'] = SuperPointDetector(**config['superpoint_config'])
                logger.info("SuperPoint detector initialized")
            except Exception as e:
                logger.error(f"Failed to initialize SuperPoint: {str(e)}")
        
        if 'D2-Net' in config['experiment_config']['methods']:
            try:
                detectors['D2-Net'] = D2NetDetector(**config['d2net_config'])
                logger.info("D2-Net detector initialized")
            except Exception as e:
                logger.error(f"Failed to initialize D2-Net: {str(e)}")
        
        return detectors
    
    def run_single_experiment(self,
                             image1: np.ndarray,
                             image2: np.ndarray,
                             method_name: str,
                             pair_name: str = "pair",
                             visualize: bool = True) -> Dict:
        """Run experiment for a single method on an image pair"""
        if method_name not in self.detectors:
            logger.error(f"Method {method_name} not available")
            return {}
        
        detector = self.detectors[method_name]
        
        logger.info(f"Running {method_name} on {pair_name}")
        
        start_time = time.time()
        keypoints1, descriptors1 = detector.detectAndCompute(image1)
        keypoints2, descriptors2 = detector.detectAndCompute(image2)
        detection_time = time.time() - start_time
        
        start_time = time.time()
        matches = self.matcher.match_features(descriptors1, descriptors2)
        matching_time = time.time() - start_time
        
        metrics = self.metrics_calculator.compute_all_metrics(
            keypoints1, keypoints2,
            descriptors1, descriptors2,
            matches
        )
        
        metrics['detection_time'] = detection_time
        metrics['matching_time'] = matching_time
        metrics['total_time'] = detection_time + matching_time
        
        if visualize and self.config['experiment_config']['save_visualizations']:
            viz_dir = self.results_dir / pair_name / method_name
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            self.plotter.plot_keypoints(
                image1, keypoints1,
                title=f"{method_name} - Image 1",
                save_path=str(viz_dir / "keypoints_img1.png")
            )
            self.plotter.close_all()
            
            self.plotter.plot_keypoints(
                image2, keypoints2,
                title=f"{method_name} - Image 2",
                save_path=str(viz_dir / "keypoints_img2.png")
            )
            self.plotter.close_all()
            
            self.plotter.plot_matches(
                image1, image2,
                keypoints1, keypoints2,
                matches,
                title=f"{method_name} - Matches",
                save_path=str(viz_dir / "matches.png")
            )
            self.plotter.close_all()
            
            self.plotter.plot_spatial_distribution(
                image1, keypoints1,
                title=f"{method_name} - Spatial Distribution (Image 1)",
                save_path=str(viz_dir / "spatial_dist_img1.png")
            )
            self.plotter.close_all()
        
        logger.info(f"{method_name} completed: {metrics['num_matches']} matches, "
                   f"{metrics['matching_efficiency']:.2f}% efficiency")
        
        # Delete large objects
        del keypoints1, keypoints2, descriptors1, descriptors2, matches
        if 'detector' in locals():
            del detector
        
        # Force garbage collection
        gc.collect()
        
        # Clear PyTorch cache if using CPU or GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return metrics
    
    def run_dataset_experiment(self,
                              dataset_name: str,
                              dataset_path: str) -> Dict[str, List[Dict]]:
        """Run experiments on entire dataset"""
        logger.info(f"Running experiments on {dataset_name} dataset")
        
        image_pairs = self.image_loader.load_dataset(dataset_path)
        
        if not image_pairs:
            logger.error(f"No image pairs found in {dataset_path}")
            return {}
        
        results = {method: [] for method in self.detectors.keys()}
        
        # Visualize only first 3 pairs to save memory
        num_pairs_to_visualize = 3

        for idx, (img1, img2, pair_name) in enumerate(tqdm(image_pairs, desc=f"Processing {dataset_name}")):
            # Only visualize first few pairs
            should_visualize = (idx < num_pairs_to_visualize) and self.config['experiment_config']['save_visualizations']
            
            for method_name in self.detectors.keys():
                metrics = self.run_single_experiment(
                    img1, img2,
                    method_name,
                    pair_name=f"{dataset_name}/{pair_name}",
                    visualize=should_visualize
                )
                
                if metrics:
                    metrics['pair_name'] = pair_name
                    results[method_name].append(metrics)
        
        if self.config['experiment_config']['save_results']:
            results_file = self.results_dir / f"{dataset_name}_results.json"
            with open(results_file, 'w') as f:
                json_results = {}
                for method, method_results in results.items():
                    json_results[method] = []
                    for result in method_results:
                        json_result = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                     for k, v in result.items()}
                        json_results[method].append(json_result)
                
                json.dump(json_results, f, indent=2)
            logger.info(f"Saved results to {results_file}")
        
        return results
    
    def run_all_experiments(self) -> Dict[str, Dict[str, List[Dict]]]:
        """Run experiments on all datasets"""
        all_results = {}
        
        for dataset_name, dataset_path in self.config['datasets'].items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Starting experiments on {dataset_name}")
            logger.info(f"{'='*50}\n")
            
            results = self.run_dataset_experiment(dataset_name, str(dataset_path))
            all_results[dataset_name] = results
        
        return all_results
    
    def generate_summary_report(self, 
                               all_results: Dict[str, Dict[str, List[Dict]]]) -> None:
        """Generate summary report with comparison visualizations"""
        logger.info("Generating summary report")
        
        for dataset_name, dataset_results in all_results.items():
            avg_results = {}
            for method, method_results in dataset_results.items():
                if not method_results:
                    continue
                
                avg_metrics = {}
                for key in method_results[0].keys():
                    if key != 'pair_name' and isinstance(method_results[0][key], (int, float)):
                        values = [r[key] for r in method_results]
                        avg_metrics[key] = np.mean(values)
                
                avg_results[method] = avg_metrics
            
            if not avg_results:
                continue
            
            viz_dir = self.results_dir / f"{dataset_name}_summary"
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            self.plotter.plot_comparison_table(
                avg_results,
                save_path=str(viz_dir / "comparison_table.png")
            )
            self.plotter.close_all()
            
            for metric in ['num_matches', 'matching_efficiency', 'avg_matching_distance']:
                self.plotter.plot_metrics_bar_chart(
                    avg_results,
                    metric,
                    save_path=str(viz_dir / f"{metric}_comparison.png")
                )
                self.plotter.close_all()
            
            logger.info(f"Summary report generated for {dataset_name}")