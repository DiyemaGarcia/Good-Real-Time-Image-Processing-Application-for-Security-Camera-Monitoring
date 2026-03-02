"""
Metrics computation for feature detection and matching evaluation
"""
import numpy as np
from typing import List, Tuple
import cv2
from scipy.stats import entropy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate various evaluation metrics for feature detection and matching"""
    
    def __init__(self, image_size: Tuple[int, int] = (512, 512), num_bins: int = 8):
        """
        Initialize MetricsCalculator
        
        Args:
            image_size: Size of the image (width, height)
            num_bins: Number of bins for spatial distribution entropy calculation
        """
        self.image_size = image_size
        self.num_bins = num_bins
    
    def compute_spatial_distribution_entropy(self, keypoints: List[cv2.KeyPoint]) -> float:
        """
        Compute spatial distribution entropy of keypoints
        Higher entropy indicates more uniform distribution
        
        Args:
            keypoints: List of detected keypoints
            
        Returns:
            Spatial distribution entropy
        """
        if not keypoints:
            return 0.0
        
        points = np.array([kp.pt for kp in keypoints])
        
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        
        x_bins = np.linspace(0, self.image_size[0], self.num_bins + 1)
        y_bins = np.linspace(0, self.image_size[1], self.num_bins + 1)
        
        hist, _, _ = np.histogram2d(x_coords, y_coords, bins=[x_bins, y_bins])
        
        hist_flat = hist.flatten()
        hist_flat = hist_flat / hist_flat.sum()
        
        spatial_entropy = entropy(hist_flat, base=2)
        
        return spatial_entropy
    
    def compute_repeatability(self, keypoints1: List[cv2.KeyPoint], 
                             keypoints2: List[cv2.KeyPoint],
                             homography: np.ndarray = None,
                             distance_threshold: float = 3.0) -> float:
        """Compute repeatability score between two sets of keypoints"""
        if not keypoints1 or not keypoints2:
            return 0.0
        
        points1 = np.float32([kp.pt for kp in keypoints1]).reshape(-1, 1, 2)
        points2 = np.float32([kp.pt for kp in keypoints2])
        
        if homography is not None:
            points1_transformed = cv2.perspectiveTransform(points1, homography).reshape(-1, 2)
        else:
            points1_transformed = points1.reshape(-1, 2)
        
        repeated_count = 0
        for p1 in points1_transformed:
            distances = np.linalg.norm(points2 - p1, axis=1)
            if np.min(distances) < distance_threshold:
                repeated_count += 1
        
        repeatability = repeated_count / len(keypoints1)
        
        return repeatability
    
    def compute_matching_score(self, matches: List[cv2.DMatch],
                               keypoints1: List[cv2.KeyPoint],
                               keypoints2: List[cv2.KeyPoint],
                               ground_truth_homography: np.ndarray = None,
                               distance_threshold: float = 3.0) -> float:
        """Compute matching score based on geometric verification"""
        if not matches or ground_truth_homography is None:
            return 0.0
        
        points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
        
        points1_transformed = cv2.perspectiveTransform(points1, ground_truth_homography).reshape(-1, 2)
        
        errors = np.linalg.norm(points2 - points1_transformed, axis=1)
        
        correct_matches = np.sum(errors < distance_threshold)
        
        score = correct_matches / len(matches) if len(matches) > 0 else 0.0
        
        return score
    
    def compute_feature_distribution_metrics(self, keypoints: List[cv2.KeyPoint]) -> dict:
        """Compute various metrics related to feature distribution"""
        if not keypoints:
            return {
                'spatial_entropy': 0.0,
                'coverage': 0.0,
                'density': 0.0,
                'std_x': 0.0,
                'std_y': 0.0
            }
        
        points = np.array([kp.pt for kp in keypoints])
        
        spatial_entropy = self.compute_spatial_distribution_entropy(keypoints)
        
        x_range = points[:, 0].max() - points[:, 0].min()
        y_range = points[:, 1].max() - points[:, 1].min()
        coverage = (x_range * y_range) / (self.image_size[0] * self.image_size[1])
        
        density = len(keypoints) / (self.image_size[0] * self.image_size[1])
        
        std_x = np.std(points[:, 0])
        std_y = np.std(points[:, 1])
        
        return {
            'spatial_entropy': spatial_entropy,
            'coverage': coverage,
            'density': density,
            'std_x': std_x,
            'std_y': std_y
        }
    
    @staticmethod
    def compute_descriptor_statistics(descriptors: np.ndarray) -> dict:
        """Compute statistics of descriptors"""
        if descriptors is None or len(descriptors) == 0:
            return {
                'mean_norm': 0.0,
                'std_norm': 0.0,
                'mean_value': 0.0,
                'std_value': 0.0
            }
        
        norms = np.linalg.norm(descriptors, axis=1)
        
        return {
            'mean_norm': np.mean(norms),
            'std_norm': np.std(norms),
            'mean_value': np.mean(descriptors),
            'std_value': np.std(descriptors)
        }
    
    def compute_all_metrics(self, 
                           keypoints1: List[cv2.KeyPoint],
                           keypoints2: List[cv2.KeyPoint],
                           descriptors1: np.ndarray,
                           descriptors2: np.ndarray,
                           matches: List[cv2.DMatch]) -> dict:
        """Compute all evaluation metrics"""
        num_features1 = len(keypoints1) if keypoints1 else 0
        num_features2 = len(keypoints2) if keypoints2 else 0
        
        num_matches = len(matches) if matches else 0
        
        if num_features1 > 0 and num_features2 > 0:
            matching_efficiency = (num_matches / min(num_features1, num_features2)) * 100
        else:
            matching_efficiency = 0.0
        
        if matches:
            avg_distance = sum(m.distance for m in matches) / len(matches)
        else:
            avg_distance = 0.0
        
        entropy1 = self.compute_spatial_distribution_entropy(keypoints1)
        entropy2 = self.compute_spatial_distribution_entropy(keypoints2)
        
        return {
            'num_features1': num_features1,
            'num_features2': num_features2,
            'num_matches': num_matches,
            'matching_efficiency': matching_efficiency,
            'avg_matching_distance': avg_distance,
            'spatial_entropy1': entropy1,
            'spatial_entropy2': entropy2
        }