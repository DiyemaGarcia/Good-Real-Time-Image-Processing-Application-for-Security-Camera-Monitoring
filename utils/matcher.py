"""
Feature matching utilities
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureMatcher:
    """Handle feature matching between image pairs"""
    
    def __init__(self, matcher_type: str = 'BF', norm_type: str = 'L2', 
                 cross_check: bool = True, ratio_test: float = 0.75):
        """
        Initialize FeatureMatcher
        
        Args:
            matcher_type: Type of matcher ('BF' for Brute Force, 'FLANN' for FLANN-based)
            norm_type: Norm type for distance calculation ('L2' or 'HAMMING')
            cross_check: Whether to use cross-check in matching
            ratio_test: Threshold for Lowe's ratio test
        """
        self.matcher_type = matcher_type
        self.norm_type = norm_type
        self.cross_check = cross_check
        self.ratio_test_threshold = ratio_test
        
        if matcher_type == 'BF':
            if norm_type == 'L2':
                self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=cross_check)
            elif norm_type == 'HAMMING':
                self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=cross_check)
            else:
                raise ValueError(f"Unknown norm type: {norm_type}")
        elif matcher_type == 'FLANN':
            if norm_type == 'L2':
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            else:
                FLANN_INDEX_LSH = 6
                index_params = dict(algorithm=FLANN_INDEX_LSH,
                                  table_number=6,
                                  key_size=12,
                                  multi_probe_level=1)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            raise ValueError(f"Unknown matcher type: {matcher_type}")
        
        logger.info(f"Initialized {matcher_type} matcher with {norm_type} norm")
    
    def match_features(self, descriptors1: np.ndarray, descriptors2: np.ndarray,
                      use_ratio_test: bool = False) -> List[cv2.DMatch]:
        """
        Match features between two descriptor sets
        
        Args:
            descriptors1: Descriptors from first image
            descriptors2: Descriptors from second image
            use_ratio_test: Whether to apply Lowe's ratio test
            
        Returns:
            List of matches
        """
        if descriptors1 is None or descriptors2 is None:
            logger.warning("One or both descriptor sets are None")
            return []
        
        if len(descriptors1) == 0 or len(descriptors2) == 0:
            logger.warning("One or both descriptor sets are empty")
            return []
        
        try:
            if use_ratio_test and not self.cross_check:
                knn_matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)
                
                good_matches = []
                for match_pair in knn_matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < self.ratio_test_threshold * n.distance:
                            good_matches.append(m)
                
                logger.info(f"Ratio test: {len(good_matches)}/{len(knn_matches)} matches kept")
                return good_matches
            else:
                matches = self.matcher.match(descriptors1, descriptors2)
                logger.info(f"Found {len(matches)} matches")
                return matches
                
        except Exception as e:
            logger.error(f"Error during matching: {str(e)}")
            return []
    
    def filter_matches_by_distance(self, matches: List[cv2.DMatch], 
                                   threshold: float = 50.0) -> List[cv2.DMatch]:
        """Filter matches by distance threshold"""
        if not matches:
            return []
        
        good_matches = [m for m in matches if m.distance < threshold]
        logger.info(f"Distance filter: {len(good_matches)}/{len(matches)} matches kept")
        return good_matches
    
    def compute_matching_efficiency(self, num_matches: int, 
                                   num_features1: int, 
                                   num_features2: int) -> float:
        """Calculate matching efficiency as percentage"""
        if num_features1 == 0 or num_features2 == 0:
            return 0.0
        
        min_features = min(num_features1, num_features2)
        efficiency = (num_matches / min_features) * 100
        return efficiency
    
    def compute_average_matching_distance(self, matches: List[cv2.DMatch]) -> float:
        """Compute average distance of matches"""
        if not matches:
            return 0.0
        
        total_distance = sum(m.distance for m in matches)
        return total_distance / len(matches)
    
    @staticmethod
    def get_matched_points(keypoints1: List[cv2.KeyPoint], 
                          keypoints2: List[cv2.KeyPoint],
                          matches: List[cv2.DMatch]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract matched point coordinates"""
        if not matches:
            return np.array([]), np.array([])
        
        points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
        points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
        
        return points1, points2
    
    @staticmethod
    def filter_matches_with_ransac(keypoints1: List[cv2.KeyPoint],
                                   keypoints2: List[cv2.KeyPoint],
                                   matches: List[cv2.DMatch],
                                   ransac_threshold: float = 5.0) -> Tuple[List[cv2.DMatch], np.ndarray]:
        """Filter matches using RANSAC with homography estimation"""
        if len(matches) < 4:
            logger.warning("Not enough matches for RANSAC (minimum 4 required)")
            return matches, None
        
        points1, points2 = FeatureMatcher.get_matched_points(keypoints1, keypoints2, matches)
        
        H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, ransac_threshold)
        
        if mask is None:
            logger.warning("RANSAC failed to find homography")
            return matches, None
        
        matches_filtered = [m for m, is_inlier in zip(matches, mask.ravel()) if is_inlier]
        
        logger.info(f"RANSAC: {len(matches_filtered)}/{len(matches)} matches kept")
        return matches_filtered, H