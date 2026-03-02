"""
SIFT (Scale-Invariant Feature Transform) detector implementation
"""
import cv2
import numpy as np
from typing import Tuple, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SIFTDetector:
    """SIFT feature detector and descriptor"""
    
    def __init__(self, 
                 nfeatures: int = 0,
                 nOctaveLayers: int = 3,
                 contrastThreshold: float = 0.04,
                 edgeThreshold: int = 10,
                 sigma: float = 1.6):
        """Initialize SIFT detector"""
        self.nfeatures = nfeatures
        self.nOctaveLayers = nOctaveLayers
        self.contrastThreshold = contrastThreshold
        self.edgeThreshold = edgeThreshold
        self.sigma = sigma
        
        self.detector = cv2.SIFT_create(
            nfeatures=nfeatures,
            nOctaveLayers=nOctaveLayers,
            contrastThreshold=contrastThreshold,
            edgeThreshold=edgeThreshold,
            sigma=sigma
        )
        
        logger.info(f"Initialized SIFT detector")
    
    def detect(self, image: np.ndarray) -> List[cv2.KeyPoint]:
        """Detect keypoints in an image"""
        if image is None:
            logger.error("Input image is None")
            return []
        
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        keypoints = self.detector.detect(image, None)
        logger.info(f"Detected {len(keypoints)} SIFT keypoints")
        
        return keypoints
    
    def compute(self, image: np.ndarray, keypoints: List[cv2.KeyPoint]) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """Compute descriptors for detected keypoints"""
        if image is None:
            logger.error("Input image is None")
            return [], None
        
        if not keypoints:
            logger.warning("No keypoints provided")
            return [], None
        
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        keypoints, descriptors = self.detector.compute(image, keypoints)
        
        if descriptors is not None:
            logger.info(f"Computed descriptors for {len(keypoints)} keypoints")
        else:
            logger.warning("Failed to compute descriptors")
        
        return keypoints, descriptors
    
    def detectAndCompute(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """Detect keypoints and compute descriptors in one step"""
        if image is None:
            logger.error("Input image is None")
            return [], None
        
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        keypoints, descriptors = self.detector.detectAndCompute(image, mask)
        
        if keypoints and descriptors is not None:
            logger.info(f"SIFT detected {len(keypoints)} keypoints")
        else:
            logger.warning("SIFT detection failed or found no keypoints")
        
        return keypoints, descriptors
    
    @staticmethod
    def draw_keypoints(image: np.ndarray, 
                       keypoints: List[cv2.KeyPoint],
                       color: Tuple[int, int, int] = (0, 255, 0),
                       flags: int = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) -> np.ndarray:
        """Draw keypoints on image"""
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        output_image = cv2.drawKeypoints(
            image, keypoints, None, 
            color=color, 
            flags=flags
        )
        
        return output_image
    
    def get_config(self) -> dict:
        """Get detector configuration"""
        return {
            'method': 'SIFT',
            'nfeatures': self.nfeatures,
            'nOctaveLayers': self.nOctaveLayers,
            'contrastThreshold': self.contrastThreshold,
            'edgeThreshold': self.edgeThreshold,
            'sigma': self.sigma
        }