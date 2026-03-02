"""
SURF (Speeded-Up Robust Features) detector implementation
"""
import cv2
import numpy as np
from typing import Tuple, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SURFDetector:
    """SURF feature detector and descriptor"""
    
    def __init__(self,
                 hessianThreshold: float = 400,
                 nOctaves: int = 4,
                 nOctaveLayers: int = 3,
                 extended: bool = False,
                 upright: bool = False):
        """Initialize SURF detector"""
        self.hessianThreshold = hessianThreshold
        self.nOctaves = nOctaves
        self.nOctaveLayers = nOctaveLayers
        self.extended = extended
        self.upright = upright
        
        try:
            self.detector = cv2.xfeatures2d.SURF_create(
                hessianThreshold=hessianThreshold,
                nOctaves=nOctaves,
                nOctaveLayers=nOctaveLayers,
                extended=extended,
                upright=upright
            )
            logger.info(f"Initialized SURF detector")
        except AttributeError:
            logger.error("SURF is not available. Make sure opencv-contrib-python is installed.")
            raise
    
    def detect(self, image: np.ndarray) -> List[cv2.KeyPoint]:
        """Detect keypoints in an image"""
        if image is None:
            logger.error("Input image is None")
            return []
        
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        keypoints = self.detector.detect(image, None)
        logger.info(f"Detected {len(keypoints)} SURF keypoints")
        
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
            logger.info(f"SURF detected {len(keypoints)} keypoints")
        else:
            logger.warning("SURF detection failed or found no keypoints")
        
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
            'method': 'SURF',
            'hessianThreshold': self.hessianThreshold,
            'nOctaves': self.nOctaves,
            'nOctaveLayers': self.nOctaveLayers,
            'extended': self.extended,
            'upright': self.upright
        }