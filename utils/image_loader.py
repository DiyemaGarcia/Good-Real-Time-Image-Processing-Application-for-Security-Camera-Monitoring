"""
Image loading and preprocessing utilities
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageLoader:
    """Handle image loading and preprocessing"""
    
    def __init__(self, target_size: Tuple[int, int] = (512, 512), grayscale: bool = True):
        """
        Initialize ImageLoader
        
        Args:
            target_size: Target size for resizing images (width, height)
            grayscale: Whether to convert images to grayscale
        """
        self.target_size = target_size
        self.grayscale = grayscale
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load a single image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Loaded image as numpy array or None if loading fails
        """
        try:
            if self.grayscale:
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            else:
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
            
            if self.target_size is not None:
                image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
            
            logger.info(f"Successfully loaded image: {image_path}, shape: {image.shape}")
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            return None
    
    def load_image_pair(self, image_path1: str, image_path2: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load a pair of images
        
        Args:
            image_path1: Path to first image
            image_path2: Path to second image
            
        Returns:
            Tuple of (image1, image2)
        """
        image1 = self.load_image(image_path1)
        image2 = self.load_image(image_path2)
        
        return image1, image2
    
    def load_dataset(self, dataset_path: str) -> List[Tuple[np.ndarray, np.ndarray, str]]:
        """
        Load all image pairs from a dataset directory
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            List of tuples (image1, image2, pair_name)
        """
        dataset_path = Path(dataset_path)
        image_pairs = []
        
        image_files = sorted(list(dataset_path.glob("*.jpg")) + 
                           list(dataset_path.glob("*.png")) + 
                           list(dataset_path.glob("*.jpeg")))
        
        for i in range(0, len(image_files) - 1, 2):
            img1_path = image_files[i]
            img2_path = image_files[i + 1]
            
            img1 = self.load_image(img1_path)
            img2 = self.load_image(img2_path)
            
            if img1 is not None and img2 is not None:
                pair_name = f"{img1_path.stem}_{img2_path.stem}"
                image_pairs.append((img1, img2, pair_name))
                logger.info(f"Loaded image pair: {pair_name}")
        
        logger.info(f"Loaded {len(image_pairs)} image pairs from {dataset_path}")
        return image_pairs
    
    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range"""
        return image.astype(np.float32) / 255.0
    
    @staticmethod
    def denormalize_image(image: np.ndarray) -> np.ndarray:
        """Denormalize image from [0, 1] to [0, 255] range"""
        return (image * 255.0).astype(np.uint8)
    
    @staticmethod
    def convert_to_rgb(image: np.ndarray) -> np.ndarray:
        """Convert grayscale image to RGB"""
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image


def create_image_pyramid(image: np.ndarray, num_levels: int = 4) -> List[np.ndarray]:
    """Create image pyramid for multi-scale processing"""
    pyramid = [image]
    current_image = image.copy()
    
    for i in range(1, num_levels):
        current_image = cv2.pyrDown(current_image)
        pyramid.append(current_image)
    
    return pyramid


def compute_integral_image(image: np.ndarray) -> np.ndarray:
    """Compute integral image (used in SURF)"""
    return cv2.integral(image)