"""
D2-Net (Detect and Describe Network) implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, List, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class D2Net(nn.Module):
    """D2-Net neural network based on VGG16 architecture"""
    
    def __init__(self, use_relu=True):
        super(D2Net, self).__init__()
        
        self.use_relu = use_relu
        
        self.conv1_1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        """Forward pass"""
        # Block 1
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool(x)
        
        # Block 2
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool(x)
        
        # Block 3
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool(x)
        
        # Block 4
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        
        if self.use_relu:
            x = F.relu(x)
        
        return x


class D2NetDetector:
    """D2-Net feature detector and descriptor"""
    
    def __init__(self,
                 model_path: Optional[str] = None,
                 use_relu: bool = True,
                 multiscale: bool = False,
                 max_edge: int = 1600,
                 max_sum_edges: int = 2800,
                 cuda: bool = True):
        """Initialize D2-Net detector"""
        self.use_relu = use_relu
        self.multiscale = multiscale
        self.max_edge = max_edge
        self.max_sum_edges = max_sum_edges
        
        self.cuda = cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.cuda else 'cpu')
        
        self.net = D2Net(use_relu=use_relu)
        self.net.to(self.device)
        
        '''if model_path and Path(model_path).exists():
            logger.info(f"Loading D2-Net model from {model_path}")
            self.net.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            logger.warning("No model weights loaded. Using random initialization.")'''
        if model_path and Path(model_path).exists():
            logger.info(f"Loading D2-Net model from {model_path}")
            try:
                # Load checkpoint
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'model' in checkpoint:
                        # Checkpoint contains 'model' key
                        state_dict = checkpoint['model']
                    elif 'state_dict' in checkpoint:
                        # Checkpoint contains 'state_dict' key
                        state_dict = checkpoint['state_dict']
                    else:
                        # Assume the dict itself is the state_dict
                        state_dict = checkpoint
                else:
                    # Direct state dict
                    state_dict = checkpoint
                
                # Remove 'module.' prefix if present (from DataParallel)
                new_state_dict = {}
                for k, v in state_dict.items():
                    name = k.replace('module.', '') if k.startswith('module.') else k
                    new_state_dict[name] = v
                
                # Load the state dict
                self.net.load_state_dict(new_state_dict, strict=False)
                logger.info("✓ D2-Net model loaded successfully")
                
            except Exception as e:
                logger.warning(f"Failed to load D2-Net weights: {str(e)}")
                logger.warning("Continuing with random initialization")
        else:
            logger.warning("No model weights loaded. Using random initialization.")
        ###########
        
        self.net.eval()
        
        logger.info(f"Initialized D2-Net detector on {self.device}")
    
    def _preprocess_image(self, image: np.ndarray) -> Tuple[torch.Tensor, float]:
        """Preprocess image for network input"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        h, w = image.shape
        
        resize_factor = 1.0
        if max(h, w) > self.max_edge:
            resize_factor = self.max_edge / max(h, w)
        if h + w > self.max_sum_edges:
            resize_factor = min(resize_factor, self.max_sum_edges / (h + w))
        
        if resize_factor < 1.0:
            new_h, new_w = int(h * resize_factor), int(w * resize_factor)
            image = cv2.resize(image, (new_w, new_h))
        
        image = image.astype(np.float32) / 255.0
        
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        
        return image_tensor, resize_factor
    
    def _detect_features(self, feature_map: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detect features from feature map"""
        feature_map = feature_map.squeeze(0)
        C, H, W = feature_map.shape
        
        # Compute detection scores
        channel_max = torch.max(feature_map, dim=0, keepdim=True)[0]
        
        alpha = F.softmax(feature_map / 0.1, dim=0)
        
        feature_map_unfold = F.unfold(feature_map.unsqueeze(0), kernel_size=3, padding=1)
        feature_map_unfold = feature_map_unfold.view(C, 9, H, W)
        spatial_max = torch.max(feature_map_unfold, dim=1)[0]
        
        beta = torch.where(
            feature_map == spatial_max,
            torch.ones_like(feature_map),
            torch.zeros_like(feature_map)
        )
        
        scores = (alpha * beta).max(dim=0)[0]
        
        scores_np = scores.cpu().numpy()
        threshold = 0.001
        
        ys, xs = np.where(scores_np > threshold)
        
        if len(xs) == 0:
            return np.array([]), None, None
        
        #########
        # Limit number of keypoints to prevent memory issues
        max_keypoints = 2000
        if len(xs) > max_keypoints:
            # Keep top keypoints by score
            top_indices = np.argsort(scores_np[ys, xs])[-max_keypoints:]
            xs = xs[top_indices]
            ys = ys[top_indices]
        #########
        
        keypoints = np.stack([xs, ys], axis=1)
        
        descriptors = feature_map[:, ys, xs].t().cpu().numpy()
        descriptors = descriptors / (np.linalg.norm(descriptors, axis=1, keepdims=True) + 1e-8)
        
        kp_scores = scores_np[ys, xs]
        
        return keypoints, descriptors, kp_scores
    
    def detectAndCompute(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """Detect keypoints and compute descriptors"""
        if image is None:
            logger.error("Input image is None")
            return [], None
        
        image_tensor, resize_factor = self._preprocess_image(image)
        
        with torch.no_grad():
            feature_map = self.net(image_tensor)
        
        keypoints_np, descriptors, scores = self._detect_features(feature_map)
        
        if len(keypoints_np) == 0:
            logger.warning("No keypoints detected")
            return [], None
        
        keypoints_np = keypoints_np / resize_factor
        
        keypoints = []
        for i, (x, y) in enumerate(keypoints_np):
            kp = cv2.KeyPoint(
                x=float(x * 8), 
                y=float(y * 8),
                size=8.0
            )
            kp.response = float(scores[i])
            keypoints.append(kp)
        
        logger.info(f"D2-Net detected {len(keypoints)} keypoints")
        
        return keypoints, descriptors.astype(np.float32)
    
    def detect(self, image: np.ndarray) -> List[cv2.KeyPoint]:
        """Detect keypoints only"""
        keypoints, _ = self.detectAndCompute(image)
        return keypoints
    
    @staticmethod
    def draw_keypoints(image: np.ndarray, 
                       keypoints: List[cv2.KeyPoint],
                       color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """Draw keypoints on image"""
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        output_image = cv2.drawKeypoints(
            image, keypoints, None, 
            color=color, 
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        return output_image
    
    def get_config(self) -> dict:
        """Get detector configuration"""
        return {
            'method': 'D2-Net',
            'use_relu': self.use_relu,
            'multiscale': self.multiscale,
            'max_edge': self.max_edge,
            'max_sum_edges': self.max_sum_edges,
            'cuda': self.cuda
        }