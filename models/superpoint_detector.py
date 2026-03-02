"""
SuperPoint detector implementation
"""
import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Tuple, List, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SuperPointNet(nn.Module):
    """SuperPoint neural network architecture"""
    
    def __init__(self):
        super(SuperPointNet, self).__init__()
        
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256
        
        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        
        # Detector Head
        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
        
        # Descriptor Head
        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(c5, 256, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        """Forward pass"""
        # Shared Encoder
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        
        # Detector Head
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)
        
        # Descriptor Head
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        
        # Normalize descriptors
        dn = torch.norm(desc, p=2, dim=1, keepdim=True)
        desc = desc.div(dn + 1e-8)
        
        return semi, desc


class SuperPointDetector:
    """SuperPoint feature detector and descriptor"""
    
    def __init__(self,
                 model_path: Optional[str] = None,
                 nms_dist: int = 4,
                 conf_thresh: float = 0.015,
                 nn_thresh: float = 0.7,
                 cuda: bool = True):
        """Initialize SuperPoint detector"""
        self.nms_dist = nms_dist
        self.conf_thresh = conf_thresh
        self.nn_thresh = nn_thresh
        self.cell_size = 8
        
        self.cuda = cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.cuda else 'cpu')
        
        self.net = SuperPointNet()
        self.net.to(self.device)
        
        if model_path and Path(model_path).exists():
            logger.info(f"Loading SuperPoint model from {model_path}")
            self.net.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            logger.warning("No model weights loaded. Using random initialization.")
        
        self.net.eval()
        
        logger.info(f"Initialized SuperPoint detector on {self.device}")
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for network input"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        image = image.astype(np.float32) / 255.0
        
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        
        return image_tensor
    
    def _nms(self, prob_map: np.ndarray, dist: int) -> np.ndarray:
        """Apply non-maximum suppression"""
        pool = nn.MaxPool2d(kernel_size=dist*2+1, stride=1, padding=dist)
        prob_tensor = torch.from_numpy(prob_map).unsqueeze(0).unsqueeze(0)
        
        if self.cuda:
            prob_tensor = prob_tensor.cuda()
        
        max_pool = pool(prob_tensor)
        nms_mask = (prob_tensor == max_pool).float().squeeze().cpu().numpy()
        
        return nms_mask
    
    def detectAndCompute(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """Detect keypoints and compute descriptors"""
        if image is None:
            logger.error("Input image is None")
            return [], None
        
        h, w = image.shape[:2]
        
        image_tensor = self._preprocess_image(image)
        
        with torch.no_grad():
            semi, desc = self.net(image_tensor)
        
        semi = semi.squeeze(0).cpu().numpy()
        
        nodust = semi[:-1, :, :]
        heatmap = np.exp(nodust)
        heatmap = heatmap / (np.sum(heatmap, axis=0, keepdims=True) + 1e-8)
        
        heatmap = heatmap.transpose(1, 2, 0)
        heatmap = heatmap.reshape(h // self.cell_size, w // self.cell_size, 
                                  self.cell_size, self.cell_size)
        heatmap = heatmap.transpose(0, 2, 1, 3)
        heatmap = heatmap.reshape(h, w)
        
        nms_mask = self._nms(heatmap, self.nms_dist)
        
        xs, ys = np.where((heatmap > self.conf_thresh) & (nms_mask > 0))
        pts = np.stack([ys, xs], axis=1)
        
        scores = heatmap[xs, ys]
        
        desc = desc.squeeze(0).cpu().numpy()
        desc = desc.transpose(1, 2, 0)
        
        desc_pts = []
        valid_pts = []
        valid_scores = []
        
        for i, (x, y) in enumerate(pts):
            desc_x = int(x / self.cell_size)
            desc_y = int(y / self.cell_size)
            
            if 0 <= desc_x < desc.shape[1] and 0 <= desc_y < desc.shape[0]:
                desc_pts.append(desc[desc_y, desc_x])
                valid_pts.append([x, y])
                valid_scores.append(scores[i])
        
        if len(valid_pts) == 0:
            logger.warning("No valid keypoints detected")
            return [], None
        
        keypoints = []
        for pt, score in zip(valid_pts, valid_scores):
            kp = cv2.KeyPoint(
                x=float(pt[0]), 
                y=float(pt[1]), 
                size=float(self.cell_size)
            )
            kp.response = float(score)
            keypoints.append(kp)
        
        descriptors = np.array(desc_pts, dtype=np.float32)
        
        logger.info(f"SuperPoint detected {len(keypoints)} keypoints")
        
        return keypoints, descriptors
    
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
            'method': 'SuperPoint',
            'nms_dist': self.nms_dist,
            'conf_thresh': self.conf_thresh,
            'nn_thresh': self.nn_thresh,
            'cuda': self.cuda
        }