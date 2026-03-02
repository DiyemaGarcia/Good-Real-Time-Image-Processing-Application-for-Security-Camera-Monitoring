"""
Visualization utilities for feature detection results
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResultsPlotter:
    """Handle visualization of feature detection and matching results"""
    
    def __init__(self, figure_size: Tuple[int, int] = (15, 10), dpi: int = 150):
        """Initialize ResultsPlotter"""
        self.figure_size = figure_size
        self.dpi = dpi
        
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = dpi
    
    def plot_keypoints(self, 
                      image: np.ndarray,
                      keypoints: List[cv2.KeyPoint],
                      title: str = "Detected Keypoints",
                      save_path: Optional[str] = None) -> plt.Figure:
        """Plot detected keypoints on image"""
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        if len(image.shape) == 2:
            display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            display_image = image.copy()
        
        if keypoints:
            for kp in keypoints:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                cv2.circle(display_image, (x, y), 3, (0, 255, 0), -1)
                if hasattr(kp, 'size') and kp.size > 0:
                    radius = int(kp.size)
                    cv2.circle(display_image, (x, y), radius, (0, 255, 0), 1)
        
        ax.imshow(display_image)
        ax.set_title(f"{title}\n{len(keypoints)} keypoints detected", fontsize=14)
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved keypoints plot to {save_path}")
        
        return fig
    
    def plot_matches(self,
                    image1: np.ndarray,
                    image2: np.ndarray,
                    keypoints1: List[cv2.KeyPoint],
                    keypoints2: List[cv2.KeyPoint],
                    matches: List[cv2.DMatch],
                    title: str = "Feature Matches",
                    max_matches: int = 100,
                    save_path: Optional[str] = None) -> plt.Figure:
        """Plot feature matches between two images"""
        if len(image1.shape) == 2:
            img1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB)
        else:
            img1 = image1.copy()
        
        if len(image2.shape) == 2:
            img2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2RGB)
        else:
            img2 = image2.copy()
        
        if len(matches) > max_matches:
            matches_sorted = sorted(matches, key=lambda x: x.distance)
            matches_to_draw = matches_sorted[:max_matches]
        else:
            matches_to_draw = matches
        
        match_image = cv2.drawMatches(
            img1, keypoints1,
            img2, keypoints2,
            matches_to_draw, None,
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        fig, ax = plt.subplots(figsize=(self.figure_size[0], self.figure_size[1] // 2))
        ax.imshow(match_image)
        ax.set_title(f"{title}\n{len(matches)} matches found", fontsize=14)
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved matches plot to {save_path}")
        
        return fig
    
    def plot_comparison_table(self,
                             results: Dict[str, Dict],
                             save_path: Optional[str] = None) -> plt.Figure:
        """Plot comparison table of different methods"""
        methods = list(results.keys())
        metrics = ['num_features1', 'num_features2', 'num_matches', 
                  'matching_efficiency', 'avg_matching_distance',
                  'spatial_entropy1', 'spatial_entropy2']
        
        data = []
        for method in methods:
            row = [results[method].get(metric, 0) for metric in metrics]
            data.append(row)
        
        fig, ax = plt.subplots(figsize=(14, len(methods) * 0.6 + 2))
        
        col_labels = ['Features\n(Image 1)', 'Features\n(Image 2)', 'Matches',
                     'Matching\nEfficiency (%)', 'Avg Match\nDistance',
                     'Spatial\nEntropy 1', 'Spatial\nEntropy 2']
        
        formatted_data = []
        for row in data:
            formatted_row = [
                f"{int(row[0])}", f"{int(row[1])}", f"{int(row[2])}",
                f"{row[3]:.2f}%", f"{row[4]:.4f}",
                f"{row[5]:.4f}", f"{row[6]:.4f}"
            ]
            formatted_data.append(formatted_row)
        
        table = ax.table(cellText=formatted_data,
                        rowLabels=methods,
                        colLabels=col_labels,
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        for i in range(len(col_labels)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for i in range(len(methods)):
            table[(i + 1, -1)].set_facecolor('#2196F3')
            table[(i + 1, -1)].set_text_props(weight='bold', color='white')
        
        ax.axis('off')
        ax.set_title('Performance Comparison Summary', fontsize=16, weight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved comparison table to {save_path}")
        
        return fig
    
    def plot_metrics_bar_chart(self,
                               results: Dict[str, Dict],
                               metric: str,
                               title: str = None,
                               save_path: Optional[str] = None) -> plt.Figure:
        """Plot bar chart comparing a specific metric across methods"""
        methods = list(results.keys())
        values = [results[method].get(metric, 0) for method in methods]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(methods, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=10)
        
        if title is None:
            title = f'{metric.replace("_", " ").title()} Comparison'
        
        ax.set_title(title, fontsize=14, weight='bold')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_xlabel('Method', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved bar chart to {save_path}")
        
        return fig
    
    def plot_spatial_distribution(self,
                                  image: np.ndarray,
                                  keypoints: List[cv2.KeyPoint],
                                  title: str = "Keypoint Spatial Distribution",
                                  bins: int = 8,
                                  save_path: Optional[str] = None) -> plt.Figure:
        """Plot spatial distribution heatmap of keypoints"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figure_size[0], self.figure_size[1] // 2))
        
        if len(image.shape) == 2:
            display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            display_image = image.copy()
        
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            cv2.circle(display_image, (x, y), 2, (0, 255, 0), -1)
        
        ax1.imshow(display_image)
        ax1.set_title('Keypoints on Image')
        ax1.axis('off')
        
        if keypoints:
            points = np.array([kp.pt for kp in keypoints])
            x_coords = points[:, 0]
            y_coords = points[:, 1]
            
            h, w = image.shape[:2]
            heatmap, xedges, yedges = np.histogram2d(
                y_coords, x_coords,
                bins=bins,
                range=[[0, h], [0, w]]
            )
            
            im = ax2.imshow(heatmap, cmap='hot', interpolation='nearest', aspect='auto')
            ax2.set_title('Spatial Distribution Heatmap')
            ax2.set_xlabel('X bins')
            ax2.set_ylabel('Y bins')
            plt.colorbar(im, ax=ax2, label='Keypoint Count')
        else:
            ax2.text(0.5, 0.5, 'No keypoints detected',
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Spatial Distribution Heatmap')
        
        fig.suptitle(title, fontsize=14, weight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved spatial distribution plot to {save_path}")
        
        return fig
    
    @staticmethod
    def close_all():
        """Close all matplotlib figures"""
        plt.close('all')