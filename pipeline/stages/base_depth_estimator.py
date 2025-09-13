import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Optional, Tuple

class BaseDepthEstimator(ABC):
    """
    Abstract base class for depth estimation models.
    """
    
    def __init__(self, device: str = "cuda", model_path: Optional[str] = None, config_path: Optional[str] = None):
        """
        Initialize the depth estimator.
        
        Args:
            device: Device to run the model on ('cuda' or 'cpu')
            model_path: Path to the model weights (optional)
            config_path: Path to the model configuration (optional)
        """
        self.device = device
        self.model_path = model_path
        self.config_path = config_path
        
    @abstractmethod
    def process_image(self, image: np.ndarray) -> Tuple[np.ndarray, Optional[dict]]:
        """
        Process an image to generate a depth map.
        
        Args:
            image: Input RGB image as numpy array (H, W, 3)
            
        Returns:
            Tuple of:
            - depth: Depth map as numpy array (H, W)
            - intrinsics: Optional camera intrinsics dictionary
        """
        pass
