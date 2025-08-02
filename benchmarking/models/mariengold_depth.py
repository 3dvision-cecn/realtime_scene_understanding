from PIL import Image
import numpy as np
import torch
import diffusers
from .base_depth_estimator import BaseDepthEstimator


class MarigoldDepthEstimator(BaseDepthEstimator):
    """A depth estimator using Marigold diffusion-based depth prediction.
    This class extends the BaseDepthEstimator to implement Marigold depth estimation.
    """

    def __init__(self, device, model_path="prs-eth/marigold-depth-v1-1", config_path=None):
        super().__init__(device, model_path, config_path)
        
        # Load the Marigold depth pipeline
        # Use fp16 variant for faster inference if CUDA is available
        if self.device == "cuda":
            self.pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
                self.model_path, 
                variant="fp16", 
                torch_dtype=torch.float16
            ).to(self.device)
        else:
            self.pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
                self.model_path
            ).to(self.device)
        
        # Configure for faster inference
        self.pipe.set_progress_bar_config(disable=True)

    def process_image(self, image: np.ndarray) -> tuple[np.ndarray, None]:
        """Process the input image and return the depth map.
        
        Args:
            image (np.ndarray): Input image in RGB format with shape (H, W, 3).
            
        Returns:
            np.ndarray: Depth map corresponding to the input image with shape (H, W).
        """
        # Convert numpy array to PIL Image
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(image)
        
        # Generate depth prediction
        # Using 1 inference step for speed, can be increased for quality
        with torch.no_grad():
            depth_result = self.pipe(
                pil_image,
                num_inference_steps=1,
                ensemble_size=1,  # Single prediction for speed
                match_input_resolution=True
            )
        
        # Extract depth prediction as numpy array
        depth_prediction = depth_result.prediction
        
        # Convert from tensor to numpy array if needed
        if isinstance(depth_prediction, torch.Tensor):
            depth_prediction = depth_prediction.cpu().numpy()
        
        # Remove batch dimension if present and ensure 2D output
        if depth_prediction.ndim == 4:  # (1, 1, H, W)
            depth_prediction = depth_prediction.squeeze()
        elif depth_prediction.ndim == 3:  # (1, H, W) or (H, W, 1)
            depth_prediction = depth_prediction.squeeze()
        
        return depth_prediction, None


class MarigoldDepthEstimatorHighQuality(BaseDepthEstimator):
    """A high-quality depth estimator using Marigold with ensembling.
    This version prioritizes quality over speed with multiple inference steps and ensembling.
    """

    def __init__(self, device, model_path="prs-eth/marigold-depth-v1-1", config_path=None):
        super().__init__(device, model_path, config_path)
        
        # Load the Marigold depth pipeline
        if self.device == "cuda":
            self.pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
                self.model_path, 
                variant="fp16", 
                torch_dtype=torch.float16
            ).to(self.device)
        else:
            self.pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
                self.model_path
            ).to(self.device)
        
        self.pipe.set_progress_bar_config(disable=True)

    def process_image(self, image: np.ndarray) -> tuple[np.ndarray, None]:
        """Process the input image and return high-quality depth map.
        
        Args:
            image (np.ndarray): Input image in RGB format with shape (H, W, 3).
            
        Returns:
            np.ndarray: High-quality depth map corresponding to the input image with shape (H, W).
        """
        # Convert numpy array to PIL Image
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(image)
        
        # Generate high-quality depth prediction with ensembling
        with torch.no_grad():
            depth_result = self.pipe(
                pil_image,
                num_inference_steps=4,   # More steps for better quality
                ensemble_size=5,         # Ensemble for better stability
                match_input_resolution=True
            )
        
        # Extract depth prediction as numpy array
        depth_prediction = depth_result.prediction
        
        # Convert from tensor to numpy array if needed
        if isinstance(depth_prediction, torch.Tensor):
            depth_prediction = depth_prediction.cpu().numpy()
        
        # Remove batch dimension if present and ensure 2D output
        if depth_prediction.ndim == 4:  # (1, 1, H, W)
            depth_prediction = depth_prediction.squeeze()
        elif depth_prediction.ndim == 3:  # (1, H, W) or (H, W, 1)
            depth_prediction = depth_prediction.squeeze()
        
        return depth_prediction, None