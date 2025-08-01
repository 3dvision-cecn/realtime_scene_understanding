from PIL import Image

from .base_depth_estimator import BaseDepthEstimator
import numpy as np
import torch
from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation

class MLDepthEstimator(BaseDepthEstimator):
    """ A depth estimator using a pre-trained model.
    This class extends the BaseDepthEstimator to implement a specific depth estimation method.
    """

    def __init__(self, device, model_path = None, config_path = None):
        super().__init__(model_path, device, config_path)
        self.image_processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
        self.model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf").to(self.device)


    def process_image(self, image: np.ndarray) -> np.ndarray:
        return image
