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
        self.image_processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf", device=device)
        self.model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf").to(self.device)


    def process_image(self, image: np.ndarray) -> np.ndarray:
        image = Image.fromarray(image.astype(np.uint8))
        inputs = self.image_processor(images=image, return_tensors="pt", device=self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        post_processed_output = self.image_processor.post_process_depth_estimation(
                outputs, target_sizes=[(image.height, image.width)])
        
        depth = post_processed_output[0]["predicted_depth"]

        return depth.cpu().numpy().squeeze()
        
