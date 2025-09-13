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
        super().__init__(device=device, model_path=model_path, config_path=config_path)
        self.image_processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf", device=device)
        self.model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf").to(self.device)
        print(f"Model loaded on device: {self.device}")

    def process_image(self, image: np.ndarray) -> np.ndarray:
        image = Image.fromarray(image.astype(np.uint8))
        inputs = self.image_processor(images=image, return_tensors="pt", device=self.device)

        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            with torch.no_grad():
                outputs = self.model(**inputs)

        post_processed_output = self.image_processor.post_process_depth_estimation(
                outputs, target_sizes=[(image.height, image.width)])
        
        depth = post_processed_output[0]["predicted_depth"]
        field_of_view = post_processed_output[0]["field_of_view"]  # in degrees (horizontal)
        focal_length = post_processed_output[0]["focal_length"]    # in mm or pixels depending on model output

        intrinsics_dict = {}
        intrinsics_dict["w"] = depth.shape[-1]
        intrinsics_dict["h"] = depth.shape[-2]

        # Compute focal length in pixels from FoV (assuming horizontal FoV)
        import math
        W = intrinsics_dict["w"]
        H = intrinsics_dict["h"]

        fx = W / (2 * math.tan(math.radians(field_of_view) / 2))
        fy = fx  # assuming square pixels; otherwise compute from vertical FoV if available

        # Principal point at image center
        cx = (W - 1) / 2
        cy = (H - 1) / 2

        intrinsics_dict["fx"] = fx
        intrinsics_dict["fy"] = fy
        intrinsics_dict["cx"] = cx
        intrinsics_dict["cy"] = cy

        return depth.cpu().to(torch.float32).numpy().squeeze(), None
        