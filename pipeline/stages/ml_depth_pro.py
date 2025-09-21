from PIL import Image

from .base_depth_estimator import BaseDepthEstimator
import numpy as np
import torch
import os

# Try to import the original depth_pro implementation
try:
    import depth_pro
    from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
    DEPTH_PRO_AVAILABLE = True
except ImportError:
    print("Warning: depth_pro package not available. Install Apple's ml-depth-pro for direct PyTorch loading.")
    DEPTH_PRO_AVAILABLE = False

from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation


class MLDepthEstimator(BaseDepthEstimator):
    """ A depth estimator using a pre-trained model.
    This class extends the BaseDepthEstimator to implement a specific depth estimation method.
    """

    def __init__(self, device, model_path = None, config_path = None):
        super().__init__(device=device, model_path=model_path, config_path=config_path)

        print(f"Initializing ML Depth Pro estimator for {device}")
        self.image_processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf", device=device)
        self.model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf").to(self.device)
        # Convert the weight to bfloat16
        self.model.to(torch.bfloat16)

        print("No model path provided or all loading methods failed")


    def process_image(self, image: np.ndarray) -> np.ndarray:
        if hasattr(self, 'transform') and DEPTH_PRO_AVAILABLE:
            # Use original depth_pro implementation
            pil_image = Image.fromarray(image.astype(np.uint8))
            
            # Transform the image for the model
            image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                prediction = self.model.infer(image_tensor)
            
            depth = prediction["depth"]  # Depth in meters
            depth = depth.squeeze().cpu().to(torch.float32).numpy()
            
            return depth, None
            
        elif hasattr(self, 'image_processor'):
            # Use HuggingFace implementation
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
        
        else:
            raise RuntimeError("No depth estimation model loaded. Check initialization.")
        