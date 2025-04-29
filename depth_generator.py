from transformers import pipeline
from PIL import Image
import numpy as np



class DepthGenerator:
    def __init__(self, cfg):
        self.cfg = cfg

        self.pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")



    def estimate_depth(self, image):
        """
        Input:
            image rgb: np.ndarray, shape (H, W, 3), dtype=uint8
        Output:
            depth_map: np.ndarray, shape (H, W), dtype=float32
        """

        # Convert the image to a PIL Image
        pil_image = Image.fromarray(image)
        # Perform depth estimation
        depth_map = self.pipe(pil_image, min_depth=0.0, max_depth=4.0)["depth"]
        # Convert the depth map to a numpy array
        depth_map = np.array(depth_map)
        
        return depth_map