from .base_depth_estimator import BaseDepthEstimator
from video_depth_anything.video_depth_stream import VideoDepthAnything as VideoDepthAnythingStream
import torch
import numpy as np

class VideoDepthAnything(BaseDepthEstimator):
    """
    VideoDepthAnything is a class that extends BaseDepthEstimator to provide
    depth estimation capabilities for video data.
    """

    def __init__(self, device, model_path: str | None = 'conf/checkpoints/video_depth_anything/metric_video_depth_anything_vitl.pth' , config_path: str = None):
        super().__init__(device=device, model_path=model_path, config_path=config_path)

        model_config = {
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }

        self.model = VideoDepthAnythingStream(**model_config['vitl'])
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device), strict=True)
        self.model.to(self.device).eval()

        self.inputs_size = 518

    def process_image(self, image: np.ndarray):
        """ Process the input image and return the depth map.
        Args:
            image (np.ndarray): Input image in RGB format.
        Returns:
            np.ndarray: Depth map corresponding to the input image.
            dict: Intrinsics dictionary containing camera parameters.
        """

        depth =  self.model.infer_video_depth_one(image, input_size=self.inputs_size, device=self.device)
        return depth, None