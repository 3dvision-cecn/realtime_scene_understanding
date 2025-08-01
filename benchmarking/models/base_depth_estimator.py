
import numpy as np


class BaseDepthEstimator:
    """ Base class for depth estimators.
    This class provides a common interface for depth estimation methods.
    """

    def __init__(self, device, model_path: str | None = None , config_path: str = None):
        self.model_path = model_path
        self.config_path = config_path
        self.device = device


    def process_image(self, image: np.ndarray):
        """ Process the input image and return the depth map.
        Args:
            image (np.ndarray): Input image in RGB format.
        Returns:
            np.ndarray: Depth map corresponding to the input image.
            dict: Intrinsics dictionary containing camera parameters.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")