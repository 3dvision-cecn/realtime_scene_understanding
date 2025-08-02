import os
import math
import cv2
import numpy as np
import onnxruntime as ort
from .base_depth_estimator import BaseDepthEstimator


class DepthAnythingONNXEstimator(BaseDepthEstimator):
    """ONNX-based Depth Anything depth estimator.
    
    This class implements depth estimation using the ONNX version of Depth Anything models.
    Supports ViT-S, ViT-B, and ViT-L variants.
    """

    def __init__(self, device="cuda", model_path=None, config_path=None):
        super().__init__(device=device, model_path=model_path, config_path=config_path)
        
        # Default model path if not provided
        if model_path is None:
            # Default to ViT-B model
            model_path = "depth_anything_v2_onnx/weights/depth_anything_vitb14.onnx"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"ONNX model not found at {model_path}. "
                f"Please download the models using: bash depth_anything_v2_onnx/weights/download.sh"
            )
        print(f"Device {device}")
        
        # Configure ONNX providers based on device
        if device == "cuda":
            providers = ["CUDAExecutionProvider"]#["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        print(providers)
        
        # Initialize ONNX Runtime session
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # Get input shape from the model
        input_shape = self.session.get_inputs()[0].shape
        # Handle dynamic shapes (strings) by using default values
        if isinstance(input_shape[2], int) and input_shape[2] != -1:
            self.input_height = input_shape[2]
        else:
            self.input_height = 518
        
        if isinstance(input_shape[3], int) and input_shape[3] != -1:
            self.input_width = input_shape[3]
        else:
            self.input_width = 518
        
        print(f"Depth Anything ONNX model loaded: {model_path}")
        print(f"Model input shape: {input_shape}")
        print(f"Using providers: {self.session.get_providers()}")

    def _preprocess_image(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        """Preprocess image for Depth Anything ONNX model.
        
        Args:
            image: Input image in RGB format (H, W, C)
            
        Returns:
            Tuple of (preprocessed_image, original_shape)
        """
        orig_shape = image.shape[:2]  # (H, W)
        
        # Convert to float and normalize to [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        # Resize image to model input size
        resized = cv2.resize(
            image, 
            (self.input_width, self.input_height), 
            interpolation=cv2.INTER_CUBIC
        )
        
        # Normalize using ImageNet statistics
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (resized - mean) / std
        
        # Convert to CHW format and add batch dimension
        preprocessed = np.transpose(normalized, (2, 0, 1))  # HWC -> CHW
        preprocessed = np.expand_dims(preprocessed, axis=0)  # Add batch dimension
        preprocessed = preprocessed.astype(np.float32)
        
        return preprocessed, orig_shape

    def _postprocess_depth(self, depth_output: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        """Postprocess depth output to target shape.
        
        Args:
            depth_output: Raw depth output from model
            target_shape: Target shape (H, W)
            
        Returns:
            Processed depth map
        """
        # Remove batch dimension and squeeze
        
        # Resize to target shape
        depth_resized = cv2.resize(depth_output, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
        
        return depth_resized

    def _compute_intrinsics(self, image_shape: tuple[int, int]) -> dict:
        """Compute reasonable intrinsics for the given image shape.
        
        Since Depth Anything doesn't provide intrinsics, we estimate them
        based on typical camera parameters.
        
        Args:
            image_shape: Image shape (H, W)
            
        Returns:
            Dictionary containing intrinsic parameters
        """
        height, width = image_shape
        
        # Assume a reasonable field of view (horizontal FOV of ~60 degrees)
        fov_horizontal = 60.0  # degrees
        
        # Compute focal length in pixels
        fx = width / (2 * math.tan(math.radians(fov_horizontal) / 2))
        fy = fx  # Assume square pixels
        
        # Principal point at image center
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        
        intrinsics = {
            "w": width,
            "h": height,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy
        }
        
        return intrinsics

    def process_image(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        """Process input image and return depth map and intrinsics.
        
        Args:
            image: Input RGB image as numpy array (H, W, C)
            
        Returns:
            Tuple of (depth_map, intrinsics_dict)
            - depth_map: Predicted depth map as numpy array
            - intrinsics_dict: Camera intrinsics dictionary (can be None)
        """
        # Preprocess image
        preprocessed_image, orig_shape = self._preprocess_image(image)
        
        # Run inference
        depth_output = self.session.run(None, {"image": preprocessed_image})[0]
        depth_output_squeezed = depth_output.squeeze(axis=0)  # Remove batch dimension

        # Postprocess depth
        depth_map = self._postprocess_depth(depth_output_squeezed, orig_shape)
        
        # Compute intrinsics
        intrinsics = self._compute_intrinsics(orig_shape)
        
        return depth_map, intrinsics

class DepthAnythingONNXEstimatorLarge(DepthAnythingONNXEstimator):
    """ViT-Large variant of Depth Anything ONNX estimator."""
    
    def __init__(self, device="cuda", model_path=None, config_path=None):
        if model_path is None:
            model_path = "conf/checkpoints/depthanythingv2/depth_anything_v2_vitl_indoor_dynamic.onnx"
        super().__init__(device=device, model_path=model_path, config_path=config_path)