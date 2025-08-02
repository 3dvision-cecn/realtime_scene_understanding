import os
import math
import cv2
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from .base_depth_estimator import BaseDepthEstimator


class Metric3DONNXEstimator(BaseDepthEstimator):
    """ONNX-based Metric3D depth estimator using Hugging Face models.
    
    This class implements depth estimation using the ONNX version of Metric3D models
    from Hugging Face, completely bypassing the mmcv configuration issues.
    
    Available models:
    - metric3d-vit-small: Fast and efficient
    - metric3d-vit-large: Balanced performance 
    - metric3d-vit-giant2: Best quality
    """

    # Model configurations
    MODEL_CONFIGS = {
        'small': {
            'repo_id': 'onnx-community/metric3d-vit-small',
            'filename': 'onnx/model.onnx',
            'input_size': (616, 1064),
            'canonical_focal': 1000.0
        },
        'large': {
            'repo_id': 'onnx-community/metric3d-vit-large', 
            'filename': 'onnx/model.onnx',
            'input_size': (616, 1064),
            'canonical_focal': 1000.0
        },
        'giant': {
            'repo_id': 'onnx-community/metric3d-vit-giant2',
            'filename': 'onnx/model.onnx', 
            'input_size': (616, 1064),
            'canonical_focal': 1000.0
        }
    }

    
    def __init__(self, device="cuda", model_path=None, config_path=None, model_variant='giant'):
        """
        Initialize the Metric3D ONNX estimator.
        
        Args:
            device: Device to run inference on ('cuda' or 'cpu')
            model_path: Not used (kept for interface compatibility)
            config_path: Not used (kept for interface compatibility)  
            model_variant: Which model to use ('small', 'large', 'giant')
        """
        super().__init__(device=device, model_path=model_path, config_path=config_path)
        
        self.model_variant = model_variant
        if model_variant not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model variant: {model_variant}. Available: {list(self.MODEL_CONFIGS.keys())}")
        
        self.config = self.MODEL_CONFIGS[model_variant]
        self.input_size = self.config['input_size']  # (H, W)
        self.canonical_focal = self.config['canonical_focal']
        
        # Download model from Hugging Face
        print(f"Loading Metric3D ONNX model: {model_variant}")
        try:
            model_path = hf_hub_download(
                repo_id=self.config['repo_id'],
                filename=self.config['filename'],
                cache_dir=None  # Use default cache
            )
            print(f"Downloaded model to: {model_path}")
        except Exception as e:
            print(f"Failed to download model: {e}")
            raise RuntimeError(f"Could not download Metric3D ONNX model: {e}")
        
        # Configure ONNX providers based on device
        if device == "cuda" and ort.get_available_providers() and "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            print("Using CUDA for ONNX inference")
        else:
            providers = ["CPUExecutionProvider"]
            if device == "cuda":
                print("CUDA requested but not available for ONNX, using CPU")
            self.device = "cpu"
        
        # Initialize ONNX Runtime session
        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
            print(f"ONNX session created with providers: {self.session.get_providers()}")
        except Exception as e:
            error_msg = str(e)
            print(f"Failed to create ONNX session: {e}")
            
            # Check for specific ONNX errors and provide helpful suggestions
            if "Type Error" in error_msg and "Add" in error_msg:
                print("\n" + "="*60)
                print("ONNX MODEL TYPE COMPATIBILITY ERROR DETECTED")
                print("="*60)
                print("This is a known issue with some Metric3D ONNX models.")
                print("The model has type mismatches (int64 vs float) in Add operations.")
                print("\nSuggested solutions:")
                print("1. Try a smaller model: --model Metric3DONNXEstimatorSmall")
                print("2. Try a different ONNX Runtime version: pip install onnxruntime-gpu==1.15.1")
                print("3. Use the working PyTorch model: --model Metric3DEstimator")
                print("4. Use alternative models: --model MLDepthEstimator")
                print("="*60)
            elif "CUDA" in error_msg:
                print("This appears to be a CUDA-related ONNX issue.")
                print("Try running with CPU: device='cpu'")
            
            raise RuntimeError(f"Could not initialize ONNX session: {e}")
        
        # Get model input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        
        print(f"Metric3D ONNX model loaded: {model_variant}")
        print(f"Model input shape: {input_shape}")
        print(f"Expected input size: {self.input_size}")

    def _preprocess_image(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Preprocess image for Metric3D ONNX model.
        
        Args:
            image: Input image in RGB format (H, W, C)
            
        Returns:
            Tuple of (preprocessed_tensor, preprocessing_info)
        """
        original_shape = image.shape[:2]  # (H, W)
        h, w = original_shape
        
        # Keep ratio resize (matching Metric3D preprocessing)
        target_h, target_w = self.input_size
        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Padding to target size
        pad_h = target_h - new_h
        pad_w = target_w - new_w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        
        # Metric3D padding values
        padding_color = [123.675, 116.28, 103.53]
        padded = cv2.copyMakeBorder(
            resized,
            pad_h_half, pad_h - pad_h_half,
            pad_w_half, pad_w - pad_w_half,
            cv2.BORDER_CONSTANT,
            value=padding_color
        )
        
        # Normalize (Metric3D normalization)
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        normalized = (padded.astype(np.float32) - mean) / std
        
        # Convert to CHW format and add batch dimension
        preprocessed = np.transpose(normalized, (2, 0, 1))  # HWC -> CHW
        preprocessed = np.expand_dims(preprocessed, axis=0)  # Add batch dimension
        preprocessed = preprocessed.astype(np.float32)
        
        # Store preprocessing info for post-processing
        preprocess_info = {
            'original_shape': original_shape,
            'scale': scale,
            'pad_info': [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half],
            'new_size': (new_h, new_w)
        }
        
        return preprocessed, preprocess_info

    def _postprocess_depth(self, depth_output: np.ndarray, preprocess_info: dict, focal_length: float) -> np.ndarray:
        """
        Postprocess depth output to original image size.
        
        Args:
            depth_output: Raw depth output from ONNX model
            preprocess_info: Information from preprocessing
            focal_length: Estimated focal length for metric conversion
            
        Returns:
            Processed depth map in metric scale
        """
        # Remove batch dimension if present
        if len(depth_output.shape) == 4:
            depth = depth_output[0, 0]  # (1, 1, H, W) -> (H, W)
        elif len(depth_output.shape) == 3:
            depth = depth_output[0]     # (1, H, W) -> (H, W)
        else:
            depth = depth_output        # Already (H, W)
        
        # Remove padding
        pad_info = preprocess_info['pad_info']
        depth_unpadded = depth[
            pad_info[0] : depth.shape[0] - pad_info[1],
            pad_info[2] : depth.shape[1] - pad_info[3]
        ]
        
        # Resize to original image size
        original_shape = preprocess_info['original_shape']
        depth_resized = cv2.resize(
            depth_unpadded, 
            (original_shape[1], original_shape[0]),  # (W, H) for cv2.resize
            interpolation=cv2.INTER_LINEAR
        )
        
        # Convert from canonical camera space to metric depth
        canonical_to_real_scale = focal_length / self.canonical_focal
        depth_metric = depth_resized * canonical_to_real_scale
        
        # Clamp to reasonable depth range
        depth_metric = np.clip(depth_metric, 0, 300)
        
        return depth_metric

    def _estimate_intrinsics(self, image_shape: tuple[int, int]) -> dict:
        """
        Estimate camera intrinsics for the given image shape.
        
        Args:
            image_shape: Image shape (H, W)
            
        Returns:
            Dictionary containing intrinsic parameters
        """
        h, w = image_shape
        
        # Estimate focal length (heuristic based on image size)
        # This is a reasonable assumption for many cameras
        focal_length_estimate = max(w, h) * 0.7
        
        intrinsics = {
            'fx': focal_length_estimate,
            'fy': focal_length_estimate,
            'cx': (w - 1) / 2.0,
            'cy': (h - 1) / 2.0,
            'w': w,
            'h': h
        }
        
        return intrinsics

    def process_image(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Process the input image and return the depth map and intrinsics.
        
        Args:
            image (np.ndarray): Input image in RGB format.
            
        Returns:
            tuple: (depth_map, intrinsics_dict)
                - depth_map: Depth map as numpy array
                - intrinsics_dict: Camera intrinsics dictionary
        """
        # Preprocess image
        preprocessed, preprocess_info = self._preprocess_image(image)
        
        # Estimate focal length for the original image
        original_shape = preprocess_info['original_shape']
        focal_length_estimate = max(original_shape) * 0.7
        
        # Run ONNX inference
        try:
            depth_output = self.session.run(
                [self.output_name], 
                {self.input_name: preprocessed}
            )[0]
        except Exception as e:
            print(f"ONNX inference failed: {e}")
            raise RuntimeError(f"ONNX inference error: {e}")
        
        # Postprocess depth
        depth_map = self._postprocess_depth(depth_output, preprocess_info, focal_length_estimate)
        
        # Estimate intrinsics
        intrinsics = self._estimate_intrinsics(original_shape)
        
        return depth_map, intrinsics


class Metric3DONNXEstimatorSmall(Metric3DONNXEstimator):
    """Metric3D ONNX estimator with ViT-Small model."""
    def __init__(self, device="cuda", model_path=None, config_path=None):
        super().__init__(device, model_path, config_path, model_variant='small')


class Metric3DONNXEstimatorLarge(Metric3DONNXEstimator):
    """Metric3D ONNX estimator with ViT-Large model."""
    def __init__(self, device="cuda", model_path=None, config_path=None):
        super().__init__(device, model_path, config_path, model_variant='large')


class Metric3DONNXEstimatorGiant(Metric3DONNXEstimator):
    """Metric3D ONNX estimator with ViT-Giant2 model."""
    def __init__(self, device="cuda", model_path=None, config_path=None):
        super().__init__(device, model_path, config_path, model_variant='giant')