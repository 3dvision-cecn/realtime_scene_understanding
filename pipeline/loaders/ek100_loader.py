import pandas as pd
import cv2
import numpy as np
import os
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import open3d as o3d
from ..stages.ml_depth_pro import MLDepthEstimator


@dataclass
class NarrationData:
    """Data class to hold narration information"""
    narration_id: str
    participant_id: str
    video_id: str
    start_frame: int
    stop_frame: int
    narration: str
    verb: str
    verb_class: int
    noun: str
    noun_class: int


class EK100Loader:
    """
    Loader for EPIC-KITCHENS-100 dataset that processes specific narration segments
    and generates sliding window hypergraphs.
    """
    
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        
        # Configuration parameters
        self.narration_csv_path = cfg.narration_csv_path
        self.video_base_path = cfg.video_base_path
        self.frames_per_hypergraph = cfg.frames_per_hypergraph
        self.sliding_window_reduction = cfg.sliding_window_reduction
        self.frame_sampling_interval = getattr(cfg, 'frame_sampling_interval', 1)
        
        # Initialize depth estimator if enabled
        self.use_depth_estimation = getattr(cfg, 'use_depth_estimation', True)
        if self.use_depth_estimation:
            # Get model path from config if available
            depth_model_path = getattr(cfg, 'depth_model_path', None)
            self.depth_estimator = MLDepthEstimator(device=device, model_path=depth_model_path)
            print("Initialized ML Depth Pro estimator for EK-100")
        else:
            self.depth_estimator = None
            print("Depth estimation disabled")
        
        # Load narration data
        self.narration_df = pd.read_csv(self.narration_csv_path)
        print(f"Loaded {len(self.narration_df)} narrations from {self.narration_csv_path}")
        
        # Current processing state
        self.current_narration: Optional[NarrationData] = None
        self.current_video_cap: Optional[cv2.VideoCapture] = None
        self.current_frame_idx: int = 0
        self.current_narration_frames: List[int] = []
        
    def get_narration_by_id(self, narration_id: str) -> Optional[NarrationData]:
        """Get narration data by narration_id"""
        row = self.narration_df[self.narration_df['narration_id'] == narration_id]
        if row.empty:
            return None
        
        row = row.iloc[0]
        return NarrationData(
            narration_id=row['narration_id'],
            participant_id=row['participant_id'],
            video_id=row['video_id'],
            start_frame=int(row['start_frame']),
            stop_frame=int(row['stop_frame']),
            narration=row['narration'],
            verb=row['verb'],
            verb_class=int(row['verb_class']),
            noun=row['noun'],
            noun_class=int(row['noun_class'])
        )
    
    def get_all_narration_ids(self) -> List[str]:
        """Get all available narration IDs"""
        return self.narration_df['narration_id'].tolist()
    
    def setup_narration(self, narration_id: str) -> bool:
        """
        Setup the loader for processing a specific narration.
        Returns True if setup successful, False otherwise.
        """
        self.current_narration = self.get_narration_by_id(narration_id)
        if self.current_narration is None:
            print(f"Narration ID {narration_id} not found")
            return False
        
        # Build video path
        video_filename = f"{self.current_narration.video_id}.MP4"
        video_path = os.path.join(
            self.video_base_path,
            self.current_narration.participant_id,
            video_filename
        )
        
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return False
        
        # Open video capture
        if self.current_video_cap is not None:
            self.current_video_cap.release()
        
        self.current_video_cap = cv2.VideoCapture(video_path)
        if not self.current_video_cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return False
        
        # Calculate frame indices for sliding windows
        self.current_narration_frames = self._calculate_sliding_windows(
            self.current_narration.start_frame,
            self.current_narration.stop_frame
        )
        
        self.current_frame_idx = 0
        print(f"Setup narration {narration_id}: {len(self.current_narration_frames)} frame windows")
        return True
    
    def _calculate_sliding_windows(self, start_frame: int, stop_frame: int) -> List[int]:
        """
        Calculate frame indices for sliding windows within the narration segment.
        Takes frame_sampling_interval into account - each window will span 
        frames_per_hypergraph * frame_sampling_interval actual frames but only 
        sample every frame_sampling_interval-th frame.
        Returns list of starting frame indices for each window.
        """
        windows = []
        current_start = start_frame
        
        # Calculate the actual frame span needed for one window when sampling
        actual_frames_per_window = (self.frames_per_hypergraph - 1) * self.frame_sampling_interval + 1
        # Calculate the actual step size when sampling
        actual_step_size = self.sliding_window_reduction * self.frame_sampling_interval
        
        while current_start + actual_frames_per_window <= stop_frame:
            windows.append(current_start)
            current_start += actual_step_size
        
        # Ensure we get the final window if there are remaining frames
        remaining_frames = stop_frame - current_start
        min_frames_needed = (self.frames_per_hypergraph // 2 - 1) * self.frame_sampling_interval + 1
        if current_start < stop_frame and remaining_frames >= min_frames_needed:
            windows.append(stop_frame - actual_frames_per_window)
        
        return windows
    
    def next_frame_window(self) -> Optional[Tuple[np.ndarray, int, int]]:
        """
        Get the next frame window for the current narration.
        Returns (frame, start_frame_idx, end_frame_idx) or None if no more windows.
        """
        if (self.current_narration is None or 
            self.current_video_cap is None or 
            self.current_frame_idx >= len(self.current_narration_frames)):
            return None
        
        start_frame = self.current_narration_frames[self.current_frame_idx]
        # Calculate end frame considering frame sampling interval
        end_frame = start_frame + (self.frames_per_hypergraph - 1) * self.frame_sampling_interval
        
        # Seek to start frame
        self.current_video_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Read the first frame of this window
        ret, frame = self.current_video_cap.read()
        if not ret:
            return None
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        self.current_frame_idx += 1
        return frame, start_frame, end_frame
    
    def get_window_frames(self, start_frame: int, num_frames: int) -> List[np.ndarray]:
        """
        Get a sequence of frames starting from start_frame.
        Respects frame_sampling_interval - only returns every Nth frame.
        Returns list of RGB frames.
        """
        if self.current_video_cap is None:
            return []
        
        frames = []
        
        for i in range(num_frames):
            # Calculate the actual frame index considering sampling interval
            frame_idx = start_frame + i * self.frame_sampling_interval
            
            # Seek to the specific frame
            self.current_video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.current_video_cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        return frames
    
    def next_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
        """
        Get the next frame with depth estimation for pipeline compatibility.
        Returns (frame_rgb, depth, pose, timestamp) where pose is None for EK-100.
        """
        if (self.current_narration is None or 
            self.current_video_cap is None or 
            self.current_frame_idx >= len(self.current_narration_frames)):
            return None, None, None, None
        
        # Get the current frame
        start_frame = self.current_narration_frames[self.current_frame_idx]
        
        # Seek to frame
        self.current_video_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, frame = self.current_video_cap.read()
        if not ret:
            return None, None, None, None
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Generate depth if enabled, otherwise use None
        if self.use_depth_estimation and self.depth_estimator is not None:
            depth, intrinsics = self.depth_estimator.process_image(frame_rgb)
        else:
            depth = None
            intrinsics = None
        
        # Store intrinsics for point cloud generation
        if intrinsics is not None:
            self.intrinsics_dict = intrinsics
        else:
            # Use default camera intrinsics if not provided by the model
            h, w = frame_rgb.shape[:2]
            self.intrinsics_dict = {
                "w": w,
                "h": h,
                "fx": w * 0.6,  # Default focal length estimation
                "fy": w * 0.6,
                "cx": w / 2.0,
                "cy": h / 2.0
            }
        
        # Calculate timestamp (frame index / 30 fps assumed)
        timestamp = start_frame / 30.0
        
        # No pose information for EK-100, use identity matrix
        pose = np.eye(4, dtype=np.float32)
        
        self.current_frame_idx += 1
        
        return frame_rgb, depth, pose, timestamp
    
    def get_intrinsics(self) -> Dict:
        """Get camera intrinsics dictionary"""
        if hasattr(self, 'intrinsics_dict'):
            return self.intrinsics_dict
        else:
            # Return default intrinsics
            return {
                "w": 1920,
                "h": 1080,
                "fx": 1152,
                "fy": 1152,
                "cx": 960,
                "cy": 540
            }
    
    def generate_pixel_indexed_pcd(self, color: np.ndarray, depth: np.ndarray, cam2world: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate a pixel-indexed point cloud from color and depth images.
        Returns array of shape (H, W, 6) with [x, y, z, r, g, b] for each pixel.
        """
        if color.shape[:2] != depth.shape[:2]:
            print(f"Warning: color shape {color.shape} does not match depth shape {depth.shape}.")
            return None
        
        h, w = depth.shape
        intrinsics = self.get_intrinsics()
        
        # Create coordinate grids
        u = np.arange(w)
        v = np.arange(h)
        uu, vv = np.meshgrid(u, v)
        
        # Unproject to 3D camera coordinates
        fx, fy = intrinsics["fx"], intrinsics["fy"]
        cx, cy = intrinsics["cx"], intrinsics["cy"]
        
        X = (uu - cx) * depth / fx
        Y = (vv - cy) * depth / fy
        Z = depth
        points_cam = np.stack((X, Y, Z), axis=-1)  # (H,W,3)
        
        # Convert to homogeneous coordinates
        ones = np.ones((h, w, 1), dtype=points_cam.dtype)
        points_cam_hom = np.concatenate([points_cam, ones], axis=-1)  # (H,W,4)
        
        # Reshape and transform points to world coordinates
        points_flat = points_cam_hom.reshape(-1, 4).T  # (4, H*W)
        points_world_flat = (cam2world @ points_flat).T  # (H*W,4)
        points_world = points_world_flat[:, :3].reshape(h, w, 3)
        
        # Combine world coordinates with color to form a pixel-indexed point cloud (H, W, 6)
        pixel_indexed_pcd = np.concatenate([points_world, color.astype(np.float32)], axis=-1)
        
        return pixel_indexed_pcd
    
    def generate_pcd(self, color: np.ndarray, depth: np.ndarray, cam2world: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a traditional point cloud from color and depth images.
        Returns (points, colors) arrays.
        """
        intrinsics = self.get_intrinsics()
        
        # Create an Open3D RGBD image
        intr = o3d.camera.PinholeCameraIntrinsic(
            intrinsics["w"], intrinsics["h"],
            intrinsics["fx"], intrinsics["fy"],
            intrinsics["cx"], intrinsics["cy"],
        )
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color),
            o3d.geometry.Image(depth * 1000),  # convert depth to mm
            convert_rgb_to_intensity=False,
        )
        
        # Backproject to a point cloud and transform into world coords
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)
        
        # Apply flip transform (standard computer vision convention)
        flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        pcd.transform(flip_transform)
        
        # Transform the point cloud into world coordinates
        pcd.transform(cam2world)
        
        # Extract numpy arrays
        pts = np.asarray(pcd.points)
        cols = np.asarray(pcd.colors)
        
        return pts, cols
    
    def get_current_narration_info(self) -> Optional[Dict]:
        """Get information about the current narration"""
        if self.current_narration is None:
            return None
        
        return {
            'narration_id': self.current_narration.narration_id,
            'participant_id': self.current_narration.participant_id,
            'video_id': self.current_narration.video_id,
            'narration': self.current_narration.narration,
            'verb': self.current_narration.verb,
            'verb_class': self.current_narration.verb_class,
            'noun': self.current_narration.noun,
            'noun_class': self.current_narration.noun_class,
            'start_frame': self.current_narration.start_frame,
            'stop_frame': self.current_narration.stop_frame,
            'total_windows': len(self.current_narration_frames)
        }
    
    def cleanup(self):
        """Release video capture resources"""
        if hasattr(self, 'current_video_cap') and self.current_video_cap is not None:
            self.current_video_cap.release()
            self.current_video_cap = None
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.cleanup()
