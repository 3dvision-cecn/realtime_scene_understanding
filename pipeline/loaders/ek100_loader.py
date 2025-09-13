import pandas as pd
import cv2
import numpy as np
import os
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass


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
        if self.current_video_cap is not None:
            self.current_video_cap.release()
            self.current_video_cap = None
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.cleanup()
