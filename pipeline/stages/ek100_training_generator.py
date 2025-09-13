import numpy as np
import torch
import os
from datetime import datetime
import h5py
from typing import List, Tuple, Optional, Dict

class EK100TrainingGenerator:
    """
    Training generator specifically designed for EK-100 dataset.
    Creates hypergraphs for sliding windows within narration segments.
    """

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        
        # Current processing state
        self.current_narration_id: Optional[str] = None
        self.current_narration_info: Optional[Dict] = None  # Store ground truth info
        self.current_sequence: List[Tuple] = []  # List of (image, graph) tuples
        self.current_start_frame: Optional[int] = None
        self.current_end_frame: Optional[int] = None
        
        # Base directory for storing graphs
        self.root_dir = cfg.training_generator.path
        os.makedirs(self.root_dir, exist_ok=True)

    def setup_narration(self, narration_id: str, narration_info: Dict):
        """Setup for processing a new narration"""
        self.current_narration_id = narration_id
        self.current_narration_info = narration_info  # Store the ground truth info
        self.current_sequence = []
        self.current_start_frame = None
        self.current_end_frame = None
        
        # Create directory for this narration
        narration_dir = os.path.join(self.root_dir, narration_id)
        os.makedirs(narration_dir, exist_ok=True)
        
        print(f"Setup EK100 training generator for narration: {narration_id}")
        print(f"Ground truth narration: {narration_info.get('narration', 'N/A')}")
        print(f"Ground truth verb: {narration_info.get('verb', 'N/A')} (class: {narration_info.get('verb_class', 'N/A')})")
        print(f"Ground truth noun: {narration_info.get('noun', 'N/A')} (class: {narration_info.get('noun_class', 'N/A')})")

    def start_window(self, start_frame: int, end_frame: int):
        """Start processing a new frame window"""
        self.current_start_frame = start_frame
        self.current_end_frame = end_frame
        self.current_sequence = []
        print(f"Starting window: frames {start_frame}-{end_frame}")

    def add_frame_to_window(self, image: np.ndarray, graph):
        """Add a frame and its graph to the current window"""
        if self.current_start_frame is None:
            print("Warning: No window started. Call start_window() first.")
            return
        
        self.current_sequence.append((image, graph))

    def finalize_window(self) -> bool:
        """
        Finalize the current window and save the hypergraph.
        Returns True if successful, False otherwise.
        """
        if (self.current_narration_id is None or 
            self.current_start_frame is None or 
            self.current_end_frame is None or
            len(self.current_sequence) == 0):
            print("Warning: Cannot finalize window - incomplete setup")
            return False

        # Generate filename: narration_id_start_frame_graph_end_frame_graph
        filename = f"{self.current_narration_id}_{self.current_start_frame}_graph_{self.current_end_frame}_graph.h5"
        sample_path = os.path.join(self.root_dir, self.current_narration_id, filename)
        
        print(f"Finalizing window: {filename} with {len(self.current_sequence)} frames")
        
        try:
            self._save_hypergraph(sample_path)
            print(f"✓ Saved hypergraph: {sample_path}")
            return True
        except Exception as e:
            print(f"✗ Failed to save hypergraph: {e}")
            return False

    def _save_hypergraph(self, sample_path: str):
        """Save the current sequence as an HDF5 hypergraph file"""
        frames = np.stack([img for img, _ in self.current_sequence], axis=0)  # (T, H, W, C)

        # Use ground truth information from CSV instead of AVION predictions
        if self.current_narration_info is not None:
            gt_verb = self.current_narration_info.get('verb', 'unknown')
            gt_verb_class = self.current_narration_info.get('verb_class', -1)
            gt_noun = self.current_narration_info.get('noun', 'unknown')
            gt_noun_class = self.current_narration_info.get('noun_class', -1)
            gt_narration = self.current_narration_info.get('narration', 'unknown')
            
            print(f"Ground truth: '{gt_narration}' (verb_class: {gt_verb_class}, noun_class: {gt_noun_class})")
        else:
            gt_verb = 'unknown'
            gt_verb_class = -1
            gt_noun = 'unknown'
            gt_noun_class = -1
            gt_narration = 'unknown'
            print("Warning: No ground truth narration info available")

        # Collect per-frame graph data
        frames_grp_data = []

        for img_idx, (_, graph) in enumerate(self.current_sequence):
            if getattr(graph["object"], "x", None) is None:
                print(f"Warning: empty graph at frame {img_idx}")
                frames_grp_data.append({
                    "features": np.empty((0, 0), dtype=np.float16),
                    "pos": np.empty((0, 3), dtype=np.float16),
                    "edges": np.empty((2, 0), dtype=np.int32),
                    "edge_lbl": np.empty((0,), dtype=np.float32),
                    "labels": np.empty((0,), dtype=np.int32)
                })
                continue

            # Extract graph features
            feat_np = graph["object"].x.cpu().numpy().astype(np.float16)
            pos_np = graph["object"].pos.cpu().numpy().astype(np.float16)
            labels_np = graph["object"].labels.cpu().numpy()

            # Extract edge information
            rel_src, rel_dst, rel_lbl = [], [], []
            for edge_type in graph.edge_types:
                ei = graph[edge_type].edge_index
                lbl = graph[edge_type].edge_attr

                rel_src.extend(ei[0].cpu().numpy())
                rel_dst.extend(ei[1].cpu().numpy())
                rel_lbl.extend(lbl.cpu().numpy())

            frames_grp_data.append({
                "features": feat_np,
                "pos": pos_np,
                "edges": np.vstack([rel_src, rel_dst]).astype(np.int32) if rel_src else np.empty((2, 0), dtype=np.int32),
                "edge_lbl": np.array(rel_lbl, dtype=np.float32) if rel_lbl else np.empty((0,), dtype=np.float32),
                "labels": labels_np.astype(np.int32)
            })

        # Write HDF5 file
        with h5py.File(sample_path, "w") as f:
            # Metadata
            f.attrs["narration_id"] = self.current_narration_id
            f.attrs["start_frame"] = self.current_start_frame
            f.attrs["end_frame"] = self.current_end_frame
            f.attrs["num_frames"] = len(self.current_sequence)
            
            # Ground truth labels from CSV
            f.attrs["gt_narration"] = gt_narration
            f.attrs["gt_verb"] = gt_verb
            f.attrs["gt_verb_class"] = gt_verb_class
            f.attrs["gt_noun"] = gt_noun
            f.attrs["gt_noun_class"] = gt_noun_class

            # Ground truth labels as datasets for easy access
            f.create_dataset(
                "ground_truth_verb_class",
                data=np.array([gt_verb_class], dtype=np.int32),
                compression="gzip",
                compression_opts=6
            )
            
            f.create_dataset(
                "ground_truth_noun_class", 
                data=np.array([gt_noun_class], dtype=np.int32),
                compression="gzip",
                compression_opts=6
            )

            # RGB frames
            f.create_dataset(
                "frames_rgb",
                data=frames,
                compression="gzip",
                compression_opts=6
            )

            # Per-frame graph data
            grp_frames = f.create_group("frames")
            for idx, frame_data in enumerate(frames_grp_data):
                g = grp_frames.create_group(f"{idx:04d}")

                for key, data in frame_data.items():
                    g.create_dataset(
                        key,
                        data=data,
                        compression="gzip",
                        compression_opts=6
                    )

    def get_saved_graphs_for_narration(self, narration_id: str) -> List[str]:
        """Get list of saved graph files for a specific narration"""
        narration_dir = os.path.join(self.root_dir, narration_id)
        if not os.path.exists(narration_dir):
            return []
        
        graphs = []
        for filename in os.listdir(narration_dir):
            if filename.endswith('.h5') and filename.startswith(narration_id):
                graphs.append(os.path.join(narration_dir, filename))
        
        return sorted(graphs)

    def cleanup_narration(self):
        """Clean up after processing a narration"""
        self.current_narration_id = None
        self.current_narration_info = None
        self.current_sequence = []
        self.current_start_frame = None
        self.current_end_frame = None
