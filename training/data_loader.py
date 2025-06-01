import os
import json
import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
import pandas as pd
import csv
import h5py

class GraphDataset(Dataset):
    def __init__(self, data_dir, embedder, metadata_csv, mapping_vn2act):

        self.data_dir = data_dir
        self.embedder = embedder
        self.mapping_vn2act = mapping_vn2act

        graph_file_count = 0
        self.filenames_list = []
        print(f"Loading graph dataset from {self.data_dir}")
        for dirpath, dirnames, filenames in os.walk(self.data_dir):
            for filename in filenames:
                if ".h5" in filename:
                    graph_file_count += 1
                    self.filenames_list.append(os.path.join(dirpath, filename))
        print(f"Found {graph_file_count} graph files in {self.data_dir}")





    def __len__(self):
        return len(self.filenames_list)

    def __getitem__(self, idx):
        # Get info for one sample
        print(f"Processing sample {idx}")
        graph_path = self.filenames_list[idx]

        # Load hf5 file
        with h5py.File(graph_path, 'r') as f:
            label =  f["logits_argmax"][()]
            print(f"Graph label: {label}")
            frames_grp = f["frames"]
            frames = []

            for frame_key in sorted(frames_grp.keys()):
                g = frames_grp[frame_key]
                frames.append(
                    {
                        "features":  g["features"][()],
                        "edges":     g["edge_index"][()],
                        "edge_lbl":  g["edge_labels"][()],
                    }
                )

            print(f"Loaded {len(frames)} frames from {graph_path}")


        # Create PyG graph
        data = HeteroData()
        data['object'].x = node_embeddings
        data['object', 'relation', 'object'].edge_index = relational_edge_index
        if relational_edges:
            data['object', 'relation', 'object'].edge_attr = relation_embeddings
        data['object', 'temporal', 'object'].edge_index = temporal_edge_index

        # Graph-level label
        data.y = torch.tensor(label, dtype=torch.long)

        return data
