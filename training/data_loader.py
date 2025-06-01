import os
import json
import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
import pandas as pd
import csv


class GraphDataset(Dataset):
    def __init__(self, data_dir, embedder, metadata_csv, mapping_vn2act):
        """
        Args:
            data_dir (str): Directory where graphs are stored.
            embedder (model instance): Text embedder.
            metadata_csv (str): CSV file with [graph_path, verb_class, noun_class] columns.
            mapping_vn2act (dict): Mapping from 'verb:noun' string to action ID.
        """
        self.data_dir = data_dir
        self.embedder = embedder
        self.mapping_vn2act = mapping_vn2act

        # Load metadata
        self.metadata = pd.read_csv(metadata_csv)
        print(f"Loaded {len(self.metadata)} samples from {metadata_csv}")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Get info for one sample
        row = self.metadata.iloc[idx]
        graph_path = os.path.join(self.data_dir, row['participant_id'],
                                row['video_id'], row['narration_id'],
                                "graph.json")
        verb_id = int(row['verb_class'])
        noun_id = int(row['noun_class'])

        # Map (verb:noun) -> action label
        vn = f"{verb_id}:{noun_id}"
        label = self.mapping_vn2act[vn]

        # Load graph JSON
        with open(graph_path, "r") as f:
            json_data = json.load(f)

        node_mapping = {} # (object_id, timestamp) → unique graph node ID
        relational_edges = []
        temporal_edges = []
        relation_labels = []
        seen_objects = {}

        node_counter = 0

        # Build the graph
        for t, frame_data in json_data.items():
            timestamp = int(t)
            for obj_id in frame_data["objs"]:
                if (obj_id, timestamp) not in node_mapping:
                    node_mapping[(obj_id, timestamp)] = node_counter
                    node_counter += 1

                # Track object across frames for temporal edges
                if obj_id in seen_objects:
                    prev_timestamp = seen_objects[obj_id]
                    prev_id = node_mapping[(obj_id, prev_timestamp)]
                    curr_id = node_mapping[(obj_id, timestamp)]
                    temporal_edges.append([prev_id, curr_id])

                seen_objects[obj_id] = timestamp

            # Extract relational edges within this frame
            for (src, tgt), relation_label in zip(frame_data["relations"], frame_data["edges_labels"]):
                src_id = node_mapping[(src, timestamp)]
                tgt_id = node_mapping[(tgt, timestamp)]
                relational_edges.append([src_id, tgt_id])
                relation_labels.append(relation_label)

        relational_edge_index = torch.tensor(relational_edges, 
                                             dtype=torch.long).t().contiguous()
        temporal_edge_index = torch.tensor(temporal_edges, 
                                           dtype=torch.long).t().contiguous()

        # Convert text labels into embeddings
        labels = [json_data[t]["objs_labels"][i] for t in json_data.keys() for i in range(len(json_data[t]["objs_labels"]))]
        node_embeddings = self.embedder.extract_text_fts(labels).clone().detach()

        # Convert relation labels into embeddings
        relation_embeddings = self.embedder.extract_text_fts(relation_labels).clone().detach() if relation_labels else torch.empty((0, 512))

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
