# graph_generator.py
from __future__ import annotations

import networkx as nx                 # pip install networkx
import numpy as np
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from torch_geometric.data import HeteroData
import torch

import rerun as rr

class GraphGenerator:
    """
    Build a simple scene-graph from:
        • SAM-2 masks      (list[dict] with "segmentation", "label", … )
        • Hand landmarks   (list[(x,y,z)] per hand from MediaPipe)
    """

    def __init__(self, cfg):
        print("Initializing GraphGenerator with config:", cfg)




    def generate_graph(self, image, objects, hands):
        hd = HeteroData()

        # ------------------------------------------------------------------
        # 1. collect object nodes
        feat, labels, pos = [], [], []
        for i, obj in enumerate(objects):
            feat.append(obj.embedding)            # (D,)
            labels.append(obj.name)
            pos.append(obj.position)              # (x,y,z)  or  (x,y)

        obj_cnt = len(feat)                       # keep for index offset

        # ------------------------------------------------------------------
        # 2. append hand nodes
        def add_hand(side, hand):
            if hand.embedding is None:
                return
            feat.append(hand.embedding)
            labels.append(f"{side}_hand")
            pos.append(hand.mean_pos)

        if hands is None:
            # retunr an empty graph if no hands are provided
            print("No hands provided, returning empty graph.")
            return hd            
        
        # check if hand has attribute embedding
        if hasattr(hands.left_hand, 'embedding'):
            add_hand("left",  hands.left_hand)

        if hasattr(hands.right_hand, 'embedding'):
            add_hand("right", hands.right_hand)

        # convert labes to 0 for unknown, 1 for left hand, 2 for right hand
        labels_int = [0 if label == "unknown" else 1 if label == "left_hand" else 2 for label in labels]

        # tensors
        hd['object'].x   = torch.tensor(feat, dtype=torch.float)
        hd['object'].labels = torch.tensor(labels_int, dtype=torch.long)  # labels as indices
        hd['object'].pos = torch.tensor(pos,  dtype=torch.float)
        hd['object'].node_id = torch.arange(len(feat))  # optional bookkeeping

        print(f"Labels: {labels_int}")

        # ------------------------------------------------------------------
        # 3. build hand→object distance edges
        edge_src, edge_dst, edge_attr = [], [], []


        obj_pos_np = np.asarray(pos[:obj_cnt])               # objects only
        for h_idx in range(obj_cnt, len(feat)):              # indices of hands
            d = np.linalg.norm(obj_pos_np - pos[h_idx], axis=1)  # (obj_cnt,)
            rel_pos = obj_pos_np - pos[h_idx]  # (obj_cnt, 3)
            # get a vector (obj_cnt, 4) with the distance and the relative position
            d = np.hstack((d[:, np.newaxis], rel_pos))  # (obj_cnt, 4)  distance + relative position
            edge_src.extend([h_idx] * obj_cnt)
            edge_dst.extend(range(obj_cnt))
            edge_attr.extend(d.tolist())


        hd['object', 'distance', 'object'].edge_index = torch.tensor(
            [edge_src, edge_dst], dtype=torch.long
        )
        hd['object', 'distance', 'object'].edge_attr  = torch.tensor(
            edge_attr, dtype=torch.float
        ).unsqueeze(1)                                       # (E,1)

        print(f"Edges source: {edge_src}")
        print(f"Edges destination: {edge_dst}")

        rerun_edges = []
        for src, dst, attr in zip(edge_src, edge_dst, edge_attr):
            rerun_edges.append((src, dst))


        # ------------------------------------------------------------------
        # 4. log to Rerun for visual sanity-check
        rr.log("nodes", rr.GraphNodes(
            node_ids=list(range(len(labels))),
            labels=labels,
        ))
        rr.log("nodes", rr.GraphEdges(
            rerun_edges,
        ))

        return hd
                
