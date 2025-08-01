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
        labels_int = []
        for label in labels:
            if label == "left_hand":
                labels_int.append(1)
            elif label == "right_hand":
                labels_int.append(2)
            else:
                labels_int.append(0)  # object or unknown
        valid_pos   = []
        valid_embed = []
        valid_name  = []

        for o in objects:
            p = getattr(o, "position", None)

            # keep only iterable 3-vectors with finite numbers
            if not (hasattr(p, "__iter__") and len(p) == 3):
                print(f"[DEBUG] dropped node: label={o.name}  raw_pos={p}")
                continue
            p = np.asarray(p, dtype=np.float32)

            if p.shape != (3,) or np.isnan(p).any():
                print(f"[DEBUG] dropped node: label={o.name}  nan_pos={p}")
                continue

            valid_pos.append(p)
            valid_embed.append(o.embedding)
            valid_name.append(o.name)

        # Nothing left? skip this frame
        if not valid_pos:
            return None             # or `continue`
        # tensors
        hd['object'].x   = torch.tensor(feat, dtype=torch.float)
        hd['object'].labels = torch.tensor(labels_int, dtype=torch.long)  # labels as indices
        bad = []
        for i, p in enumerate(pos):
            if not (hasattr(p, "__iter__") and len(p) == 3):
                bad.append((i, p, objects[i].name))

        if bad:
            print("\n===== INVALID position vector(s) detected =====")
            for idx, p, name in bad:
                print(f" idx={idx:<2d}  type={type(p).__name__:>10}  value={p}  label={name}")
            raise RuntimeError("Aborting on invalid position")
        hd['object'].pos = torch.tensor(pos,  dtype=torch.float)
        hd['object'].node_id = torch.arange(len(feat))  # optional bookkeeping

        # print(f"Labels: {labels_int}")

        # ------------------------------------------------------------------
        # 3. build hand→object distance edges
        edge_src, edge_dst, edge_attr = [], [], []

        for h_idx in range(obj_cnt, len(feat)):       # every hand node
            edge_src.extend([h_idx] * obj_cnt)        # connect to *all* objects
            edge_dst.extend(range(obj_cnt))

        # create edge_index only; do NOT create edge_attr
        hd['object', 'relation', 'object'].edge_index = torch.tensor(
            [edge_src, edge_dst], dtype=torch.long
        )

        # ------------- after you assign edge_index -----------------
        print("relation edge_index shape:", 
            hd['object', 'relation', 'object'].edge_index.shape)

        # confirm no edge_attr exists
        print("'edge_attr' present?", 
            'edge_attr' in hd['object', 'relation', 'object'])

        # IMPORTANT: do not assign edge_attr at all


        #ablation study

        # obj_pos_np = np.asarray(pos[:obj_cnt])               # objects only
        # for h_idx in range(obj_cnt, len(feat)):              # indices of hands
        #     d = np.linalg.norm(obj_pos_np - pos[h_idx], axis=1)  # (obj_cnt,)
        #     rel_pos = obj_pos_np - pos[h_idx]  # (obj_cnt, 3)
        #     # get a vector (obj_cnt, 4) with the distance and the relative position
        #     d = np.hstack((d[:, np.newaxis], rel_pos))  # (obj_cnt, 4)  distance + relative position
        #     edge_src.extend([h_idx] * obj_cnt)
        #     edge_dst.extend(range(obj_cnt))
        #     edge_attr.extend(d.tolist())


        # hd['object', 'distance', 'object'].edge_index = torch.tensor(
        #     [edge_src, edge_dst], dtype=torch.long
        # )
        # hd['object', 'distance', 'object'].edge_attr  = torch.tensor(
        #     edge_attr, dtype=torch.float
        # ).unsqueeze(1)                                       # (E,1)

        # print(f"Edges source: {edge_src}")
        # print(f"Edges destination: {edge_dst}")

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
                
