import os
import json
import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
import pandas as pd
import csv
import h5py
import itertools 
import collections
import networkx as nx
import numpy as np

def verify_hypergraph(data: 'HeteroData'):
    """
    Quick integrity / sanity checks for the aggregated graph.
    Prints a readable summary and raises AssertionError on failures.
    """
    import collections

    num_nodes  = data['object'].num_nodes
    rel_e      = data['object', 'relation', 'object'].edge_index.size(1)
    temp_e     = data['object', 'temporal', 'object'].edge_index.size(1)
    frames     = data['object'].frame_id.unique().tolist()

    print("🔎  GRAPH SUMMARY")
    print(f"• nodes:            {num_nodes}")
    print(f"• relation edges:   {rel_e}")
    print(f"• temporal edges:   {temp_e}")
    print(f"• frames present:   {frames}")

    # label histogram
    label_vec = data['object'].label
    hist = collections.Counter(label_vec.tolist())
    print("• label histogram:")
    for lab, cnt in sorted(hist.items()):
        print(f"    label {lab}: {cnt}")
    print()

    # --- consistency checks ---------------------------------------
    src, dst = data['object', 'temporal', 'object'].edge_index
    assert (label_vec[src] == label_vec[dst]).all(), "Temporal edge joins different labels!"
    assert (data['object'].frame_id[src] < data['object'].frame_id[dst]).all(), \
           "Temporal edge points backward in time!"
    print("✅  all basic checks passed")

    # peek first few temporal edges
    print("\nFirst 10 temporal edges:")
    for s, d in zip(src[:10].tolist(), dst[:10].tolist()):
        print(f"  {s:>4} (f{data['object'].frame_id[s].item():02d}, L{label_vec[s].item()})"
              f" → {d:>4} (f{data['object'].frame_id[d].item():02d}, L{label_vec[d].item()})")


class GraphDataset(Dataset):
    def __init__(self, data_dir, embedder, metadata_csv, mapping_vn2act):

        self.data_dir = data_dir
        self.embedder = embedder
        self.mapping_vn2act = mapping_vn2act

        graph_file_count = 0
        self.filenames_list = []
        print(f"Loading graph dataset from {self.data_dir}")
        for _, dirnames, _ in os.walk(self.data_dir):
            # look at all the directories
            for dirname in dirnames:
                for dirpath, _, filenames in os.walk(os.path.join(self.data_dir, dirname)):
                    for filename in filenames:
                        if ".h5" in filename:
                            graph_file_count += 1
                            self.filenames_list.append(os.path.join(dirpath, filename))
        print(f"Found {graph_file_count} graph files in {self.data_dir}")





    def __len__(self):
        return len(self.filenames_list)

    def __getitem__(self, idx):
        # Get info for one sample
        # print(f"Processing sample {idx}")
        graph_path = self.filenames_list[idx]

        # Load hf5 file
        with h5py.File(graph_path, 'r') as f:
            label =  f["logits_argmax"][()]
            # print(f"Graph label: {label}")
            frames_grp = f["frames"]
            frames = []

            for frame_key in sorted(frames_grp.keys()):
                g = frames_grp[frame_key]
                frames.append(
                    {
                        "features":  g["features"][()],
                        "edges":     g["edge_index"][()],
                        "edge_lbl":  g["edge_labels"][()],
                        "pos":       g["pos"][()],
                        "labels":    g["labels"][()]
                    }
                )

        #     print(f"Loaded {len(frames)} frames from {graph_path}")

        # print("features shape:", frames[0]["features"].shape)
        # print("edges shape:", frames[0]["edges"].shape)
        # print("edge_lbl shape:", frames[0]["edge_lbl"].shape)
        # print("pos shape:", frames[0]["pos"].shape)
        # print("labels shape:", frames[0]["labels"].shape)
        # print("Graph count:", len(frames))




        node_feats   = []            # (Σ N_i, 384)
        node_pos     = []            # (Σ N_i, 3) 
        orig_ids     = []            # original local id within its frame
        frame_ids    = []            # which frame the node came from
        node_labels  = []
        global_id    = 0             # running node counter

        rel_edges_src = []           # intra-frame "relation" edges
        rel_edges_dst = []
        rel_edge_attr = []       
        prev_label_nodes = {1: None, 2: None}   # None == “no previous frame yet”    
        temp_edges_src = []          # temporal edges (prev → curr)
        temp_edges_dst = []

        #here we create a hyper graph connecting 16 frames together
        total_node_count = 0

        for f_idx, fr in enumerate(frames):
            num_nodes = fr["features"].shape[0]
            if num_nodes == 0:
                continue                        # skip empty frames

            # ---- nodes ----------------------------------------------------
            feats = torch.from_numpy(fr["features"]).float()          # (N,384)
            node_feats.append(feats)
            feats = torch.from_numpy(fr["features"]).float()          # (N,384)
            node_feats.append(feats)

            pos   = torch.from_numpy(fr["pos"]).float()               # (N,3) ← NEW
            node_pos.append(pos)                                      #        ← NEW

            orig_ids.append(torch.arange(num_nodes, dtype=torch.long))
            frame_ids.append(torch.full((num_nodes,), f_idx, dtype=torch.long))

            # ---- in-frame 'relation' edges --------------------------------
            ei = torch.from_numpy(fr["edges"]).long()          # (2,E)
            rel_edges_src.append(ei[0] + global_id)
            rel_edges_dst.append(ei[1] + global_id)

            if fr["edge_lbl"].size != 0:                       # optional
                edge_attr = torch.from_numpy(fr["edge_lbl"]).squeeze(1).float()
                rel_edge_attr.append(edge_attr)

            # ---- temporal edges  (label-wise full bipartite) -------------
            labels_t = torch.from_numpy(fr["labels"]).long()   # (N,)
            node_labels.append(labels_t)

            for lab in (1, 2):
                curr_nodes = torch.nonzero(labels_t == lab, as_tuple=False).flatten()
                if prev_label_nodes[lab] is not None and curr_nodes.numel() and prev_label_nodes[lab].numel():
                    # full Cartesian product: every prev → every curr
                    src_repeat = prev_label_nodes[lab].repeat_interleave(curr_nodes.numel())
                    dst_tile   = curr_nodes.repeat(prev_label_nodes[lab].numel())
                    temp_edges_src.append(src_repeat + 0)          # already global
                    temp_edges_dst.append(dst_tile  + global_id)   # offset

                # remember current nodes (in global indexing) for next loop
                prev_label_nodes[lab] = curr_nodes + global_id

            # advance global node counter
            global_id += num_nodes

        # 2.  Stack everything into tensors
        # ------------------------------------------------------------------
        x           = torch.cat(node_feats, dim=0)                       # (ΣN,384)
        pos_all     = torch.cat(node_pos,   dim=0) 
        labels_all = torch.cat(node_labels, dim=0) 
        orig_ids    = torch.cat(orig_ids)                                # (ΣN,)
        frame_ids   = torch.cat(frame_ids)                               # (ΣN,)

        rel_index   = torch.stack([
                            torch.cat(rel_edges_src),
                            torch.cat(rel_edges_dst)
                    ], dim=0)                                         # (2,ΣE_rel)

        if rel_edge_attr:                                                # may be empty
            rel_attr = torch.cat(rel_edge_attr)                          # (ΣE_rel, ...)
        else:
            rel_attr = None

        if temp_edges_src:                                               # may be empty
            temp_index = torch.stack([
                            torch.cat(temp_edges_src),
                            torch.cat(temp_edges_dst)
                        ], dim=0)                                       # (2,ΣE_temp)
        else:
            temp_index = torch.empty(2, 0, dtype=torch.long)    




        # Create PyG graph
        data = HeteroData()
        

        data['object'].x        = x
        data['object'].pos      = pos_all
        data['object'].label    = labels_all

        data['object'].orig_id  = orig_ids
        data['object'].frame_id = frame_ids

        # augment the rek_index and rel_attr by sampling edges in the other direction as well
        # augmented_edge_index = torch.zeros((2, rel_index.shape[1] * 2) , dtype=torch.int)
        # # print("edge index shape:", rel_index.shape)
        # # print("augmented edge index shape:", augmented_edge_index.shape)
        # augmented_edge_index[0, :rel_index.shape[1]] = rel_index[0, :]
        # augmented_edge_index[1, :rel_index.shape[1]] = rel_index[1, :]

        # augmented_edge_index[0, rel_index.shape[1]:] = rel_index[1, :]
        # augmented_edge_index[1, rel_index.shape[1]:] = rel_index[0, :]



        # augment the edge features too
        # augmented_rel_attr = torch.cat((rel_attr, rel_attr), dim=0)

        data['object', 'relation', 'object'].edge_index = rel_index
        if rel_attr is not None:
            data['object', 'relation', 'object'].edge_attr  = rel_attr

        data['object', 'temporal', 'object'].edge_index = temp_index




        # Graph-level label
        zeros = torch.zeros((3806,), dtype=torch.float)  
        zeros[label] = 1.0
        data.y = zeros.unsqueeze(0)

        # verify_hypergraph(data)

        return data
