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
import random
import collections
import torch

def verify_hypergraph(data: 'HeteroData'):
    """
    Quick integrity / sanity checks for the aggregated graph.
    Prints a readable summary and raises AssertionError on failures.
    """


    num_nodes  = data['object'].num_nodes
    rel_e      = data['object', 'relation', 'object'].edge_index.size(1)
    temp_e     = data['object', 'temporal', 'object'].edge_index.size(1)
    frames     = data['object'].frame_id.unique().tolist()

    # print("🔎  GRAPH SUMMARY")
    # print(f"• nodes:            {num_nodes}")
    # print(f"• relation edges:   {rel_e}")
    # print(f"• temporal edges:   {temp_e}")
    # print(f"• frames present:   {frames}")

    # label histogram
    label_vec = data['object'].label
    hist = collections.Counter(label_vec.tolist())
    # print("• label histogram:")
    # for lab, cnt in sorted(hist.items()):
    #     print(f"    label {lab}: {cnt}")
    # print()

    # --- consistency checks ---------------------------------------
    src, dst = data['object', 'temporal', 'object'].edge_index
    assert (label_vec[src] == label_vec[dst]).all(), "Temporal edge joins different labels!"
    assert (data['object'].frame_id[src] < data['object'].frame_id[dst]).all(), \
           "Temporal edge points backward in time!"
    # print("✅  all basic checks passed")

    # peek first few temporal edges
    # print("\nFirst 10 temporal edges:")
    # for s, d in zip(src[:10].tolist(), dst[:10].tolist()):
    #     print(f"  {s:>4} (f{data['object'].frame_id[s].item():02d}, L{label_vec[s].item()})"
    #           f" → {d:>4} (f{data['object'].frame_id[d].item():02d}, L{label_vec[d].item()})")

    if len(frames) > 0:
        return True
    else:
        return False


class GraphDataset(Dataset):
    def __init__(self, data_dir, embedder, metadata_csv, mapping_vn2act, node_drop_p = 0.0, is_train=True):

        self.data_dir = data_dir
        self.embedder = embedder
        self.mapping_vn2act = mapping_vn2act

        self.epic_hd_val_prob = 0.0
        self.iphone_val_prob = 0.5

        graph_file_count = 0
        self.filenames_list = []
        print(f"Loading graph dataset from {self.data_dir}")
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                # print(root + file)
                if file.endswith(".h5"):
                    file_path = os.path.join(root, file)
                    self.filenames_list.append(file_path)
                    graph_file_count += 1
        print(f"Found {graph_file_count} graph files in {self.data_dir}")
        self.node_drop_p = node_drop_p

        # check if a metadata CSV file exists
        self.metadata_csv_path = self.data_dir + "/metadata.csv"
        
        # randomly sample 10% of the dataset for validation
        if not os.path.exists(self.metadata_csv_path):
            print(f"Metadata CSV file not found at {self.metadata_csv_path}. Creating a new one.")
            
            # create a new metadata CSV file
            with open(self.metadata_csv_path, 'w', newline='') as csvfile:
                fieldnames = ['path', 'is_train']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for graph_path in self.filenames_list:
                    # save the path and whether it's for training or validation
                    # diffrent prob for epic_hd and iphone
                    if "phone" in graph_path:
                        is_train = random.random() < (1 - self.iphone_val_prob)
                    elif "epic_hd" in graph_path:
                        is_train = random.random() < (1 - self.epic_hd_val_prob)
                    else:
                        raise ValueError(f"Unknown dataset type in path: {graph_path}")

                    writer.writerow({'path': graph_path, 'is_train': is_train})

        # load the metadata CSV file
        self.metadata_df = pd.read_csv(self.metadata_csv_path)
        # filter the dataset based on the is_train flag
        if is_train:
            self.metadata_df = self.metadata_df[self.metadata_df['is_train'] == True]
        else:
            self.metadata_df = self.metadata_df[self.metadata_df['is_train'] == False]

        # load the filenames from the metadata DataFrame
        self.filenames_list = self.metadata_df['path'].tolist()
        print(f"Filtered dataset contains {len(self.filenames_list)} files for {'training' if is_train else 'validation'}.")




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
        if len(node_feats) == 0:
            new_id = idx+1
            return self.__getitem__(new_id)


        x           = torch.cat(node_feats, dim=0)                       # (ΣN,384)
        pos_all     = None #torch.cat(node_pos,   dim=0)  #for ablation study
        labels_all = torch.cat(node_labels, dim=0) 
        orig_ids    = torch.cat(orig_ids)                                # (ΣN,)
        frame_ids   = torch.cat(frame_ids)                               # (ΣN,)

        rel_index   = torch.stack([
                            torch.cat(rel_edges_src),
                            torch.cat(rel_edges_dst)
                    ], dim=0)                                         # (2,ΣE_rel)

        # if rel_edge_attr:                                                # may be empty
        #     #rel_attr = torch.cat(rel_edge_attr)                          # (ΣE_rel, ...) for ablation study
        #     rel_attr = None
        #     EDGE_DIM = None
        # else:
        #     rel_attr = None
        rel_attr = None  #None #ablation study
        if temp_edges_src:                                               # may be empty
            temp_index = torch.stack([
                            torch.cat(temp_edges_src),
                            torch.cat(temp_edges_dst)
                        ], dim=0)                                       # (2,ΣE_temp)
        else:
            temp_index = torch.empty(2, 0, dtype=torch.long)    


        # Create PyG graph
        data = HeteroData()

        # Set a drop probability for nodes (objects)
        drop_rate = self.node_drop_p
        num_nodes_total = x.size(0)
        keep_mask = torch.rand(num_nodes_total) > drop_rate

        # Update node features and attributes
        x = x[keep_mask]
        #pos_all = pos_all[keep_mask] #apblation study
        labels_all = labels_all[keep_mask]
        orig_ids = orig_ids[keep_mask]
        frame_ids = frame_ids[keep_mask]

        # Build a mapping from old node indices to new indices
        old_indices = torch.arange(num_nodes_total)
        new_indices = -torch.ones(num_nodes_total, dtype=torch.long)
        new_indices[keep_mask] = torch.arange(keep_mask.sum())

        # Filter in-frame (relation) edges: keep only edges with both endpoints retained
        rel_src = rel_index[0]
        rel_dst = rel_index[1]
        edge_mask = keep_mask[rel_src] & keep_mask[rel_dst]
        rel_index = torch.stack([new_indices[rel_src[edge_mask]], new_indices[rel_dst[edge_mask]]], dim=0)
        if rel_attr is not None:
            rel_attr = rel_attr[edge_mask]

        # Filter temporal edges in the same way
        if temp_index.size(1) > 0:
            temp_src = temp_index[0]
            temp_dst = temp_index[1]
            temp_mask = keep_mask[temp_src] & keep_mask[temp_dst]
            temp_index = torch.stack([new_indices[temp_src[temp_mask]], new_indices[temp_dst[temp_mask]]], dim=0)
        

        data['object'].x        = x
        #data['object'].pos      = pos_all $ablation study
        data['object'].label    = labels_all

        data['object'].orig_id  = orig_ids
        data['object'].frame_id = frame_ids


        data['object', 'relation', 'object'].edge_index = rel_index
        # if rel_attr is not None: # ablation study
        #     data['object', 'relation', 'object'].edge_attr  = rel_attr
      # --- geometry ablation: don't store any attributes ---
        # DO NOT assign edge_attr at all
        # If it might already exist from earlier code, delete it:
        if 'edge_attr' in data['object', 'relation', 'object']:
            del data['object', 'relation', 'object'].edge_attr   

        data['object', 'temporal', 'object'].edge_index = temp_index




        # Graph-level label
        zeros = torch.zeros((3806,), dtype=torch.float)  
        zeros[label] = 1.0
        data.y = zeros.unsqueeze(0)

                # at the end of __getitem__
        print("rel edge_index :", data['object','relation','object'].edge_index.shape)
        print("'edge_attr' present:", 'edge_attr' in data['object','relation','object'])


        return data

# import os
# import json
# import torch
# from torch.utils.data import Dataset
# from torch_geometric.data import HeteroData
# import pandas as pd
# import csv
# import h5py
# import itertools 
# import collections
# import networkx as nx
# import numpy as np
# from pathlib import Path

# def verify_hypergraph(data: 'HeteroData'):
#     """
#     Quick integrity / sanity checks for the aggregated graph.
#     Prints a readable summary and raises AssertionError on failures.
#     """
#     import collections

#     num_nodes  = data['object'].num_nodes
#     rel_e      = data['object', 'relation', 'object'].edge_index.size(1)
#     temp_e     = data['object', 'temporal', 'object'].edge_index.size(1)
#     frames     = data['object'].frame_id.unique().tolist()

#     print("🔎  GRAPH SUMMARY")
#     print(f"• nodes:            {num_nodes}")
#     print(f"• relation edges:   {rel_e}")
#     print(f"• temporal edges:   {temp_e}")
#     print(f"• frames present:   {frames}")

#     # label histogram
#     label_vec = data['object'].label
#     hist = collections.Counter(label_vec.tolist())
#     print("• label histogram:")
#     for lab, cnt in sorted(hist.items()):
#         print(f"    label {lab}: {cnt}")
#     print()

#     # --- consistency checks ---------------------------------------
#     src, dst = data['object', 'temporal', 'object'].edge_index
#     assert (label_vec[src] == label_vec[dst]).all(), "Temporal edge joins different labels!"
#     assert (data['object'].frame_id[src] < data['object'].frame_id[dst]).all(), \
#            "Temporal edge points backward in time!"
#     print("✅  all basic checks passed")

#     # peek first few temporal edges
#     print("\nFirst 10 temporal edges:")
#     for s, d in zip(src[:10].tolist(), dst[:10].tolist()):
#         print(f"  {s:>4} (f{data['object'].frame_id[s].item():02d}, L{label_vec[s].item()})"
#               f" → {d:>4} (f{data['object'].frame_id[d].item():02d}, L{label_vec[d].item()})")


# class GraphDataset(Dataset):
#     def __init__(self, data_dir, embedder, metadata_csv, mapping_vn2act):

#         self.data_dir = data_dir
#         self.embedder = embedder
#         self.mapping_vn2act = mapping_vn2act

#         graph_file_count = 0
#         self.filenames_list = []
#         abs_dir = Path(data_dir).resolve()
#         print(f"Loading graph dataset from {abs_dir}")
#         print(f"Loading graph dataset from {self.data_dir}")
#         self.filenames_list = [str(p)                                       # keep as strings
#                        for p in Path(self.data_dir).rglob("*.h5")]  # recurses automatically
#         graph_file_count = len(self.filenames_list)
#         # for _, dirnames, _ in os.walk(self.data_dir):
#         #     # look at all the directories
#         #     for dirname in dirnames:
#         #         for dirpath, _, filenames in os.walk(os.path.join(self.data_dir, dirname)):
#         #             for filename in filenames:
#         #                 if filename.endswith(".h5"):
#         #                     graph_file_count += 1
#         #                     self.filenames_list.append(os.path.join(dirpath, filename))
#         print(f"Found {graph_file_count} graph files in {self.data_dir}")





#     def __len__(self):
#         return len(self.filenames_list)

#     def __getitem__(self, idx):
#         # Get info for one sample
#         # print(f"Processing sample {idx}")
#         graph_path = self.filenames_list[idx]

#         # Load hf5 file
#         with h5py.File(graph_path, 'r') as f:
#             label =  f["logits_argmax"][()]
#             # print(f"Graph label: {label}")
#             frames_grp = f["frames"]
#             frames = []

#             for frame_key in sorted(frames_grp.keys()):
#                 g = frames_grp[frame_key]
#                 frames.append(
#                     {
#                         "features":  g["features"][()],
#                         "edges":     g["edge_index"][()],
#                         "edge_lbl":  g["edge_labels"][()],
#                         "pos":       g["pos"][()],
#                         "labels":    g["labels"][()]
#                     }
#                 )

#         #     print(f"Loaded {len(frames)} frames from {graph_path}")

#         # print("features shape:", frames[0]["features"].shape)
#         # print("edges shape:", frames[0]["edges"].shape)
#         # print("edge_lbl shape:", frames[0]["edge_lbl"].shape)
#         # print("pos shape:", frames[0]["pos"].shape)
#         # print("labels shape:", frames[0]["labels"].shape)
#         # print("Graph count:", len(frames))




#         node_feats   = []            # (Σ N_i, 384)
#         node_pos     = []            # (Σ N_i, 3) 
#         orig_ids     = []            # original local id within its frame
#         frame_ids    = []            # which frame the node came from
#         node_labels  = []
#         global_id    = 0             # running node counter

#         rel_edges_src = []           # intra-frame "relation" edges
#         rel_edges_dst = []
#         rel_edge_attr = []       
#         prev_label_nodes = {1: None, 2: None}   # None == “no previous frame yet”    
#         temp_edges_src = []          # temporal edges (prev → curr)
#         temp_edges_dst = []

#         missing_cnt = 0

#         #here we create a hyper graph connecting 16 frames together
#         total_node_count = 0

#         for f_idx, fr in enumerate(frames):
#             num_nodes = fr["features"].shape[0]
#             if num_nodes == 0:
#                 continue                        # skip empty frames

#             # ---- nodes ----------------------------------------------------
#             feats = torch.from_numpy(fr["features"]).float()          # (N,384)
#             node_feats.append(feats)
#             feats = torch.from_numpy(fr["features"]).float()          # (N,384)
#             node_feats.append(feats)

#             pos   = torch.from_numpy(fr["pos"]).float()               # (N,3) ← NEW
#             node_pos.append(pos)                                      #        ← NEW

#             orig_ids.append(torch.arange(num_nodes, dtype=torch.long))
#             frame_ids.append(torch.full((num_nodes,), f_idx, dtype=torch.long))

#             # ---- in-frame 'relation' edges --------------------------------
#             ei = torch.from_numpy(fr["edges"]).long()          # (2,E)
#             rel_edges_src.append(ei[0] + global_id)
#             rel_edges_dst.append(ei[1] + global_id)

#             if fr["edge_lbl"].size != 0:                       # optional
#                 edge_attr = torch.from_numpy(fr["edge_lbl"]).squeeze(1).float()
#                 rel_edge_attr.append(edge_attr)

#             # ---- temporal edges  (label-wise full bipartite) -------------
#             labels_t = torch.from_numpy(fr["labels"]).long()   # (N,)
#             node_labels.append(labels_t)

#             for lab in (1, 2):
#                 curr_nodes = torch.nonzero(labels_t == lab, as_tuple=False).flatten()
#                 if prev_label_nodes[lab] is not None and curr_nodes.numel() and prev_label_nodes[lab].numel():
#                     # full Cartesian product: every prev → every curr
#                     src_repeat = prev_label_nodes[lab].repeat_interleave(curr_nodes.numel())
#                     dst_tile   = curr_nodes.repeat(prev_label_nodes[lab].numel())
#                     temp_edges_src.append(src_repeat + 0)          # already global
#                     temp_edges_dst.append(dst_tile  + global_id)   # offset

#                 # remember current nodes (in global indexing) for next loop
#                 prev_label_nodes[lab] = curr_nodes + global_id

#             # advance global node counter
#             global_id += num_nodes
#         if len(node_feats) == 0:
#             # Skip empty graph, try next index (wrap around if at end)
#             print(f"Skipping empty graph at index {idx} ({graph_path})")
#             next_idx = (idx + 1) % len(self.filenames_list)
#             return self.__getitem__(next_idx)

#         # 2.  Stack everything into tensors
#         # ------------------------------------------------------------------
#         x           = torch.cat(node_feats, dim=0)                       # (ΣN,384)
#         pos_all     = torch.cat(node_pos,   dim=0) 
#         labels_all = torch.cat(node_labels, dim=0) 
#         orig_ids    = torch.cat(orig_ids)                                # (ΣN,)
#         frame_ids   = torch.cat(frame_ids)                               # (ΣN,)

#         rel_index   = torch.stack([
#                             torch.cat(rel_edges_src),
#                             torch.cat(rel_edges_dst)
#                     ], dim=0)                                         # (2,ΣE_rel)

#         if rel_edge_attr:                                                # may be empty
#             rel_attr = torch.cat(rel_edge_attr)                          # (ΣE_rel, ...)
#         else:
#             rel_attr = None

#         if temp_edges_src:                                               # may be empty
#             temp_index = torch.stack([
#                             torch.cat(temp_edges_src),
#                             torch.cat(temp_edges_dst)
#                         ], dim=0)                                       # (2,ΣE_temp)
#         else:
#             temp_index = torch.empty(2, 0, dtype=torch.long)    




#         # Create PyG graph
#         data = HeteroData()
        

#         data['object'].x        = x
#         data['object'].pos      = pos_all
#         data['object'].label    = labels_all

#         data['object'].orig_id  = orig_ids
#         data['object'].frame_id = frame_ids

#         # augment the rek_index and rel_attr by sampling edges in the other direction as well
#         # augmented_edge_index = torch.zeros((2, rel_index.shape[1] * 2) , dtype=torch.int)
#         # # print("edge index shape:", rel_index.shape)
#         # # print("augmented edge index shape:", augmented_edge_index.shape)
#         # augmented_edge_index[0, :rel_index.shape[1]] = rel_index[0, :]
#         # augmented_edge_index[1, :rel_index.shape[1]] = rel_index[1, :]

#         # augmented_edge_index[0, rel_index.shape[1]:] = rel_index[1, :]
#         # augmented_edge_index[1, rel_index.shape[1]:] = rel_index[0, :]



#         # augment the edge features too
#         # augmented_rel_attr = torch.cat((rel_attr, rel_attr), dim=0)

#         src, dst = rel_index            # alias for readability, both are (E,)

#         # build the reverse edge list
#         rev_index = torch.stack([dst, src], dim=0)          # (2, E)

#         # concatenate originals + reverse
#         rel_index_aug = torch.cat([rel_index, rev_index], dim=1)   # (2, 2E)

#         # If you have edge features, just duplicate them…
#         if rel_attr is not None:
#             rel_attr_aug = torch.cat([rel_attr, rel_attr.clone()], dim=0)  # (2E, …)
#         else:
#             rel_attr_aug = None

#         data['object', 'relation', 'object'].edge_index = rel_index_aug
#         if rel_attr is not None:
#             data['object', 'relation', 'object'].edge_attr  = rel_attr_aug

#         data['object', 'temporal', 'object'].edge_index = temp_index




#         # Graph-level label
#         zeros = torch.zeros((3806,), dtype=torch.float)  
#         zeros[label] = 1.0
#         data.y = zeros.unsqueeze(0)

#         # verify_hypergraph(data)

#         return data
