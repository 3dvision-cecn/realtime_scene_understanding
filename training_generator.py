from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
import avion.models.model_clip as model_clip
from avion.models.utils import inflate_positional_embeds
from avion.utils.misc import generate_label_map
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

import orjson, pathlib
import json as js

from datetime import datetime
import h5py

from avion_wrapper import AVIONForwardModule


class TrainingGenerator():

    def __init__(self, cfg):
        self.cfg = cfg

        self.avion = AVIONForwardModule()

        self.sequence = []

        # create a new directory for the training samples
        root_dir = "graph_samples"
        # add date and time to the directory name

        source = cfg.video.path

        # get the last part of the source path
        last_slash = source.rfind('/')
        name = source[last_slash + 1:] 

        # get the string between the last two slashes either train or val
        second_last_slash = source[:last_slash].rfind('/')
        folder = source[second_last_slash + 1:last_slash]
        


        self.sample_dir = f"{root_dir}/{folder}/{name}"
        os.makedirs(self.sample_dir, exist_ok=True)

        self.sequence_count = 0


    def add_sequence(self, image, graph):
        """
        Adds a sequence to the training generator.
        """
        # add the tuple (image, graph) to the sequence
        self.sequence.append((image, graph))

        if len(self.sequence) == 16:
            print("Sequence ready, generating sample...")
            self.generate_sample()
            self.sequence = []  # reset the sequence after generating a sample
            self.sequence_count += 1


    def generate_sample(self):
        sample_path = os.path.join(self.sample_dir,
                           f"sample_{self.sequence_count:03d}.h5")
        print("Saving sample to:", sample_path)
        frames = np.stack([img for img, _ in self.sequence], axis=0)  # (T, H, W, C)

        # run the AVION forward module
        logits_margmax, logits_softmax = self.avion(frames)
        logits_argmax = logits_margmax.cpu().numpy().astype(np.int16)   # (T,) → smaller dtype
        
        act_text = self.avion.get_text(logits_argmax[0])
        print(f"++++++++++++++++++++++++++++++++++++++++++++++++++Person is performing: {act_text}")
        print(f"++++++++++++++++++++++++++++++++++++++++++++++++++Person is performing: {act_text}")
        print(f"++++++++++++++++++++++++++++++++++++++++++++++++++Person is performing: {act_text}")
        print(f"++++++++++++++++++++++++++++++++++++++++++++++++++Person is performing: {act_text}")

        # ------------------------------------------------------------------
        # 1.  collect per-frame tensors in *bulk* to stay off the Python fast-path
        #     (converting inside the inner loop kills performance)

        # (a) logits

        # (b) build per-frame containers
        frames_grp_data = []        # list of dicts, one per frame

        for img_idx, (_, graph) in enumerate(self.sequence):

            if getattr(graph["object"], "x", None) is None:
                print(f"Warning: graph['object'].x is None at frame {img_idx}  emppty graph")
                # make an empty graph
                frames_grp_data.append({
                    "features": np.empty((0, 0), dtype=np.float16),
                    "pos":     np.empty((0, 3), dtype=np.float16),
                    "edges":   np.empty((2, 0), dtype=np.int32),
                    "edge_lbl": np.empty((0,), dtype=np.float32),
                    "labels":  np.empty((0,), dtype=np.int32)
                })
                continue

            # --- features ----------------------------------------------------------
            feat_np = graph["object"].x.cpu().numpy().astype(np.float16)   # (N, D)

            # -- position (optional) ---------------------------------------------
            pos_np = graph["object"].pos.cpu().numpy().astype(np.float16)  # (N, 3) or (N, 2)
            labels_np = graph["object"].labels.cpu().numpy()  # (N,)


            # --- relations ---------------------------------------------------------
            rel_src, rel_dst, rel_lbl = [], [], []
            for edge_type in graph.edge_types:
                ei  = graph[edge_type].edge_index
                lbl = graph[edge_type].edge_attr

                rel_src.extend(ei[0].cpu().numpy())
                rel_dst.extend(ei[1].cpu().numpy())
                rel_lbl.extend(lbl.cpu().numpy())          # float / scalar

            frames_grp_data.append({
                "features": feat_np,
                "pos":     pos_np,                        # (N, 3) or (N, 2)
                "edges":    np.vstack([rel_src, rel_dst]).astype(np.int32),   # (2,E)
                "edge_lbl": np.array(rel_lbl, dtype=np.float32),              # (E,)
                "labels": labels_np.astype(np.int32)  # (N,)  int32
            })

        # ------------------------------------------------------------------
        # 2.  write the HDF5 file (gzip-compressed)
        with h5py.File(sample_path, "w") as f:

            # root-level sequence summary (logits)
            f.create_dataset(
                "logits_argmax",
                data        = logits_argmax,
                compression = "gzip",  compression_opts = 6
            )

            # one group per frame
            grp_frames = f.create_group("frames")
            for idx, frame in enumerate(frames_grp_data):
                g = grp_frames.create_group(f"{idx:04d}")

                g.create_dataset("features",
                                data = frame["features"],
                                compression = "gzip", compression_opts = 6)
                
                g.create_dataset("pos",
                                data = frame["pos"],
                                compression = "gzip", compression_opts = 6)
                
                g.create_dataset("labels",
                                data = frame["labels"],
                                compression = "gzip", compression_opts = 6)

                g.create_dataset("edge_index",
                                data = frame["edges"],
                                compression = "gzip", compression_opts = 6)

                g.create_dataset("edge_labels",
                                data = frame["edge_lbl"],
                                compression = "gzip", compression_opts = 6)

        print("✓ HDF5 sample written\n")





if __name__ == "__main__":
    # Example usage
    model = AVIONForwardModule()
    