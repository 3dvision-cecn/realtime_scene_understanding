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



class TrainingGenerator():

    def __init__(self, cfg):
        self.cfg = cfg

        self.avion = AVIONForwardModule()

        self.sequence = []

        # create a new directory for the training samples
        root_dir = "graph_samples"
        # add date and time to the directory name

        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.sample_dir = f"{root_dir}/{now}"
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
        # ------------------------------------------------------------------
        # 1.  collect per-frame tensors in *bulk* to stay off the Python fast-path
        #     (converting inside the inner loop kills performance)

        # (a) logits
        logits_argmax = logits_margmax.cpu().numpy().astype(np.int16)   # (T,) → smaller dtype

        # (b) build per-frame containers
        frames_grp_data = []        # list of dicts, one per frame

        for img_idx, (_, graph) in enumerate(self.sequence):

            if getattr(graph["object"], "x", None) is None:
                print(f"Warning: graph['object'].x is None at frame {img_idx} – skipped")
                continue

            # --- features ----------------------------------------------------------
            feat_np = graph["object"].x.cpu().numpy().astype(np.float16)   # (N, D)

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
                "edges":    np.vstack([rel_src, rel_dst]).astype(np.int32),   # (2,E)
                "edge_lbl": np.array(rel_lbl, dtype=np.float32),              # (E,)
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

                g.create_dataset("edge_index",
                                data = frame["edges"],
                                compression = "gzip", compression_opts = 6)

                g.create_dataset("edge_labels",
                                data = frame["edge_lbl"],
                                compression = "gzip", compression_opts = 6)

        print("✓ HDF5 sample written\n")


class AVIONForwardModule(nn.Module):
    """
    Wraps preprocessing + AVION classifier
    so it can be TorchScripted and reused.
    Expects a float32 tensor shaped (T, H, W, C) in BGR.
    Returns (argmax, softmax).
    """
    def __init__(self):
        super().__init__()
        self.backbone = self.initialize_backbone()
        self.crop_size = 224

        # register mean / std as buffers so they are part of the .pt file
        mean = torch.tensor([108.3272985, 116.7460125, 104.09373615])
        std  = torch.tensor([68.5005327, 66.6321579, 70.32316305])
        self.register_buffer("mean", mean)
        self.register_buffer("std",  std)


    """
    initializes the backbone model
    """
    def initialize_backbone(self, pretrain_path="conf/checkpoints/avion/avion_pretrain_lavila_vitb_best.pt", fine_tune_path="conf/checkpoints/avion/avion_finetune_cls_lavila_vitb_best.pt"):

        ckpt = torch.load(pretrain_path, map_location='cpu')
        state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            state_dict[k.replace('module.', '')] = v

        old_args = ckpt['args']
        print("=> creating model: {}".format(old_args.model))

        model = getattr(model_clip, "CLIP_VITB16")(
            freeze_temperature=True,
            use_grad_checkpointing=False,
            context_length=old_args.context_length,
            vocab_size=old_args.vocab_size,
            patch_dropout=0,
            num_frames=16,
            drop_path_rate=0.1,
            use_fast_conv1=True,
            use_flash_attn=True,
            use_quick_gelu=True,
            project_embed_dim=old_args.project_embed_dim,
            pretrain_zoo=old_args.pretrain_zoo,
            pretrain_path=old_args.pretrain_path,
        )

        model.logit_scale.requires_grad = False

        state_dict = inflate_positional_embeds(
            model.state_dict(), state_dict,
            num_frames=16,
            load_temporal_fix='bilinear',
        )
        model.load_state_dict(state_dict, strict=True)

        model = model_clip.VideoClassifier(
            model.visual,
            dropout=0.0,
            num_classes=3806
        )
        model = model.cuda()

        # Load finetuning checkpoint correctly
        checkpoint = torch.load(fine_tune_path, map_location='cpu')
        print("Checkpoint keys:", list(checkpoint.keys()))

        # Fix module prefix issue in checkpoint
        if 'state_dict' in checkpoint:
            state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                # Remove the 'module.' prefix from keys
                name = k.replace('module.', '')
                state_dict[name] = v
            
            # Now load the fixed state dict
            result = model.load_state_dict(state_dict, strict=False)
            print("Loaded model weights:", result)
        else:
            print("Error: Checkpoint doesn't contain 'state_dict' key")

        model.eval()
        # AFTER loading weights, convert to bfloat16
        model = model.to(torch.bfloat16)

        return model


    def forward(self, frames: np.ndarray):                # (T,H,W,C)  BGR
        """
        * frames : (T, H, W, C)  BGR  float32
        * returns: (argmax, softmax)   int64, float32
        """     
        frames = torch.tensor(frames, dtype=torch.bfloat16)
        # --- preprocessing --------------------------------------------------
        # 1. BGR → RGB   and   THWC → TCHW
        frames = frames[:, :, :, [2, 1, 0]].permute(0, 3, 1, 2)

        # 2. resize so the shorter side == crop_size
        t, c, h, w = frames.shape
        scale      = self.crop_size / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        frames = F.interpolate(frames, size=(new_h, new_w),
                               mode="bilinear", align_corners=False)

        # 3. centre-crop
        sh = (new_h - self.crop_size) // 2
        sw = (new_w - self.crop_size) // 2
        frames = frames[:, :, sh : sh + self.crop_size,
                               sw : sw + self.crop_size]

        # 4. normalise exactly as in training (0-255 range!)
        mean = self.mean.view(1, 3, 1, 1)
        std  = self.std.view(1, 3, 1, 1)
        frames = (frames - mean) / std

        # 5. TCHW → CTHW, add batch dim, move to CUDA
        frames = frames.permute(1, 0, 2, 3).unsqueeze(0).to("cuda").to(torch.bfloat16)

        # 6. imitate the autocast in your update() loop:
        #    keep the graph JIT-safe by *explicitly* casting to BF16
        # --------------------------------------------------------------------
        with torch.no_grad():                     # same as update()
            logits = self.backbone(frames)

        # logits are BF16 – do softmax in FP32 for numerical stability
        probs = torch.softmax(logits.float(), dim=1)
        return logits.argmax(1), probs
    




if __name__ == "__main__":
    # Example usage
    model = AVIONForwardModule()
    