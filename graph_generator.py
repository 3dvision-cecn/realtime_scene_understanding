# graph_generator.py
from __future__ import annotations

import networkx as nx                 # pip install networkx
import numpy as np
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import os
from pathlib import Path
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class GraphGenerator:
    """
    Build a simple scene-graph from:
        • SAM-2 masks      (list[dict] with "segmentation", "label", … )
        • Hand landmarks   (list[(x,y,z)] per hand from MediaPipe)
    """

    def __init__(
        self,
        cfg,
    ):
        overlap_thresh: float = 0.15,      # α  –  holding if ≥15 % overlap
        near_thresh: float = 80.0,         # δ  –  pixels for 'near'
        self.alpha = 0.15 #overlap_thresh
        self.delta = 80.0 #near_thresh

    # ------------------------------------------------------------------ #
    def _centroid(self, mask: np.ndarray) -> Tuple[float, float]:
        """Return (cx, cy) of a 2-D boolean mask."""
        xs, ys = np.where(mask)
        return float(xs.mean()), float(ys.mean())

    # ------------------------------------------------------------------ #
    def generate_graph(
        self,
        image: np.ndarray,
        masks: List[Dict[str, Any]],
        hand_data: dict,
        #hands: List[List[Tuple[float, float, float]]],
    ) -> nx.Graph:
        """
        Parameters
        ----------
        image : RGB frame (unused except for width/height)
        masks : SAM-2 masks, each with `"segmentation"` and `"label"`
        hand_landmarks : list of 21-point lists (MediaPipe order) per hand

        Returns
        -------
        networkx.Graph  –  nodes with `type`, `label`, `centroid`;
                           edges with `relation`
        """
        G = nx.Graph()

        #hand_landmarks = hands.hand_landmarks

        # 1) add object nodes --------------------------------------------
        for i, m in enumerate(masks):
            node_id = f"obj_{i}"
            G.add_node(
                node_id,
                type="object",
                label=m.get("label", f"mask_{i}"),
                prob=m.get("prob", 0.0),
                centroid=self._centroid(m["segmentation"]),
            )
            m["node"] = node_id   # keep reference for later
        H, W = image.shape[:2]
        # 2) add hand nodes ----------------------------------------------
        '''
        for hidx, hand in enumerate(hand_landmarks):
            cx = [landmark.x * W for landmark in hand]
            cy = [landmark.y * H for landmark in hand]
            node_id = f"hand_{hidx}"

            cx = float(np.mean(cx))
            cy = float(np.mean(cy))
            G.add_node(node_id, type="hand", centroid=(cx, cy))
        '''
        for hidx, center in enumerate(hand_data["box_centers"]):
            #verts_2d = verts[:, :2]  # drop z
            cx, cy = center #verts_2d.mean(axis=0)

            node_id = f"hand_{hidx}"
            G.add_node(
                node_id,
                type="hand",
                centroid=(cx, cy),
                cam_t=hand_data["cam_t"][hidx], #.tolist(),     # optional: for future 3D use
                is_right=bool(hand_data["is_right"][hidx])   # optional: left/right hand info
            )        
        # 3) holding edges -----------------------------------------------
        for m in masks:
            obj_node = m["node"]
            seg = m["segmentation"]
            obj_area = seg.sum()
            for hidx, verts in enumerate(hand_data["verts"]):
                verts_2d = verts[:, :2]
                xs = verts_2d[:, 0]
                ys = verts_2d[:, 1]

                x0, x1 = int(xs.min()), int(xs.max())
                y0, y1 = int(ys.min()), int(ys.max())

                # Clamp to image bounds
                x0 = max(x0, 0)
                y0 = max(y0, 0)
                x1 = min(x1, seg.shape[1] - 1)
                y1 = min(y1, seg.shape[0] - 1)

                # Create a hand region mask
                hand_mask = np.zeros_like(seg, dtype=bool)
                hand_mask[y0:y1 + 1, x0:x1 + 1] = True

                overlap = np.logical_and(seg, hand_mask).sum() / obj_area

                if overlap >= self.alpha:
                    G.add_edge(f"hand_{hidx}", obj_node, relation="holding")

            '''
            for hidx, hand in enumerate(hand_landmarks):
                # build a small binary mask for hand bbox
                xs = [landmark.x * W for landmark in hand]
                ys = [landmark.y * H for landmark in hand]
                x0, x1 = int(min(xs)), int(max(xs))
                y0, y1 = int(min(ys)), int(max(ys))
                hand_mask = np.zeros_like(seg, dtype=bool)
                hand_mask[y0:y1 + 1, x0:x1 + 1] = True

                overlap = np.logical_and(seg, hand_mask).sum() / obj_area
                if overlap >= self.alpha:
                    G.add_edge(f"hand_{hidx}", obj_node, relation="holding")
            '''

        # 4) near edges (objects↔objects, hands↔objects) ------------------
        nodes = list(G.nodes(data=True))
        for i, (u, d1) in enumerate(nodes):
            for v, d2 in nodes[i + 1:]:
                dist = np.hypot(
                    d1["centroid"][0] - d2["centroid"][0],
                    d1["centroid"][1] - d2["centroid"][1],
                )
                if dist <= self.delta:
                    G.add_edge(u, v, relation="near", distance=dist)

        # draw_graph(G)  # for debugging
        graph_img = self.draw_graph(G)  # for debugging

        return G, graph_img
    # ------------------------------------------------------------------ #


    def draw_graph(self, G: nx.Graph) -> np.ndarray:
        pos = {n: d["centroid"] for n, d in G.nodes(data=True)}
        node_colors = [
            "#FFD447" if G.nodes[n]["type"] == "hand" else "#4682B4"
            for n in G.nodes
        ]
        labels = nx.get_node_attributes(G, "label")

        # Set your fixed image resolution here
        W, H = 1920, 1080

        fig = plt.figure(figsize=(W / 150, H / 150), dpi=150)  # Keep scaling consistent
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')

        # Match image orientation
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)  # critical: invert y-axis
        ax.axis("off")

        nx.draw(
            G,
            pos=pos,
            labels=labels,
            with_labels=True,
            ax=ax,
            node_color=node_colors,
            node_size=1000,
            font_size=10,
            font_color="black",
            edge_color="#555555",
            width=3.0,
        )

        edge_labels = {
            (u, v): d["relation"]
            if d["relation"] != "near"
            else f"near\n{int(d['distance']):d}px"
            for u, v, d in G.edges(data=True)
        }
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7, ax=ax)

        # Convert canvas → RGB image
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        h, w = int(renderer.height), int(renderer.width)

        rgba = np.frombuffer(renderer.buffer_rgba(), dtype=np.uint8)
        rgb = rgba.reshape(h, w, 4)[..., :3].copy()
        plt.close(fig)
        return rgb
