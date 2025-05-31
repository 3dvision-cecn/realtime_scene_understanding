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

    def __init__(self, cfg):
        print("Initializing GraphGenerator with config:", cfg)


    def generate_graph(self, image, object, hands):
        print("Generating graph from image, object, and hands")
