import uuid
import numpy as np

class Object:
    def __init__(self, name: str, pos: np.ndarray, bbox, mask: np.ndarray, pcd: np.ndarray = None):
        self.unique_id = str(uuid.uuid4())
        self.name = name
        self.position = pos
        self.mask = mask
        self.pcd = pcd
        self.bbox = bbox
