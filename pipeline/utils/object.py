import uuid
import numpy as np

class Object:
    def __init__(self, name: str, pos: np.ndarray, bbox, mask: np.ndarray, pcd: np.ndarray = None, obb = None, embedding: np.ndarray = None):
        self.unique_id = str(uuid.uuid4())
        self.name = name
        self.position = pos
        self.mask = mask
        self.pcd = pcd
        self.bbox = bbox
        self.obb = obb
        self.embedding = embedding



class Hand:
    def __init__(self,keypoints_image: np.ndarray, keypoints_pcd: np.ndarray, mean_pos: np.ndarray):
        self.keypoints_image = keypoints_image
        self.keypoints_pcd = keypoints_pcd
        self.mean_pos = mean_pos

    def add_embedding(self, embedding: np.ndarray):
        self.embedding = embedding


class BothHands:
    def __init__(self, left_hand: Hand, right_hand: Hand):
        self.left_hand = left_hand
        self.right_hand = right_hand

    