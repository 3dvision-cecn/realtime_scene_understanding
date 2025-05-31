from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full
from detectron2 import model_zoo
from detectron2.config import get_cfg
from vitpose_model import ViTPoseModel
from detectron2.config import LazyConfig
from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)
GREEN=(0.0, 1.0, 0.0)
import time
import numpy as np
from sklearn.cluster import KMeans

from object import Hand, BothHands

class HandDetection:

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model, self.model_cfg = load_hamer(cfg.checkpoint)
        self.model = self.model.to(self.device)
        self.model.eval()

        if cfg.body_detector == 'vitdet':
            cfg_path = "third_party/hamer/hamer/configs/cascade_mask_rcnn_vitdet_h_75ep.py"
            detectron2_cfg = LazyConfig.load(str(cfg_path))
            detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
            for i in range(3):
                detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
            self.detector = DefaultPredictor_Lazy(detectron2_cfg)
        elif cfg.body_detector == 'regnety':
            detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
            detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
            detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
            self.detector= DefaultPredictor_Lazy(detectron2_cfg)

        self.cpm = ViTPoseModel(self.device)
        self.renderer = Renderer(self.model_cfg, faces=self.model.mano.faces)


    def detect_hands(self, image, pixel_indexed_pcd,timestamp_ms: int = 0):
        """
        Input:
            image rgb: np.ndarray, shape (H, W, 3), dtype=uint8
        Output:

        """
        t0 = time.time()
        # run inference with bf16
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            det_out = self.detector(image)
        t1 = time.time()
        print(f"Body detection took {t1 - t0:.3f} seconds")
        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
        pred_bboxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores=det_instances.scores[valid_idx].cpu().numpy()
        t0 = time.time()
        vitposes_out = self.cpm.predict_pose(
            image,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )
        t1 = time.time()
        print(f"ViTPose took {t1 - t0:.3f} seconds")

        bboxes = []
        is_right = []
        keypoints = []

        # Use hands based on hand keypoint detections
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            # Rejecting not confident detections
            keyp = left_hand_keyp

            valid = keyp[:,2] > 0.4
            if sum(valid) > 3:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                keypoints.append(keyp[valid])
                bboxes.append(bbox)
                is_right.append(0)

            keyp = right_hand_keyp
            valid = keyp[:,2] > 0.4
            if sum(valid) > 3:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                keypoints.append(keyp[valid])
                bboxes.append(bbox)
                is_right.append(1)

        # create for each keypoint a new instance [x, y, is_right]
        points = []
        for i in range(len(keypoints)):
            for j in range(len(keypoints[i])):
                points.append([keypoints[i][j][0], keypoints[i][j][1], is_right[i]])

        # cluster all keypoints into two clusters: and check the majority class
        # Convert points list to numpy array: columns are [x, y, is_right]
        points_arr = np.array(points)
        if len(points_arr) < 2:
            print("Not enough points for clustering.")
            return None, image
        else:
            # Cluster based on the x,y coordinates
            kmeans = KMeans(n_clusters=2, random_state=0).fit(points_arr[:, :2])
            labels = kmeans.labels_

            # Determine the majority class for each cluster based on the is_right flag
            majority_classes = {}
            for cluster in range(2):
                cluster_indices = np.where(labels == cluster)[0]
                if cluster_indices.size:
                    # Count the number of right (1) and left (0) labels
                    rights = points_arr[cluster_indices, 2].astype(int)
                    counts = np.bincount(rights, minlength=2)
                    majority = int(np.argmax(counts))
                    majority_classes[cluster] = majority
                else:
                    majority_classes[cluster] = None

            # Generate new filtered keypoints: only keep keypoints that match the cluster’s majority classification.
            filtered_points = []
            for i, point in enumerate(points_arr):
                cluster = labels[i]
                if majority_classes[cluster] is not None and majority_classes[cluster] == int(point[2]):
                    filtered_points.append(point)
            filtered_points = np.array(filtered_points) if filtered_points else np.empty((0, 3))


        left_hand_keypoints = [ point for point in filtered_points if point[2] == 0 ]
        right_hand_keypoints = [ point for point in filtered_points if point[2] == 1 ]


        # sample the pixels in 2 pixel neighborhood around each keypoint and get the corresponding mean position
        def sample_neighborhood(keypoint, pixel_indexed_pcd, neighborhood_size=2):
            x, y, _ = keypoint
            x, y = int(x), int(y)
            neighborhood = pixel_indexed_pcd[y-neighborhood_size:y+neighborhood_size+1, x-neighborhood_size:x+neighborhood_size+1, :3]
            # get the mean position of the neighborhood
            # flaten the neighborhood to 2D array
            neighborhood_flat = neighborhood.reshape(-1, 3)
            # mean position, 
            mean_pos = np.mean(neighborhood_flat, axis=0)
            return mean_pos

        left_hand_keypoints_pos = np.array([sample_neighborhood(kp, pixel_indexed_pcd) for kp in left_hand_keypoints])
        right_hand_keypoints_pos = np.array([sample_neighborhood(kp, pixel_indexed_pcd) for kp in right_hand_keypoints])

        # Create Hand objects
        left_hand = Hand(
            keypoints_image=left_hand_keypoints,
            keypoints_pcd=left_hand_keypoints_pos,
            mean_pos=np.mean(left_hand_keypoints_pos, axis=0)
        )

        right_hand = Hand(
            keypoints_image=right_hand_keypoints,
            keypoints_pcd=right_hand_keypoints_pos,
            mean_pos=np.mean(right_hand_keypoints_pos, axis=0)
        )
        # Create BothHands object
        both_hands = BothHands(left_hand=left_hand, right_hand=right_hand)

        # visualize the keypoints on the image
        image = image.copy()
        for point in filtered_points:
            x, y, is_right = point
            if is_right == 1:
                cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)
            else:
                cv2.circle(image, (int(x), int(y)), 5, (255, 0, 0), -1)

        # sample the pixels in 2 pixel neightborhood around each keypoint and get the corresponding mean pos

        return both_hands, image

