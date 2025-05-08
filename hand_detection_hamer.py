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


    def detect_hands(self, image, timestamp_ms: int = 0):
        """
        Input:
            image rgb: np.ndarray, shape (H, W, 3), dtype=uint8
        Output:

        """
        det_out = self.detector(image)
        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
        pred_bboxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores=det_instances.scores[valid_idx].cpu().numpy()
        vitposes_out = self.cpm.predict_pose(
            image,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        bboxes = []
        is_right = []

        # Use hands based on hand keypoint detections
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            # Rejecting not confident detections
            keyp = left_hand_keyp
            valid = keyp[:,2] > 0.6
            if sum(valid) > 3:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                found_left = False
                for x in is_right:
                    if x == 0:
                        found_left = True
                if not found_left:
                    bboxes.append(bbox)
                    is_right.append(0)
            keyp = right_hand_keyp
            valid = keyp[:,2] > 0.6
            if sum(valid) > 3:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                found_right = False
                for x in is_right:
                    if x == 1:
                        found_right = True
                if not found_right:
                    bboxes.append(bbox)
                    is_right.append(1)

            



        if len(bboxes) != 0:
            boxes = np.stack(bboxes)
            right = np.stack(is_right)

            dataset = ViTDetDataset(self.model_cfg, image, boxes, right, rescale_factor=2.0)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

            all_cam_t = []
            all_verts = []
            all_right = []

            for batch in dataloader:
                batch = recursive_to(batch, self.device)
                with torch.no_grad():
                    out = self.model(batch)

                multiplier = (2*batch['right']-1)
                pred_cam = out['pred_cam']
                pred_cam[:,1] = multiplier*pred_cam[:,1]
                box_center = batch["box_center"].float()
                box_size = batch["box_size"].float()
                img_size = batch["img_size"].float()
                multiplier = (2*batch['right']-1)
                scaled_focal_length = self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE * img_size.max()
                pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

                batch_size = batch['img'].shape[0]
                for n in range(batch_size):
                    # Add all verts and cams to list
                    verts = out['pred_vertices'][n].detach().cpu().numpy()
                    is_right = batch['right'][n].cpu().numpy()
                    verts[:,0] = (2*is_right-1)*verts[:,0]
                    cam_t = pred_cam_t_full[n]
                    all_verts.append(verts)
                    all_cam_t.append(cam_t)
                    all_right.append(is_right)

            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            
            cam_view = self.renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)

            # overlaty cam_view[:,:,:3] on image
            # cam_view: HxWx4 RGBA float image in [0,1]
            # Convert to 0–255 uint8
            overlay = (cam_view * 255).astype(np.uint8)
            # Split channels
            rgb, alpha = overlay[:, :, :3], overlay[:, :, 3] / 255.0

            # Prepare output
            annotated_image = image.copy().astype(np.float32)

            # Broadcast alpha to 3 channels
            alpha_3 = np.stack([alpha]*3, axis=-1)

            # Composite
            annotated_image = (rgb.astype(np.float32) * alpha_3 +
                               annotated_image * (1 - alpha_3))
            annotated_image = annotated_image.astype(np.uint8)

            hand_data = {
                "verts": all_verts,         # list of (778, 3) np.ndarrays
                "cam_t": all_cam_t,         # list of (3,) np.ndarrays
                "is_right": all_right,       # list of bools or ints
                "box_centers": [bc.cpu().numpy().tolist() for bc in box_center],  # ADD THIS
            }
            
            return hand_data, annotated_image

        return None, image

