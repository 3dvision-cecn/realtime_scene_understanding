import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO  # ➜  pip install -U ultralytics

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import open3d as o3d

from object import Object
from vlm import VLM
from transformers import AutoImageProcessor, AutoModel


def mask_radius(mask: np.ndarray) -> int:
    """
    Compute the radius of the smallest circle centered at the mask centroid
    that covers all mask pixels.
    """
    # find centroid
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0
    cx, cy = xs.mean(), ys.mean()
    # compute distances from centroid to every mask pixel
    dists = np.sqrt((xs - cx)**2 + (ys - cy)**2) + 10
    # round up to an integer radius
    return int(np.ceil(dists.max()))


def remove_outliers_statistical(pcd: np.ndarray,
                                nb_neighbors: int = 20,
                                std_ratio: float = 2.0) -> np.ndarray:
    """Statistical outlier removal (fast C++ backend)."""
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points  = o3d.utility.Vector3dVector(pcd[:, :3])
    pcd_o3d.colors  = o3d.utility.Vector3dVector(pcd[:, 3:6])

    clean_cloud, _ = pcd_o3d.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
    return np.hstack((np.asarray(clean_cloud.points),
                      np.asarray(clean_cloud.colors)))


class Segmentation:
    """Fast detector‑prompted **SAM‑2** segmentation with custom CLIP zero‑shot labels.

    1. **YOLO** at lower resolution → coarse boxes + class names.
    2. **SAM‑2** (`SAM2ImagePredictor`) refines each box to a pixel‑accurate mask.
    3. **CLIP** uses a custom vocab and prompt logic to (optionally) override labels.
    """

    def __init__(
        self,
        cfg
    ):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.imgsz = cfg.imgsz
        self.conf = cfg.conf
        self.iou = cfg.iou
        self.zero_shot = cfg.zero_shot
        self.vocab_path = cfg.vocab_path

        if self.vocab_path == "":
            self.vocab_path = None

        # this is where we keep all of the detected objects
        self.objects = []

        # cuda stuff
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # ────────────── SAM‑2 predictor (box prompt) ────────────────
        self.sam_net = build_sam2(cfg.model_cfg, 'conf/' + cfg.model_path, apply_postprocessing=False).to(self.device).eval()
        # self.sam = SAM2AutomaticMaskGenerator(
        #         self.sam_net,
        #         points_per_side=32,
        #         pred_iou_thresh=0.9,
        #         stability_score_thresh=0.9,
        #         min_mask_region_area=250,
        #         points_per_batch=256,
        #     )
        # ────────────── YOLOv12 for coarse boxes ────────────────
        self.yolo = YOLO("yolo12x.pt")  
        # ────────────── vlM for zero-shot labels ────────────────
        self.vlm = VLM(device=self.device) 
        # ────────────── Object embedding generator ────────────────
        self.object_embedding_generator = ObjectEmbeddingGenerator()
            




    @torch.no_grad()
    def segment(self, image: np.ndarray, pixel_indexed_pcd: np.ndarray, hand_data, timestamp_ms: int = 0, iteration: int = 0):
        """Returns (masks, annotated_img)."""
        # clear previous objects, we are doing it one-shot fashion
        self.objects.clear()


        # run yolo with ver low confidence to get all of the boxes and show the boxes with opencv
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        yolo_results = self.yolo(image_bgr, conf=self.conf, iou=self.iou, imgsz=self.imgsz)

        points = []
        for result in yolo_results:
            if result.boxes is not None:
                for box in result.boxes:
                    # box.xyxy is a tensor with [x1, y1, x2, y2]
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    # add the center of the box to points
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    # normalize the center to the range [0, 1]
                    center_x = center_x / image.shape[1]
                    center_y = center_y / image.shape[0]
                    points.append((center_x, center_y))

        # convert to numpy array
        points = np.array(points, dtype=np.half)
        print(f"Number of points for SAM-2: {len(points)}")
        sam2_points = []
        sam2_points.append(points)

        # give the list of points to the SAM-2 predictor
        t0 = time.time()
        self.sam = SAM2AutomaticMaskGenerator(
                self.sam_net,
                points_per_side=None,
                point_grids = sam2_points,
                pred_iou_thresh=0.5,
                stability_score_thresh=0.9,
                box_nms_thresh=0.3,
                min_mask_region_area=300,
                points_per_batch=256,
                use_m2m=False,
            )
        # bf16 image
        masks = self.sam.generate(image)
        t1 = time.time()
        print(f"Time taken for SAM-2 segmentation: {t1 - t0:.2f} seconds")

        # # visualize the masks on the image
        annotated_img = image.copy()
        for mask in masks:
            segm = mask['segmentation']
            # generate a random color for the mask overlay
            color = np.random.randint(0, 255, 3).tolist()
            # create an overlay image
            overlay = annotated_img.copy()
            overlay[segm.astype(bool)] = color
            # blend the overlay with the original image
            annotated_img = cv2.addWeighted(overlay, 0.5, annotated_img, 0.5, 0)


        t0 = time.time()
        pcd_segments = []
        for mask in masks:
            segm = mask['segmentation']
            args = np.where(segm)
            pcd_segment = pixel_indexed_pcd[args]
            pcd_segments.append(pcd_segment)
            # radius of a circle that covers the mask
            radius = mask_radius(segm)

            # skip small masks
            if radius < 30:
                # skip small masks
                continue
            # skip very large masks like the background
            if len(pcd_segment) > 400 * 400:
                continue

            # skip masks that contain any hand keypoints
            if hand_data is not None:                    
                left_hand_keypoints = hand_data.left_hand.keypoints_image
                right_hand_keypoints = hand_data.right_hand.keypoints_image

                keypoints = []
                # add all keypoints to the list
                if len(left_hand_keypoints) > 0:
                    keypoints.append(left_hand_keypoints)
                if len(right_hand_keypoints) > 0:
                    keypoints.append(right_hand_keypoints)

                # concatenate the keypoints
                keypoints = np.concatenate(keypoints, axis=0)
                # round the keypoints to the nearest integer
                # but smaller than the image size
                keypoints = np.round(keypoints).astype(int)
                keypoints[:, 0] = np.clip(keypoints[:, 0], 0, image.shape[1] - 1)
                keypoints[:, 1] = np.clip(keypoints[:, 1], 0, image.shape[0] - 1)
                # check if any keypoint is inside the mask
                num_collisions = 0
                for keypoint in keypoints:
                    if segm[keypoint[1], keypoint[0]]:
                        num_collisions += 1
                if num_collisions > 8:
                    continue

            # find the embeeding for the hands
            if hand_data is not None:
                left_hand_embedding = self.object_embedding_generator.generate_embeddings_hands(image, hand_data.left_hand.keypoints_image)
                right_hand_embedding = self.object_embedding_generator.generate_embeddings_hands(image, hand_data.right_hand.keypoints_image)
                hand_data.left_hand.add_embedding(left_hand_embedding)
                hand_data.right_hand.add_embedding(right_hand_embedding)
                

            # center of the mask in the original image
            bbox = mask['bbox']
            center = (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2))
            # get the name of the object
            # name = self.vlm.ask(image, center, radius)
            name = "unknown"
            # print(f"Object: {name}, Radius: {radius}, Center: {center}")
            # calc the centroid of the pcd_segment

            # remove outliers from the point cloud segment
            pcd_segment = remove_outliers_statistical(pcd_segment, 
                                                      nb_neighbors=20,
                                                      std_ratio=2.0)
            centroid = pcd_segment[:, :3].mean(axis=0)
            # fit a bounding box to the point cloud segment
            if pcd_segment.shape[0] == 0:
                print(f"Skipping empty point cloud segment for object: {name}")
                continue
            # create a bounding box from the point cloud segment
            pcd_o3d = o3d.geometry.PointCloud()
            pcd_o3d.points = o3d.utility.Vector3dVector(pcd_segment[:, :3])

            embedding  = self.object_embedding_generator.generate_embedding(image, segm)
        

            obb = pcd_o3d.get_oriented_bounding_box()

            obj = Object(name, centroid, bbox, segm, pcd=pcd_segment, obb=obb, embedding=embedding)
            self.objects.append(obj)

        t1 = time.time()
        print(f"Time taken for PCD and VLM processing: {t1 - t0:.2f} seconds")

        # create an annotated image
        for obj in self.objects:
            x, y, w, h = map(int, obj.bbox)
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(annotated_img, obj.name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return self.objects, annotated_img, hand_data






class ObjectEmbeddingGenerator:
    """Generates embeddings for objects using a pre-trained model."""

    def __init__(self, model_name: str =  "facebook/dinov2-small", device: str = "cuda"):
        self.device = device    
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)


    def generate_embedding(self, image: np.ndarray, mask: np.ndarray) -> torch.Tensor:
        """Generates an embedding for the given image."""
        # create three diffrent crops of the image
        # 1. Crop a box around the mask and leave all pixels untouched.
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            crop1 = image
        else:
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            crop1 = image[y_min:y_max+1, x_min:x_max+1]

        # 2. Crop a box around the mask and fill all pixels outside the mask with black.
        if len(xs) == 0 or len(ys) == 0:
            crop2 = image.copy()
        else:
            crop2 = image[y_min:y_max+1, x_min:x_max+1].copy()
            mask_crop = mask[y_min:y_max+1, x_min:x_max+1]
            crop2[mask_crop == 0] = 0

        # 3. Crop a larger box around the mask and leave all pixels untouched.
        if len(xs) == 0 or len(ys) == 0:
            crop3 = image
        else:
            h, w, _ = image.shape
            pad_x = int((x_max - x_min) * 0.4)
            pad_y = int((y_max - y_min) * 0.4)
            x_min_ext = max(x_min - pad_x, 0)
            y_min_ext = max(y_min - pad_y, 0)
            x_max_ext = min(x_max + pad_x, w - 1)
            y_max_ext = min(y_max + pad_y, h - 1)
            crop3 = image[y_min_ext:y_max_ext+1, x_min_ext:x_max_ext+1]

        # visualize the cropsq
        # crop1_bgr = cv2.cvtColor(crop1, cv2.COLOR_RGB2BGR)
        # crop2_bgr = cv2.cvtColor(crop2, cv2.COLOR_RGB2BGR)
        # crop3_bgr = cv2.cvtColor(crop3, cv2.COLOR_RGB2BGR)
        # cv2.imshow("Crop 1", crop1_bgr)
        # cv2.imshow("Crop 2", crop2_bgr)
        # cv2.imshow("Crop 3", crop3_bgr)
        # cv2.waitKey(1000000)

        images = [crop1, crop2, crop3]

        input = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**input)
            embedding = outputs.last_hidden_state

        avg_embedding = embedding.mean(dim=0).mean(dim=0)
        return avg_embedding.cpu().numpy()
    

    def generate_embeddings_hands(self, image: np.ndarray, hand_keypoints: list) -> torch.Tensor:
        """Generates embeddings for the hands in the image."""
        if hand_keypoints is None or len(hand_keypoints) == 0:
            return None

        hand_keypoints = np.array(hand_keypoints, dtype=np.float32)

        # find the bounding box of the hand keypoints and add some padding
        x_min = int(hand_keypoints[:, 0].min())
        x_max = int(hand_keypoints[:, 0].max())
        y_min = int(hand_keypoints[:, 1].min())
        y_max = int(hand_keypoints[:, 1].max())

        pad_x = int((x_max - x_min) * 0.4)
        pad_y = int((y_max - y_min) * 0.4)

        x_min = max(x_min - pad_x, 0)
        y_min = max(y_min - pad_y, 0)
        x_max = min(x_max + pad_x, image.shape[1] - 1)
        y_max = min(y_max + pad_y, image.shape[0] - 1)

        # check the size of the crop
        if (x_max - x_min) < 10 or (y_max - y_min) < 10:
            print("Hand crop is too small, skipping embedding generation.")
            return None

        hand_crop = image[y_min:y_max+1, x_min:x_max+1]

        # visualize the hand crop
        # cv2.imshow("Hand Crop", hand_crop)
        # cv2.waitKey(1000)


        # # visalize the hand crop
        # cv2.imshow("Hand Crop", hand_crop)
        # cv2.waitKey(1000)

        input_data = self.processor(images=[hand_crop], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**input_data)
        # average the features across token sequence dimension
        embedding = outputs.last_hidden_state.squeeze(0).mean(dim=0)
        return embedding.cpu().numpy().squeeze()