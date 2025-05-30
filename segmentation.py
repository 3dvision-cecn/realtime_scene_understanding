import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO  # ➜  pip install -U ultralytics

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from transformers import AutoProcessor, AutoModelForVision2Seq

from object import Object


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
        sam_net = build_sam2(cfg.model_cfg, 'conf/' + cfg.model_path, apply_postprocessing=False).to(self.device).eval()
        self.sam = SAM2AutomaticMaskGenerator(
                sam_net,
                points_per_side=16,
                pred_iou_thresh=0.9,
                stability_score_thresh=0.9,
                min_mask_region_area=250,
                points_per_batch=256,
            )
        
        # ────────────── vlM for zero-shot labels ────────────────
        self.vlm = VLM(device=self.device) 




    @torch.no_grad()
    def segment(self, image: np.ndarray, pixel_indexed_pcd: np.ndarray,  timestamp_ms: int = 0, iteration: int = 0):
        """Returns (masks, annotated_img)."""
        # clear previous objects, we are doing it one-shot fashion
        self.objects.clear()

        masks = self.sam.generate(image)

        pcd_segments = []
        for mask in masks:
            segm = mask['segmentation']
            args = np.where(segm)
            pcd_segment = pixel_indexed_pcd[args]
            pcd_segments.append(pcd_segment)
            # radius of a circle that covers the mask
            radius = mask_radius(segm)
            # center of the mask in the original image
            bbox = mask['bbox']
            center = (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2))
            # get the name of the object
            name = self.vlm.ask(image, center, radius)
            print(f"Object: {name}, Radius: {radius}, Center: {center}")
            # calc the centroid of the pcd_segment
            centroid = pcd_segment[:, :3].mean(axis=0)

            obj = Object(name, centroid, bbox, segm, pcd=pcd_segment)
            self.objects.append(obj)

        # create an annotated image
        annotated_img = image.copy()
        for obj in self.objects:
            x, y, w, h = map(int, obj.bbox)
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(annotated_img, obj.name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return self.objects, annotated_img



class VLM:
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.processor = AutoProcessor.from_pretrained('HuggingFaceTB/SmolVLM-Instruct')
        self.model = AutoModelForVision2Seq.from_pretrained(
            'HuggingFaceTB/SmolVLM-Instruct', torch_dtype=torch.bfloat16
        ).to(self.device)

    def ask(self, frame: np.ndarray, center: tuple[int,int], radius: int = 64) -> str:
        x, y = center
        # draw a red circle on the image
        # draw a red circle on a copy of the frame
        circ = frame.copy()
        # conver rgb to bgr
        circ = cv2.cvtColor(circ, cv2.COLOR_RGB2BGR)
        cv2.circle(circ, (x, y), radius, (0, 0, 255), 2)
        # crop the image to the circle
        offset = int(radius + 0)
        x0, y0 = max(x - offset, 0), max(y - offset, 0)
        x1, y1 = min(x + offset, frame.shape[1]), min(y + offset, frame.shape[0])
        circ = circ[y0:y1, x0:x1]
        # uf cropping the image, the circle is not centered return unknown
        if circ.shape[0] != circ.shape[1]:
            # cv2.putText(circ, "unknown", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # cv2.imshow("Circle", circ)
            # cv2.waitKey(100)
            return "unknown"
        # show the image with the circle
        messages = [
            {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is the object in the middle of the red circle, in maximum two words !"},
            ]
            },
        ]
        # convert the image to RGB
        circ = cv2.cvtColor(circ, cv2.COLOR_BGR2RGB)
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[circ], return_tensors="pt").to(self.device)
        # bfloat16 is needed for the model to run on GPU
        with torch.autocast("cuda", dtype=torch.bfloat16):
            generated_ids = self.model.generate(**inputs, max_new_tokens=20)
        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        asssitant = generated_texts[0].split("Assistant:")[-1]
        # for debugging, show the circle with the text
        # cv2.putText(circ, asssitant.strip(), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # cv2.imshow("Circle", circ)
        # cv2.waitKey(0)
        return asssitant.strip()