import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO  # ➜  pip install -U ultralytics (v8 or v9)

import clip
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor  # <-- correct import

from typing import List, Dict, Any


class Segmentation:
    """Fast detector‑prompted **SAM‑2** segmentation with optional CLIP zero‑shot labels.

    1. **YOLO** at lower resolution → coarse boxes + class names.
    2. **SAM‑2** (`SAM2ImagePredictor`) refines each box to a pixel‑accurate mask.
    3. (Optional) **CLIP** overrides class if detector confidence is low.
    """

    def __init__(
        self,
        cfg,
        *,
        detector_weights: str | Path | None = None,
        imgsz: int = 640,
        conf: float = 0.35,
        iou: float = 0.6,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        zero_shot: bool = True,
        vocab_path: str | None = None,
    ):
        self.cfg = cfg
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.zero_shot = zero_shot

        # ────────────── 1️⃣ YOLO detector ───────────────────────────────
        weights = detector_weights or cfg.get("detector_weights", "yolov9c.pt")
        self.det = YOLO(weights).to(device)
        self.det_names = self.det.names

        # ────────────── 2️⃣ SAM‑2 predictor (box prompt) ────────────────
        sam_net = build_sam2(cfg.model_cfg, 'conf/'+cfg.model_path).to(device).eval()
        self.sam = SAM2ImagePredictor(sam_net)
        self.sam.mask_threshold = 0.0  # we’ll binarise manually

        # ────────────── 3️⃣ CLIP zero‑shot head (optional) ──────────────
        if self.zero_shot:
            self.clip_model, self.clip_preprocess = clip.load("ViT-L/14@336px", device=device)
            self.clip_model.eval()

            self.vocab = Path(vocab_path).read_text().splitlines() if vocab_path else [
                "hand", "bottle", "cup", "desk", "box", "chair", "wooden table", "sink",
                "water glass", "bag", "spoon", "fork", "knife", "plate", "bowl", "food",
            ]
            with torch.no_grad():
                txt = clip.tokenize([f"a photo of a {v}" for v in self.vocab]).to(device)
                self.text_emb = self.clip_model.encode_text(txt)
                self.text_emb /= self.text_emb.norm(dim=-1, keepdim=True)

    # ──────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def segment(self, image: np.ndarray, timestamp_ms: int = 0):
        """Returns (masks, annotated_img)."""
        # 🚦 YOLO detection
        det_res = self.det(image, imgsz=self.imgsz, conf=self.conf, iou=self.iou, verbose=False)[0]
        if len(det_res.boxes) == 0:
            return [], image  # nothing

        boxes = det_res.boxes.xyxy.int().tolist()
        cls_ids = det_res.boxes.cls.int().tolist()
        confs = det_res.boxes.conf.tolist()

        # 🖌️ SAM‑2 segmentation
        self.sam.set_image(image)
        masks: List[Dict[str, Any]] = []
        for (x0, y0, x1, y1), cid, score in zip(boxes, cls_ids, confs):
            m_np, _, _ = self.sam.predict(box=np.array([x0, y0, x1, y1]), multimask_output=False)
            mask_bool = m_np[0].astype(bool)  # first (only) mask
            masks.append({
                "segmentation": mask_bool,
                "bbox": (x0, y0, x1, y1),
                "label": self.det_names[cid],
                "prob": float(score),
            })

        # 🔤 Optional CLIP relabel
        if self.zero_shot:
            masks = self._clip_label(image, masks)

        annotated = self._draw_masks_on_image(image.copy(), masks)
        return masks, annotated

    # ──────────────────────────────────────────────────────────────────
    def _clip_label(self, img: np.ndarray, masks: List[Dict[str, Any]]):
        crops, keep = [], []
        for m in masks:
            x0, y0, x1, y1 = m["bbox"]
            crops.append(self.clip_preprocess(Image.fromarray(img[y0:y1, x0:x1])))
            keep.append(m)
        if not crops:
            return masks
        batch = torch.stack(crops).to(self.device)
        with torch.no_grad():
            img_emb = self.clip_model.encode_image(batch)
            img_emb /= img_emb.norm(dim=-1, keepdim=True)
            probs = (img_emb @ self.text_emb.T).softmax(-1)
        for m, p in zip(keep, probs):
            idx = int(p.argmax())
            if p[idx] > 0.5:
                m["label"] = self.vocab[idx]
                m["prob"] = float(p[idx])
        return masks

    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _draw_masks_on_image(im: np.ndarray, masks: List[Dict[str, Any]]):
        overlay = im.copy()
        for m in masks:
            seg = m["segmentation"].astype(np.uint8)
            colour = tuple(int(c) for c in np.random.randint(0, 255, 3))
            contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, colour, cv2.FILLED)
            ys, xs = np.where(seg)
            if ys.size:
                cx, cy = int(xs.mean()), int(ys.mean())
                cv2.putText(overlay, f"{m['label']} {m['prob']*100:.0f}%", (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.5, im, 0.5, 0, dst=im)
        return im
