import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO  # ➜  pip install -U ultralytics

import clip
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from typing import List, Dict, Any


class Segmentation:
    """Fast detector‑prompted **SAM‑2** segmentation with custom CLIP zero‑shot labels.

    1. **YOLO** at lower resolution → coarse boxes + class names.
    2. **SAM‑2** (`SAM2ImagePredictor`) refines each box to a pixel‑accurate mask.
    3. **CLIP** uses a custom vocab and prompt logic to (optionally) override labels.
    """

    def __init__(
        self,
        cfg,
        *,
        detector_weights: str | Path | None = None,
        imgsz: int = 1280,
        conf: float = 0.25,
        iou: float = 0.7,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        zero_shot: bool = True,
        vocab_path: str | None = None,
        debug_dir: str | Path | None = None,       # ← NEW
    ):
        self.cfg = cfg
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.zero_shot = zero_shot

        # ────────────── DEBUG OUTPUT ───────────────────
        if debug_dir is not None:
            self.debug_dir = Path(debug_dir)
            self.debug_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.debug_dir = None

        # ────────────── YOLO detector ───────────────────────────────
        weights = detector_weights or cfg.get("detector_weights", "yolo11l.pt")
        self.det = YOLO(weights).to(device)
        self.det_names = self.det.names

        # ────────────── SAM‑2 predictor (box prompt) ────────────────
        sam_net = build_sam2(cfg.model_cfg, 'conf/' + cfg.model_path).to(device).eval()
        self.sam = SAM2ImagePredictor(sam_net)
        self.sam.mask_threshold = 0.0  # binarise manually

        # ────────────── CLIP zero‑shot head ─────────────────────────
        if self.zero_shot:
            # load CLIP
            self.clip_model, self.clip_preprocess = clip.load("ViT-L/14@336px", device=device)
            self.clip_model.eval()

            # build vocab
            if vocab_path:
                self.vocab = Path(vocab_path).read_text().splitlines()
            else:
                self.vocab = [
                "hand",
                "bottle",
                "cup",
                "desk",
                "box",
                "chair",
                "couch",
                "wooden table",
                "table",
                "sink",
                "water glass",
                "bag",
                "spoon",
                "fork",
                "knife",
                "plate",
                "bowl",
                "food",
                "fridge",
                "stove",
                "pan",
                "plastic cutting board",
                "paper towel",
                "plastic bag",
                "plastic container",
                "plastic wrap",
                "plastic cup",
                "plastic bottle",
                "kettle",
                "oven",
                "microwave",
                "toaster",
                "dishwasher",
                "cutting board",
                "groceries",
                "fruits",
                "vegetable",
                "bread",
                "milk",
                "trash bin",
                "lid",
                "paper towel",
                "peeler",
                "wine",
                "juice",
                "rice cooker",
                "cleaning rag",
                "towel",
                "door",
                "cabinet",
                "spice"
            ]

            # prompts
            SELF_PROMPTS = ["a close-up of a {}"]
            # tokenize and embed text prompts
            tokens = []
            for prompt in SELF_PROMPTS:
                texts = [prompt.format(v) for v in self.vocab]
                tokens.append(clip.tokenize(texts).to(device))
            with torch.no_grad():
                emb = [self.clip_model.encode_text(t) for t in tokens]
                # normalize and average embeddings across prompts
                text_embs = [e / e.norm(dim=-1, keepdim=True) for e in emb]
                self.text_emb = torch.stack(text_embs).mean(0)
                self.text_emb /= self.text_emb.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def segment(self, image: np.ndarray, timestamp_ms: int = 0, iteration: int = 0):
        """Returns (masks, annotated_img)."""
        # YOLO detection
        det_res = self.det(image, imgsz=self.imgsz, conf=self.conf,
                           iou=self.iou, verbose=False)[0]
        if len(det_res.boxes) == 0:
            return [], image

        boxes = det_res.boxes.xyxy.int().tolist()
        cls_ids = det_res.boxes.cls.int().tolist()
        confs = det_res.boxes.conf.tolist()

        # ────────── DEBUG: save YOLO‐only overlay ──────────
        if self.debug_dir is not None:
            img_dbg = image.copy()
            for (x0, y0, x1, y1), cid, score in zip(boxes, cls_ids, confs):
                cv2.rectangle(img_dbg, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.putText(
                    img_dbg,
                    f"{self.det_names[cid]} {score:.2f}",
                    (x0, y0 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
            debug_path = self.debug_dir / f"yolo_{iteration}.jpg"
            cv2.imwrite(str(debug_path), img_dbg, [cv2.IMWRITE_JPEG_QUALITY, 60])


        # SAM‑2 segmentation
        self.sam.set_image(image)
        masks: List[Dict[str, Any]] = []
        for (x0, y0, x1, y1), cid, score in zip(boxes, cls_ids, confs):
            m_np, _, _ = self.sam.predict(box=np.array([x0, y0, x1, y1]), multimask_output=False)
            mask_bool = m_np[0].astype(bool)
            masks.append({
                "segmentation": mask_bool,
                "bbox": (x0, y0, x1, y1),
                "label": self.det_names[cid],
                "prob": float(score),
            })

        # CLIP relabel (original logic)
        if self.zero_shot and masks:
            masks = self._clip_label_custom(image, masks)

        # draw
        annotated = self._draw_masks_on_image(image.copy(), masks)
        return masks, annotated

    def _clip_label_custom(self, img: np.ndarray, masks: List[Dict[str, Any]]):
        filtered = []
        for m in masks:
            seg = m["segmentation"]
            ys, xs = np.where(seg)
            if ys.size == 0:
                continue
            # bounding box crop with padding
            offset = 10
            y0, y1 = max(ys.min() - offset, 0), min(ys.max() + offset, img.shape[0])
            x0, x1 = max(xs.min() - offset, 0), min(xs.max() + offset, img.shape[1])
            crop = img[y0:y1, x0:x1].copy()
            pil = Image.fromarray(crop)
            inp = self.clip_preprocess(pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                img_emb = self.clip_model.encode_image(inp)
                img_emb /= img_emb.norm(dim=-1, keepdim=True)
                logits = 100.0 * img_emb @ self.text_emb.T
                probs = logits.softmax(dim=-1)[0]
            best_idx = int(probs.argmax())
            best_prob = float(probs[best_idx])
            best_label = self.vocab[best_idx] if best_prob >= 0.5 else "unknown"
            if best_label == "hand":
                best_label = "unknown"
            m["label"] = best_label
            m["prob"] = best_prob
            if best_label != "unknown":
                filtered.append(m)
        return filtered

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
