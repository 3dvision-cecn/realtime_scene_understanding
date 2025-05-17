import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

import clip
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from typing import List, Dict, Any


class Segmentation:
    """Fast detector‑prompted **SAM‑2** segmentation with custom CLIP zero‑shot labels,
    enhanced with ByteTrack for persistent object tracking and CLIP caching.
    """

    def __init__(
        self,
        cfg  # This would be cfg.segmentation from your main script
    ):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.imgsz = cfg.imgsz
        self.conf = cfg.conf  # Confidence threshold for YOLO detections
        self.iou = cfg.iou    # IoU threshold for NMS
        self.zero_shot = cfg.zero_shot
        self.vocab_path = cfg.vocab_path

        # Cache for CLIP predictions per track_id
        self.clip_cache = {}
        self.clip_display_threshold = cfg.get("clip_threshold", 0.5) # For displaying a CLIP label
        self.clip_cache_threshold = cfg.get("clip_cache_threshold", 0.5) # Min confidence to store in cache

        if self.vocab_path == "":
            self.vocab_path = None

        # ────────────── YOLO detector (will be used for tracking) ────────────────
        weights = cfg.get("detector_weights", "yolov8l.pt") # Example
        self.det = YOLO(weights).to(self.device)
        self.det_names = self.det.names

        # ────────────── SAM‑2 predictor (box prompt) ────────────────
        sam_net = build_sam2(cfg.model_cfg, 'conf/' + cfg.model_path).to(self.device).eval()
        self.sam = SAM2ImagePredictor(sam_net)
        self.sam.mask_threshold = 0.0

        # ────────────── CLIP zero‑shot head ─────────────────────────
        if self.zero_shot:
            self.clip_model, self.clip_preprocess = clip.load("ViT-L/14@336px", device=self.device)
            self.clip_model.eval()
            if self.vocab_path:
                self.vocab = Path(self.vocab_path).read_text().splitlines()
            else:
                self.vocab = [
                    "hand", "arm", "bottle", "cup", "desk", "box", "chair", "couch",
                    "wooden table", "table", "sink", "water glass", "bag", "spoon",
                    "fork", "knife", "plate", "bowl", "food", "fridge", "stove",
                    "pan", "plastic cutting board", "paper towel", "plastic bag",
                    "plastic container", "plastic wrap", "plastic cup", "plastic bottle",
                    "kettle", "oven", "microwave", "toaster", "dishwasher",
                    "cutting board", "groceries", "fruits", "vegetable", "bread",
                    "milk", "trash bin", "lid", "peeler", "wine", "juice",
                    "rice cooker", "cleaning rag", "towel", "door", "cabinet", "spice", "lid"
                ]
            SELF_PROMPTS = ["a close-up of a {}"]
            tokens = []
            for prompt in SELF_PROMPTS:
                texts = [prompt.format(v) for v in self.vocab]
                tokens.append(clip.tokenize(texts).to(self.device))
            with torch.no_grad():
                emb = [self.clip_model.encode_text(t) for t in tokens]
                text_embs = [e / e.norm(dim=-1, keepdim=True) for e in emb]
                self.text_emb = torch.stack(text_embs).mean(0)
                self.text_emb /= self.text_emb.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def segment(self, image: np.ndarray, timestamp_ms: int = 0, iteration: int = 0):
        track_results_list = self.det.track(
            source=image,
            persist=True,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
            tracker="bytetrack.yaml",
        )
        
        if not track_results_list: return [], image
        track_res = track_results_list[0]
        if track_res.boxes is None or len(track_res.boxes) == 0: return [], image

        boxes_xyxy = track_res.boxes.xyxy.int().tolist()
        cls_ids = track_res.boxes.cls.int().tolist()
        confs = track_res.boxes.conf.tolist()
        track_ids = track_res.boxes.id.int().tolist() if track_res.boxes.id is not None else [None] * len(boxes_xyxy)

        self.sam.set_image(image)
        masks: List[Dict[str, Any]] = []
        for i in range(len(boxes_xyxy)):
            x0, y0, x1, y1 = boxes_xyxy[i]
            cid = cls_ids[i]
            score = confs[i]
            tid = track_ids[i]

            m_np, _, _ = self.sam.predict(box=np.array([x0, y0, x1, y1]), multimask_output=False)
            mask_bool = m_np[0].astype(bool)

            mask_data = {
                "segmentation": mask_bool,
                "bbox": (x0, y0, x1, y1),
                "label": self.det_names[cid], # Initial YOLO label
                "prob": float(score),         # YOLO confidence
            }
            if tid is not None:
                mask_data["track_id"] = int(tid)
            
            masks.append(mask_data)

        if self.zero_shot and masks:
            masks = self._clip_label_custom(image, masks)

        annotated_img = self._draw_masks_on_image(image.copy(), masks)
        return masks, annotated_img

    def _clip_label_custom(self, img: np.ndarray, masks_data: List[Dict[str, Any]]):
        processed_masks = []
        for m_data in masks_data:
            tid = m_data.get("track_id")

            # Check cache first
            if tid is not None and tid in self.clip_cache:
                cached_label, cached_prob = self.clip_cache[tid]
                m_data["label"] = cached_label
                m_data["prob"] = cached_prob
                # Use the display threshold for deciding if this cached label is good enough to show
                if cached_label != "unknown" and cached_prob >= self.clip_display_threshold:
                    processed_masks.append(m_data)
                elif m_data["label"] != "unknown": # Fallback to YOLO label if cached is unknown but YOLO wasn't
                    processed_masks.append(m_data)
                # else, if cached is "unknown", we might drop it or try to re-CLIP (more complex)
                continue # Move to next mask

            # If not in cache or no track_id, run CLIP
            seg_mask = m_data["segmentation"]
            ys, xs = np.where(seg_mask)
            if ys.size == 0:
                if m_data["label"] != "unknown": # Keep YOLO if mask is empty but YOLO had a label
                     processed_masks.append(m_data)
                continue
            
            offset = 100
            y0, y1 = max(ys.min() - offset, 0), min(ys.max() + offset, img.shape[0])
            x0, x1 = max(xs.min() - offset, 0), min(xs.max() + offset, img.shape[1])
            crop = img[y0:y1, x0:x1].copy()
            
            if crop.size == 0:
                if m_data["label"] != "unknown": # Keep YOLO if crop is empty
                    processed_masks.append(m_data)
                continue

            pil_image = Image.fromarray(crop)
            clip_input = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)
            
            img_emb = self.clip_model.encode_image(clip_input)
            img_emb /= img_emb.norm(dim=-1, keepdim=True)
            logits = 100.0 * img_emb @ self.text_emb.T
            probs = logits.softmax(dim=-1)[0]
            
            best_idx = int(probs.argmax())
            best_prob = float(probs[best_idx])
            # Use clip_display_threshold to determine if label is "unknown" for current frame
            best_label = self.vocab[best_idx] if best_prob >= self.clip_display_threshold else "unknown"
            
            if best_label == "hand": # Or "arm" if you added it to vocab
                best_label = "unknown" # Or revert to m_data["label"] (YOLO label)

            # Update m_data with new CLIP result for this frame
            m_data["label"] = best_label
            m_data["prob"] = best_prob
            
            # Store in cache if good enough and has a track ID
            if tid is not None and best_label != "unknown" and best_prob >= self.clip_cache_threshold:
                self.clip_cache[tid] = (best_label, best_prob)
            
            if best_label != "unknown":
                processed_masks.append(m_data)
            elif m_data.get("label") != "unknown" and m_data.get("label") != self.det_names[m_data.get("cid_yolo", -1)]:
                # If CLIP is unknown, but original YOLO label was something, keep that one if it's not being overwritten by "unknown"
                # This logic gets a bit tricky: what's the source of truth for "prob" if we revert to YOLO label?
                # For simplicity here, if CLIP is "unknown", we just don't add it unless YOLO had a good label.
                # The provided code already sets m_data["label"] to best_label (CLIP's result).
                # So this `elif` might not be strictly necessary if the `if best_label != "unknown":` is the main filter.
                 pass


        return processed_masks

    @staticmethod
    def _draw_masks_on_image(im: np.ndarray, masks_data: List[Dict[str, Any]]):
        overlay = im.copy()
        for m_data in masks_data:
            # Skip drawing if the final label is "unknown" AND it doesn't have a track ID
            # (meaning it was never a strong, persistent detection).
            # Or, if it has a track_id but the cached/current label is unknown, maybe still draw its box with ID?
            # Current logic: if label is unknown, don't draw text, maybe just the mask?
            # The original code drew "unknown" if track_id was present. Let's stick to that.
            # if m_data["label"] == "unknown" and not m_data.get("track_id"):
            #    continue
            if m_data["label"] == "unknown" and m_data.get("prob", 0) < 0.1: # Heuristic: don't draw very low conf unknowns
                 continue


            seg_mask = m_data["segmentation"].astype(np.uint8)
            tid = m_data.get("track_id")

            if tid is not None:
                colour = ( (tid * 30) % 255, (tid * 50) % 255, (tid * 70) % 255 )
            else: # Should ideally not happen if we only process tracked items for CLIP cache
                colour = tuple(int(c) for c in np.random.randint(0, 255, 3))

            contours, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours: # Only draw if there are contours
                cv2.drawContours(overlay, contours, -1, colour, cv2.FILLED)
            
            ys, xs = np.where(seg_mask)
            if ys.size > 0:
                cx, cy = int(xs.mean()), int(ys.mean())
                label_text = f"{m_data['label']} {m_data['prob']*100:.0f}%"
                if tid is not None:
                    label_text += f" ID:{tid}"
                
                cv2.putText(overlay, label_text, (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(overlay, label_text, (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30,30,30), 1, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.6, im, 0.4, 0, dst=im)
        return im