import sam2
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import cv2
import numpy as np
import torch
import clip
from PIL import Image

from typing import List, Dict, Any



class Segmentation:
    """
    SAM-2 mask generator + CLIP zero-shot labeller.

    Each mask dict returned by `segment` now has:
        • "segmentation" – H×W boolean mask  (unchanged)
        • "area", "bbox", "score"            (unchanged, from SAM2)
        • **"label"**   – best open-vocab description string
        • "prob"        – CLIP softmax probability of that label
    """

    def __init__(
        self,
        cfg,
        *,
        vocab_path: str | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.cfg = cfg

        # ───────────────── SAM-2 ──────────────────────────────────────────
        self.model = build_sam2(
            cfg.model_cfg, cfg.model_path, apply_postprocessing=False
        )
        self.mask_generator = SAM2AutomaticMaskGenerator(self.model, points_per_side=32, points_per_batch=256)

        # ───────────────── CLIP zero-shot head ────────────────────────────
        self.device = device
        self.clip_model, self.clip_preprocess = clip.load("ViT-L/14@336px", device=device)
        self.clip_model.eval()

        # vocabulary: one label per line
        if vocab_path is None:
            # fallback: dummy Cityscapes-style palette (edit as needed)
            self.vocab: List[str] = [
                "hand",
                "bottle",
                "cup",
                "desk",
                "box",
                "chair",
                "wooden table",
                "sink",
                "water glass",
                "bag",
                "spoon",
                "fork",
                "knife",
                "plate",
                "bowl",
                "food",
                "plastic cutting board",
                "paper towel",
                "plastic bag",
                "plastic container",
                "plastic wrap",
                "plastic cup",
                "plastic bottle",
            ]
        else:
            self.vocab = Path(vocab_path).read_text().splitlines()

        # pre-compute text embeddings
        SELF_PROMPTS = [
                 "a close-up of a {}", 
        ]
        ...
        # inside __init__
        tokens = [clip.tokenize([t.format(v) for v in self.vocab]).to(device)
                for t in SELF_PROMPTS]
        with torch.no_grad():
            emb = [self.clip_model.encode_text(t) for t in tokens]
            self.text_emb = torch.stack([e/e.norm(dim=-1, keepdim=True) for e in emb]).mean(0)
            self.text_emb /= self.text_emb.norm(dim=-1, keepdim=True)
    # ──────────────────────────────────────────────────────────────────────
    def segment(self, image: np.ndarray, timestamp_ms: int = 0):
        """
        Parameters
        ----------
        image : np.ndarray  (H,W,3)  RGB uint8
        Returns
        -------
        masks : List[dict]   – each with extra "label" & "prob"
        annotated_image : np.ndarray with coloured masks + text labels
        """
        masks: List[Dict[str, Any]] = self.mask_generator.generate(image)

        # CLIP requires float RGB in [0,1]
        filtered_m = []
        for m in masks:
            seg = m["segmentation"]

            # tight bounding-box (with 1-px padding)
            ys, xs = np.where(seg)
            if ys.size == 0:           # safety (degenerate mask)
                continue
            # add padding around the mask bounding-box
            offset = 10  # pixels of border
            y0 = max(int(ys.min()) - offset, 0)
            y1 = min(int(ys.max()) + offset, image.shape[0])
            x0 = max(int(xs.min()) - offset, 0)
            x1 = min(int(xs.max()) + offset, image.shape[1])

            # crop & *mask* it: everything outside the region → 0  (black)
            crop_rgb = image[y0:y1, x0:x1].copy()              # (Hc,Wc,3)
            crop_mask = seg[y0:y1, x0:x1]
            # crop_rgb[~crop_mask] = 0                           # apply the mask

            # debug: display the cropped & masked region for inspection
            # convert RGB to BGR for proper OpenCV display
            crop_rgb = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
            # cv2.imshow("Cropped Masked Region", crop_rgb_vis)
            # cv2.waitKey(1000)  # wait for a key press (1 ms)

            # CLIP preprocessing expects RGB PIL.Image
            pil = Image.fromarray(crop_rgb)
            inp = self.clip_preprocess(pil).unsqueeze(0).to(self.device)

            with torch.no_grad():
                img_emb = self.clip_model.encode_image(inp)
                img_emb /= img_emb.norm(dim=-1, keepdim=True)
                logits = 100.0 * img_emb @ self.text_emb.T
                probs = logits.softmax(dim=-1)[0]

            best_idx = int(probs.argmax())
            best_prob = float(probs[best_idx])
            best_label = self.vocab[best_idx] if best_prob >= 0.5 else "unknown"

            # if the predicted label is "hand", override to "unknown"
            if best_label == "hand":
                best_label = "unknown"

            

            m["label"] = best_label
            m["prob"] = best_prob

            if best_label != "unknown":
                filtered_m.append(m)


        masks = filtered_m

        annotated = self.draw_masks_on_image(image, masks)
        return masks, annotated

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def draw_masks_on_image(image, masks):
            """
            Colour-fills every mask and writes "<label> (prob%)" at its centroid.
            """
            overlay = image.copy()

            for m in masks:
                mask = (m["segmentation"] > 0).astype(np.uint8)  # binary H×W
                colour = tuple(int(c) for c in np.random.randint(0, 255, size=3))

                if m["label"] == "unknown":
                    continue

                # fill region
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, colour, thickness=cv2.FILLED)

                # ---- text position = centroid of the mask --------------------
                M = cv2.moments(mask)
                if M["m00"] == 0:          # degenerate (shouldn’t happen)
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                text = f"{m['label']}  {m['prob']*100:.0f}%"
                cv2.putText(
                    overlay, text, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                    1, cv2.LINE_AA, bottomLeftOrigin=False
                )

            # alpha-blend with original image
            cv2.addWeighted(overlay, 0.5, image, 0.5, 0, dst=image)
            return image
