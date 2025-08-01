import numpy as np
import torch
import cv2
import open_clip
from PIL import Image



class ObjectEmbeddingGenerator:
    """Generates object / hand embeddings with an OpenCLIP vision encoder.

    *Interface identical* to the previous RADIO‑L version so the rest of the
    pipeline (segmentation → graph generator) does not change.
    Default backbone = ViT‑B/32 (512‑D output).
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",        # any OpenCLIP vision backbone
        pretrained: str = "openai",          # or "laion2b_s34b_b79k", etc.
        device: str = "cuda",
    ):
        self.device = device

        # OpenCLIP returns: model, tokenizer (unused), preprocess
        full_model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
            precision="bf16" if torch.cuda.is_bf16_supported() else "fp16",
        )
        # only keep the vision tower; it's already contained in model.visual
        #self.model = self.model.visual.eval().to(device)
        #self.tokenizer  = open_clip.get_tokenizer(model_name)   # add this line

        self.visual      = full_model.visual.to(device).eval()        # image path
        self.text_model  = full_model.to(device).eval()               # text path
        self.tokenizer   = open_clip.get_tokenizer(model_name)

    # ------------------------------------------------------------------
    # Object embedding (mask + RGB frame) → 512‑D vector
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_text_embedding(self, text: str) -> np.ndarray:
        tokens = self.tokenizer([text]).to(self.device)
        #feats  = self.model.encode_text(tokens)
        feats_txt  = self.text_model.encode_text(tokens)  # text
        feats_txt  = torch.nn.functional.normalize(feats_txt, dim=-1)
        feats_txt = feats_txt.to(torch.float32)
        return feats_txt.cpu().numpy().squeeze().astype(np.float32)

    @torch.no_grad()
    def generate_embedding(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Three‑crop strategy (same boxes as before) → averaged CLIP embedding."""
        crops = self._three_crops(image, mask)          # list of H×W×3 RGB
        # crops = [self.preprocess(cv2.cvtColor(c, cv2.COLOR_RGB2BGR)) for c in crops]
        crops = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_RGB2BGR)) for c in crops]
        crops = [self.preprocess(p) for p in crops]

        batch = torch.stack(crops).to(self.device).bfloat16()      # (3, 3, 224, 224)

        #feats = self.model(batch)                       # (3, 512)
        feats_img  = self.visual(batch)
        return feats_img.mean(0).float().cpu().numpy().astype(np.float32)              # (512,)

    # ------------------------------------------------------------------
    # Hand embedding (keypoint bbox) – single crop
    # ------------------------------------------------------------------
    @torch.no_grad()
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

        # resize the hand crop to the input size of the model
        #next_support_size = self.model.get_nearest_supported_resolution(hand_crop.shape[0], hand_crop.shape[1])
        #hand_crop = cv2.resize(hand_crop, (next_support_size[1], next_support_size[0]))
        hand_crop = image[y_min:y_max+1, x_min:x_max+1]
        # input_data = self.processor(images=hand_crop, return_tensors="pt", do_resize=False).pixel_values.to(self.device)
        # with torch.no_grad():
        #     outputs, raw = self.model(input_data)
        # # average the features across token sequence dimension
        # embedding = outputs
        # return embedding.cpu().numpy().squeeze()
        hand_rgb = cv2.cvtColor(hand_crop, cv2.COLOR_RGB2BGR)
        img_pil  = Image.fromarray(hand_rgb)          # single PIL image
        clip_tensor = self.preprocess(img_pil).to(self.device).bfloat16()  # (3,224,224)
        if clip_tensor.ndim == 3:                     # add batch dim only if needed
            clip_tensor = clip_tensor.unsqueeze(0)    # (1,3,224,224)
        # --------------------------------------------------------------------
        with torch.no_grad():
            feat = self.visual(clip_tensor)                # ok for Conv2d
        return feat.float().cpu().numpy().squeeze().astype(np.float32)
    
    @torch.no_grad()
    def generate_fused_embedding(self, image: np.ndarray, mask: np.ndarray, text: str) -> np.ndarray:
        """Returns concatenated text + image embeddings."""
        text_feat  = self.generate_text_embedding(text)
        image_feat = self.generate_embedding(image, mask)
        return np.concatenate([text_feat, image_feat], axis=-1)
    
    @torch.no_grad()
    def generate_fused_embedding_hands(self, image: np.ndarray, hand_keypoints: list, label: str) -> np.ndarray:
        """Returns concatenated text + image embeddings for hands."""
        if hand_keypoints is None or len(hand_keypoints) == 0:
            print(f"[Warning] No keypoints for {label}. Skipping hand embedding.")
            return None
        
        image_feat = self.generate_embeddings_hands(image, hand_keypoints)
        if image_feat is None or image_feat.ndim != 1:
            print(f"[Warning] Hand crop too small or invalid for {label}. Skipping hand embedding.")
            return None
        
        text_feat  = self.generate_text_embedding(label)
        return np.concatenate([text_feat, image_feat], axis=-1)

    # ------------------------------------------------------------------
    # Helper: replicate the three‑crop box logic unchanged
    # ------------------------------------------------------------------
    def _three_crops(self, img: np.ndarray, mask: np.ndarray):
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            x_min = y_min = 0
            y_max, x_max = img.shape[:2]
        else:
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()

        h, w = img.shape[:2]
        def pad(a, p, hi):
            return max(min(a + p, hi - 1), 0)

        # crop‑1 (tight)
        c1 = img[y_min:y_max+1, x_min:x_max+1]

        # crop‑2 (mask background black)
        c2 = c1.copy()
        if len(xs):
            c2_mask = mask[y_min:y_max+1, x_min:x_max+1]
            c2[c2_mask == 0] = 0

        # crop‑3 (context box)
        px = int(0.4 * (x_max - x_min))
        py = int(0.4 * (y_max - y_min))
        xm2, xM2 = pad(x_min, -px, w), pad(x_max, px, w)
        ym2, yM2 = pad(y_min, -py, h), pad(y_max, py, h)
        c3 = img[ym2:yM2+1, xm2:xM2+1]

        return [c1, c2, c3]

# import numpy as np
# import torch
# from transformers import AutoImageProcessor, AutoModel
# import cv2



# class ObjectEmbeddingGenerator:
#     """Generates embeddings for objects using a pre-trained model."""

#     def __init__(self, model_name: str =  "nvidia/RADIO-L", device: str = "cuda"):
#         self.device = device    
#         self.processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
#         self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)


#     def generate_embedding(self, image: np.ndarray, mask: np.ndarray) -> torch.Tensor:
#         """Generates an embedding for the given image."""
#         # create three diffrent crops of the image
#         # 1. Crop a box around the mask and leave all pixels untouched.
#         ys, xs = np.where(mask)
#         if len(xs) == 0 or len(ys) == 0:
#             crop1 = image
#         else:
#             x_min, x_max = xs.min(), xs.max()
#             y_min, y_max = ys.min(), ys.max()
#             # add a small padding to the crop
#             pad_x = int((x_max - x_min) * 0.1)
#             pad_y = int((y_max - y_min) * 0.1)
#             x_min = max(x_min - pad_x, 0)
#             y_min = max(y_min - pad_y, 0)
#             x_max = min(x_max + pad_x, image.shape[1] - 1)
#             y_max = min(y_max + pad_y, image.shape[0] - 1)

#             crop1 = image[y_min:y_max+1, x_min:x_max+1]

#         # 2. Crop a box around the mask and fill all pixels outside the mask with black.
#         if len(xs) == 0 or len(ys) == 0:
#             crop2 = image.copy()
#         else:
#             crop2 = image[y_min:y_max+1, x_min:x_max+1].copy()
#             mask_crop = mask[y_min:y_max+1, x_min:x_max+1]
#             crop2[mask_crop == 0] = 0

#         # 3. Crop a larger box around the mask and leave all pixels untouched.
#         if len(xs) == 0 or len(ys) == 0:
#             crop3 = image
#         else:
#             h, w, _ = image.shape
#             pad_x = int((x_max - x_min) * 0.4)
#             pad_y = int((y_max - y_min) * 0.4)
#             x_min_ext = max(x_min - pad_x, 0)
#             y_min_ext = max(y_min - pad_y, 0)
#             x_max_ext = min(x_max + pad_x, w - 1)
#             y_max_ext = min(y_max + pad_y, h - 1)
#             crop3 = image[y_min_ext:y_max_ext+1, x_min_ext:x_max_ext+1]

#         # visualize the cropsq
#         # crop1_bgr = cv2.cvtColor(crop1, cv2.COLOR_RGB2BGR)
#         # crop2_bgr = cv2.cvtColor(crop2, cv2.COLOR_RGB2BGR)
#         # crop3_bgr = cv2.cvtColor(crop3, cv2.COLOR_RGB2BGR)
#         # cv2.imshow("Crop 1", crop1_bgr)
#         # cv2.imshow("Crop 2", crop2_bgr)
#         # cv2.imshow("Crop 3", crop3_bgr)
#         # cv2.waitKey(1000000)

#         images = [crop1, crop2, crop3]

#         next_support_size = self.model.get_nearest_supported_resolution(images[0].shape[0], images[0].shape[1])


#         for i in range(len(images)):
#             # resize the image to the input size of the model
#             images[i] = cv2.resize(images[i], (next_support_size[1], next_support_size[0]))

#         input = self.processor(images=images, return_tensors="pt").pixel_values.to(self.device)
#         with torch.no_grad():
#             outputs, _ = self.model(input)

#         avg_embedding = outputs.mean(dim=0)
#         return avg_embedding.cpu().numpy()
    

#     def generate_embeddings_hands(self, image: np.ndarray, hand_keypoints: list) -> torch.Tensor:
#         """Generates embeddings for the hands in the image."""
#         if hand_keypoints is None or len(hand_keypoints) == 0:
#             return None

#         hand_keypoints = np.array(hand_keypoints, dtype=np.float32)

#         # find the bounding box of the hand keypoints and add some padding
#         x_min = int(hand_keypoints[:, 0].min())
#         x_max = int(hand_keypoints[:, 0].max())
#         y_min = int(hand_keypoints[:, 1].min())
#         y_max = int(hand_keypoints[:, 1].max())

#         pad_x = int((x_max - x_min) * 0.4)
#         pad_y = int((y_max - y_min) * 0.4)

#         x_min = max(x_min - pad_x, 0)
#         y_min = max(y_min - pad_y, 0)
#         x_max = min(x_max + pad_x, image.shape[1] - 1)
#         y_max = min(y_max + pad_y, image.shape[0] - 1)

#         # check the size of the crop
#         if (x_max - x_min) < 10 or (y_max - y_min) < 10:
#             print("Hand crop is too small, skipping embedding generation.")
#             return None

#         hand_crop = image[y_min:y_max+1, x_min:x_max+1]

#         # visualize the hand crop
#         # cv2.imshow("Hand Crop", hand_crop)
#         # cv2.waitKey(1000)


#         # # visalize the hand crop
#         # cv2.imshow("Hand Crop", hand_crop)
#         # cv2.waitKey(1000)

#         # resize the hand crop to the input size of the model
#         next_support_size = self.model.get_nearest_supported_resolution(hand_crop.shape[0], hand_crop.shape[1])
#         hand_crop = cv2.resize(hand_crop, (next_support_size[1], next_support_size[0]))

#         input_data = self.processor(images=hand_crop, return_tensors="pt", do_resize=False).pixel_values.to(self.device)
#         with torch.no_grad():
#             outputs, raw = self.model(input_data)
#         # average the features across token sequence dimension
#         embedding = outputs
#         return embedding.cpu().numpy().squeeze()