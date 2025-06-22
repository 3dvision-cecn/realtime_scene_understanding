
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
import cv2



class ObjectEmbeddingGenerator:
    """Generates embeddings for objects using a pre-trained model."""

    def __init__(self, model_name: str =  "nvidia/RADIO-L", device: str = "cuda"):
        self.device = device    
        self.processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)


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
            # add a small padding to the crop
            pad_x = int((x_max - x_min) * 0.1)
            pad_y = int((y_max - y_min) * 0.1)
            x_min = max(x_min - pad_x, 0)
            y_min = max(y_min - pad_y, 0)
            x_max = min(x_max + pad_x, image.shape[1] - 1)
            y_max = min(y_max + pad_y, image.shape[0] - 1)

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

        next_support_size = self.model.get_nearest_supported_resolution(images[0].shape[0], images[0].shape[1])


        for i in range(len(images)):
            # resize the image to the input size of the model
            images[i] = cv2.resize(images[i], (next_support_size[1], next_support_size[0]))

        input = self.processor(images=images, return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad():
            outputs, _ = self.model(input)

        avg_embedding = outputs.mean(dim=0)
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

        # resize the hand crop to the input size of the model
        next_support_size = self.model.get_nearest_supported_resolution(hand_crop.shape[0], hand_crop.shape[1])
        hand_crop = cv2.resize(hand_crop, (next_support_size[1], next_support_size[0]))

        input_data = self.processor(images=hand_crop, return_tensors="pt", do_resize=False).pixel_values.to(self.device)
        with torch.no_grad():
            outputs, raw = self.model(input_data)
        # average the features across token sequence dimension
        embedding = outputs
        return embedding.cpu().numpy().squeeze()