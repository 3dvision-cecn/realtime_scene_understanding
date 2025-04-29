import numpy as np
import cv2
from ultralytics import YOLO



class HandDetection:

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = YOLO(cfg.model_path)




    def detect_hands(self, image, timestamp_ms: int = 0):
        """
        Input:
            image rgb: np.ndarray, shape (H, W, 3), dtype=uint8
        Output:

        """


        # use native resolution
        h, w = image.shape[:2]
        result = self.model.track(image, imgsz=(w, h))
        
        # draw landmarks on the image
        annotated_image = result[0].plot()

        return result[0], annotated_image



    def draw_landmarks_on_image(self, image, result):
        # unpack boxes and keypoints
        boxes = result.boxes.xyxy.cpu().numpy()          # shape: (n, 4)
        keypoints = None
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            keypoints = result.keypoints.cpu().numpy()   # shape: (n, k, 3)

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            # draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return image
