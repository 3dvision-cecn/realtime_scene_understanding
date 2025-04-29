import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import numpy as np
import cv2

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode




class HandDetection:

    def __init__(self, cfg):
        self.cfg = cfg

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=cfg.model_path),
            num_hands=2
            )
        
        self.landmarker = HandLandmarker.create_from_options(options)

        self.MARGIN = 10
        self.FONT_SIZE = 0.5
        self.FONT_THICKNESS = 2
        self.HANDEDNESS_TEXT_COLOR = (0, 255, 0)

    


    def detect_hands(self, image, timestamp_ms: int = 0):
        """
        Input:
            image rgb: np.ndarray, shape (H, W, 3), dtype=uint8
        Output:

        """

        image_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        result = self.landmarker.detect(image_mp)

        # draw landmarks on the image
        annotated_image = self.draw_landmarks_on_image(image, result)

        return result, annotated_image



    def draw_landmarks_on_image(self, rgb_image, detection_result):
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        annotated_image = np.copy(rgb_image)

        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

            # Get the top left corner of the detected hand's bounding box.
            height, width, _ = annotated_image.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - self.MARGIN

            # Draw handedness (left or right hand) on the image.
            cv2.putText(annotated_image, f"{handedness[0].category_name}",
                        (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                        self.FONT_SIZE, self.HANDEDNESS_TEXT_COLOR, self.FONT_THICKNESS, cv2.LINE_AA)

        return annotated_image