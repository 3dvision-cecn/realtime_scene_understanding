# pipeline.py
import hydra
from omegaconf import DictConfig

import rerun as rr               # ➜ pip install -U rerun-sdk
import numpy as np
from hydra.core.global_hydra import GlobalHydra


from video_loader import VideoLoader
from hand_detection import HandDetection
from yolo_hand_detection import HandDetection as YoloHandDetection
from segmentation import Segmentation
from graph_generator import GraphGenerator
from depth_generator import DepthGenerator



@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    rr.init("video_stream", spawn=True)          # spawn=True ⇒ open viewer


    video_loader = VideoLoader(cfg.video.path)         # cfg.video.path from YAML

    # hand detection
    hand_detection = HandDetection(cfg.hand_detection_mediapipe)
    # hand_detection = YoloHandDetection(cfg.hand_detection_yolo)

    # segmentation
    segmentation = Segmentation(cfg.segmentation)

    # depth generator
    depth_generator = DepthGenerator(cfg.depth_generator)

    # graph generator
    graph_generator = GraphGenerator(cfg.graph_generator)

    
    # processing loop
    while True:
        frame_rgb, timestamp = video_loader.next_frame()

        # Tag this log with an integer timeline for easy scrubbing
        rr.set_time("time", duration=timestamp)

        # Entity path decides where it shows up in the viewer
        rr.log("raw_video/frame", rr.Image(frame_rgb))

        # Hand detection
        result, annotated_image = hand_detection.detect_hands(frame_rgb, timestamp_ms=int(timestamp * 1000))
        rr.log("hand_detection/annotated_image", rr.Image(annotated_image))

        # Segmentation
        masks, annotated_image = segmentation.segment(frame_rgb, timestamp_ms=int(timestamp * 1000))
        rr.log("segmentation/annotated_image", rr.Image(annotated_image))

        # Depth generation
        depth_map = depth_generator.estimate_depth(frame_rgb)
        rr.log("depth_map", rr.Image(depth_map))

        # create a 3D point cloud from the depth map with the image
        # create a 3D point cloud from the depth map with the image
        h, w = depth_map.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        fx = cfg.camera.fx
        fy = cfg.camera.fy
        cx = cfg.camera.cx
        cy = cfg.camera.cy

        # back‐project to 3D (assume depth_map in meters)
        Z = depth_map.astype(np.float32)
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        # flatten and pack positions + colors
        points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
        colors = frame_rgb.reshape(-1, 3) / 255.0  # normalize to [0,1]

        rr.log("point_cloud", rr.Points3D(points, colors=colors))

        # Graph generation
        graph, graph_image = graph_generator.generate_graph(frame_rgb, masks, result)
        rr.log("graph_image", rr.Image(graph_image))





if __name__ == "__main__":
    # Load the config file
    # clear any previous Hydra instance
    GlobalHydra.instance().clear()
    main()