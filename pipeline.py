import hydra
from omegaconf import DictConfig

import rerun as rr               # ➜ pip install -U rerun-sdk
import numpy as np
import cv2
from hydra.core.global_hydra import GlobalHydra
import datetime, os
import time

from video_loader import VideoLoader
from hand_detection import HandDetection
from yolo_hand_detection import HandDetection as YoloHandDetection
from segmentation import Segmentation
from graph_generator import GraphGenerator
from depth_generator import DepthGenerator


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    rr.init("video_stream", spawn=False)  # spawn=True ⇒ open viewer

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rec_path = os.path.join("/workspace", f"video_stream_{ts}.rrd")
    rr.save(rec_path)                     # write to disk while logging 🡅

    video_loader = VideoLoader(cfg.video.path)

    # hand detection
    hand_detection = HandDetection(cfg.hand_detection_mediapipe)
    # hand_detection = YoloHandDetection(cfg.hand_detection_yolo)

    # segmentation
    segmentation = Segmentation(cfg.segmentation)

    # depth generator
    depth_generator = DepthGenerator(cfg.depth_generator)

    # graph generator
    graph_generator = GraphGenerator(cfg.graph_generator)

    # ---- Reduce to ~10 FPS ----
    last_process_ts = -float('inf')
    target_interval = 1.0 / 1.0  # seconds between frames

    # processing loop
    while True:
        frame_rgb, timestamp = video_loader.next_frame()

        # only process at target FPS
        if timestamp - last_process_ts < target_interval:
            continue
        last_process_ts = timestamp

        # Tag this log with an integer timeline for easy scrubbing
        rr.set_time("time", duration=timestamp)

        # Optionally downscale if needed:
        # img = cv2.resize(frame_rgb, (frame_rgb.shape[1]//2, frame_rgb.shape[0]//2))
        img = frame_rgb

        # raw frame
        rr.log("raw_video/frame", rr.Image(img).compress(jpeg_quality=85))

        # Hand detection
        result, hd_img = hand_detection.detect_hands(
            img, timestamp_ms=int(timestamp * 1000)
        )
        rr.log("hand_detection/annotated_image", rr.Image(hd_img))

        t0 = time.perf_counter()

        # Segmentation
        masks, seg_img = segmentation.segment(
            img, timestamp_ms=int(timestamp * 1000)
        )

        t1 = time.perf_counter()

        print(f"Segmentation took {t1 - t0:.3f} seconds")
        rr.log("segmentation/annotated_image", rr.Image(seg_img))

        # Depth generation
        depth_map = depth_generator.estimate_depth(img)
        rr.log("depth_map", rr.Image(depth_map))

        # 3D point cloud
        h, w = depth_map.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        fx, fy = cfg.camera.fx, cfg.camera.fy
        cx, cy = cfg.camera.cx, cfg.camera.cy
        Z = depth_map.astype(np.float32)
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
        colors = img.reshape(-1, 3) / 255.0
        rr.log("point_cloud", rr.Points3D(points, colors=colors))

        # Graph generation
        graph, graph_img = graph_generator.generate_graph(img, masks, result)
        rr.log("graph_image", rr.Image(graph_img))


if __name__ == "__main__":
    GlobalHydra.instance().clear()
    main()
