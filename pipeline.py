import hydra
from omegaconf import DictConfig

import rerun as rr               # ➜ pip install -U rerun-sdk
import numpy as np
import cv2
from hydra.core.global_hydra import GlobalHydra
import datetime, os
import time

from video_loader import VideoLoader
from hand_detection_hamer import HandDetection
from segmentation import Segmentation
from graph_generator import GraphGenerator
from red_loader import R3D_loader

# ──────────── CONFIGURE SEGMENT OUTPUT ────────────


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    rr.init("unique12839")  # spawn=True ⇒ open viewer
    stream = rr.RecordingStream("my_app324324")

    stream.serve_web(open_browser=True, web_port=9090, grpc_port=9876)
    
    stream.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    video_loader = R3D_loader(cfg.video)
    #video_loader = VideoLoader(cfg.video.path)


    # hand detection
    #hand_detection = HandDetection(cfg.hand_detection_hamer)
    #hand_detection = YoloHandDetection(cfg.hand_detection_yolo)

    # segmentation
    segmentation = Segmentation(cfg.segmentation)

    # graph generator
    graph_generator = GraphGenerator(cfg.graph_generator)

    # ---- Reduce to ~10 FPS ----
    last_process_ts = -float('inf')
    target_interval = 1.0 / 10.0  # seconds between frames


    SEGMENT_OUTPUT_DIR = "segmented_frames"
    if cfg.pipeline.record_seg:
        os.makedirs(SEGMENT_OUTPUT_DIR, exist_ok=True)

    hand_data = None

    itr = 0
    # processing loop
    while True:
        itr+=1
        frame_rgb, depth, pose, timestamp = video_loader.next_frame()

        # only process at target FPS
        if timestamp - last_process_ts < target_interval:
            continue
        last_process_ts = timestamp

        # Tag this log with an integer timeline for easy scrubbing
        stream.set_time("time", duration=timestamp)

        # Optionally downscale if needed:
        # img = cv2.resize(frame_rgb, (frame_rgb.shape[1]//2, frame_rgb.shape[0]//2))
        img = frame_rgb

        # raw frame
        stream.log("raw_video/frame", rr.Image(img).compress(jpeg_quality=85))

        # Hand detection
        '''
        hand_data, hd_img = hand_detection.detect_hands(
            img, timestamp_ms=int(timestamp * 1000)
        )
        stream.log("hand_detection/annotated_image", rr.Image(hd_img).compress(jpeg_quality=85))
        '''
        t0 = time.perf_counter()

        # Segmentation
        masks, seg_img = segmentation.segment(
            img, timestamp_ms=int(timestamp * 1000), iteration=itr
        )

        t1 = time.perf_counter()

        print(f"Segmentation took {t1 - t0:.3f} seconds")
        stream.log("segmentation/annotated_image", rr.Image(seg_img).compress(jpeg_quality=85))

        # point cloud
        stream.log("depth_map", rr.Image(depth).compress(jpeg_quality=85))
        #points, colors = video_loader.generate_pcd(hd_img, depth, pose)
        #stream.log("world/point_cloud", rr.Points3D(points, colors=colors))

        # Graph generation
        if hand_data is not None:
         graph, graph_img = graph_generator.generate_graph(img, masks, hand_data)
         stream.log("graph_image", rr.Image(graph_img).compress(jpeg_quality=85))


if __name__ == "__main__":
    GlobalHydra.instance().clear()
    main()
