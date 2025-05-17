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
    rr.init("video_stream", spawn=True)  # spawn=True ⇒ open viewer
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    rr.serve_web_viewer(open_browser=False)

    #ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #rec_path = os.path.join("/workspace", f"video_stream_{ts}.rrd")
    #rr.save(rec_path)                     # write to disk while logging 🡅

    video_loader = R3D_loader(cfg.video)

    # hand detection
    hand_detection = HandDetection(cfg.hand_detection_hamer)
    # hand_detection = YoloHandDetection(cfg.hand_detection_yolo)

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
        rr.set_time("time", duration=timestamp)

        # Optionally downscale if needed:
        # img = cv2.resize(frame_rgb, (frame_rgb.shape[1]//2, frame_rgb.shape[0]//2))
        img = frame_rgb

        # raw frame
        rr.log("raw_video/frame", rr.Image(img).compress(jpeg_quality=85))

        # Hand detection
        hand_data, hd_img = hand_detection.detect_hands(
            img, timestamp_ms=int(timestamp * 1000)
        )
        rr.log("hand_detection/annotated_image", rr.Image(hd_img))

        t0 = time.perf_counter()

        # Segmentation
        objects, seg_img = segmentation.segment(
            img, timestamp_ms=int(timestamp * 1000), iteration=itr
        )

        t1 = time.perf_counter()

        print(f"Segmentation took {t1 - t0:.3f} seconds")
        rr.log("segmentation/annotated_image", rr.Image(seg_img))

        # # ───── Save segmented frame ─────
        # if cfg.pipeline.record_seg:
        #     seg_filename = os.path.join(SEGMENT_OUTPUT_DIR, f"seg_{itr}.png")
        #     cv2.imwrite(seg_filename, seg_img, [cv2.IMWRITE_JPEG_QUALITY, 60])

        # point cloud
        rr.log("depth_map", rr.Image(depth))
        points, colors = video_loader.generate_pcd(hd_img, depth, pose)
        rr.log("world/point_cloud", rr.Points3D(points, colors=colors))

        # # Graph generation
        # if hand_data is not None:
        #  graph, graph_img = graph_generator.generate_graph(img, masks, hand_data)
        #  rr.log("graph_image", rr.Image(graph_img))


if __name__ == "__main__":
    GlobalHydra.instance().clear()
    main()
