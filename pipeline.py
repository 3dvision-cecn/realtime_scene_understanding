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
from training_generator import TrainingGenerator


from scipy.spatial.transform import Rotation

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

    # training generator
    training_generator = TrainingGenerator(cfg)

    # ---- Reduce to ~10 FPS ----
    last_process_ts = -float('inf')
    target_interval = 1.0 / 4.0  # seconds between frames


    SEGMENT_OUTPUT_DIR = "segmented_frames"
    if cfg.pipeline.record_seg:
        os.makedirs(SEGMENT_OUTPUT_DIR, exist_ok=True)

    itr = 0
    # processing loop
    while True:
        itr+=1
        start_time = time.perf_counter()
        frame_rgb, depth, pose, timestamp = video_loader.next_frame()

        if frame_rgb is None or depth is None or pose is None:
            print("No more frames available, exiting.")
            break

        pixel_indexed_pcd = video_loader.generate_pixel_indexed_pcd(frame_rgb, depth, pose)
        if pixel_indexed_pcd is None:
            print("No pixel indexed point cloud available, skipping frame.")
            continue

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
        t0 = time.time()
        hand_data, hd_img = hand_detection.detect_hands(
            img, pixel_indexed_pcd, timestamp_ms=int(timestamp * 1000)
        )
        t1 = time.time()
        print(f"Hand detection took {t1 - t0:.3f} seconds")
        rr.log("hand_detection/annotated_image", rr.Image(hd_img))
        


        t0 = time.perf_counter()

        # Segmentation
        t0 = time.time()
        objects, seg_img, hand_data = segmentation.segment(
            img, pixel_indexed_pcd, hand_data, timestamp_ms=int(timestamp * 1000), iteration=itr
        )
        t1 = time.time()
        print(f"Segmentation took {t1 - t0:.3f} seconds")

        t1 = time.perf_counter()

        print(f"Segmentation took {t1 - t0:.3f} seconds")
        rr.log("segmentation/annotated_image", rr.Image(seg_img))

        # # ───── Save segmented frame ─────
        # if cfg.pipeline.record_seg:
        #     seg_filename = os.path.join(SEGMENT_OUTPUT_DIR, f"seg_{itr}.png")
        #     cv2.imwrite(seg_filename, seg_img, [cv2.IMWRITE_JPEG_QUALITY, 60])

        # point cloud
        rr.log("depth_map", rr.Image(depth))
        points, colors = video_loader.generate_pcd(img, depth, pose)
        rr.log("world/point_cloud", rr.Points3D(points, colors=colors))

        # log objects
        labels = []
        centroids = []
        rotations = []
        centers = []
        half_sizes = []
        for obj in objects:
            labels.append(obj.name)
            centroids.append(obj.position)
            rot_mat = obj.obb.R
            quat_xyzw = Rotation.from_matrix(rot_mat).as_quat()
            rotations.append(quat_xyzw)
            centers.append(obj.obb.center)
            half_sizes.append(obj.obb.extent / 2.0)

        rr.log(
            "world/objects",
            rr.Boxes3D(
                centers=np.array(centers),
                half_sizes=np.array(half_sizes),
                rotations=np.array(rotations),
                labels=labels,
            )
        )

        # log hand keypoints
        # as 3d points
        if hand_data is not None:
            left_hand = hand_data.left_hand
            rr.log(
                "world/left_hand_keypoints",
                rr.Points3D(
                    left_hand.keypoints_pcd,
                    colors=np.array([[255, 0, 0]] * len(left_hand.keypoints_pcd)),
                    radii=0.01,
                ),
            )
            right_hand = hand_data.right_hand
            rr.log(
                "world/right_hand_keypoints",
                rr.Points3D(
                    right_hand.keypoints_pcd,
                    colors=np.array([[0, 0, 255]] * len(right_hand.keypoints_pcd)),
                    radii=0.01,
                ),
            )





        graph = graph_generator.generate_graph(img, objects, hand_data)

        # Training generator
        training_generator.add_sequence(img, graph)

        end_time = time.perf_counter()
        print(f"Processing time for frame {itr}: {end_time - start_time:.3f} seconds")
        # Log the processing time

if __name__ == "__main__":
    GlobalHydra.instance().clear()
    main()
