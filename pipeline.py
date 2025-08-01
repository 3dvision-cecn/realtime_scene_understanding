import hydra
from omegaconf import DictConfig

import rerun as rr               # ➜ pip install -U rerun-sdk
import numpy as np
import cv2
from hydra.core.global_hydra import GlobalHydra
import datetime, os
import time

from hydra.utils import instantiate
from omegaconf import OmegaConf
from hydra import compose, initialize


from pipeline.loaders.vrs_loader import VRSLoader
from pipeline.stages.hand_detection_hamer import HandDetection
from pipeline.stages.segmentation import Segmentation
from pipeline.stages.graph_generator import GraphGenerator
from pipeline.loaders.red_loader import R3D_loader
from pipeline.stages.training_generator import TrainingGenerator


from scipy.spatial.transform import Rotation
import torch
import gc

# multi processing
import torch.multiprocessing as mp




# ──────────── CONFIGURE SEGMENT OUTPUT ────────────

def main(cfg, start_rerun: bool = False, device = "cuda"):
    # disable rerun's default logging
    # do not record the rerun if start_rerun is False
    # clear the rerun log if start_rerun is True


    #ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #rec_path = os.path.join("/workspace", f"video_stream_{ts}.rrd")
    #rr.save(rec_path)                     # write to disk while logging 🡅

    if cfg.dataset == "red":
        video_loader = R3D_loader(cfg.video, device)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    elif cfg.dataset == "hd_epic":
        video_loader = VRSLoader(cfg.vrs_loader, device=device)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP)

    # hand detection
    hand_detection = HandDetection(cfg.hand_detection_hamer, device=device)
    # hand_detection = YoloHandDetection(cfg.hand_detection_yolo)

    # segmentation
    segmentation = Segmentation(cfg.segmentation, device=device)

    # graph generator
    graph_generator = GraphGenerator(cfg.graph_generator, device=device)

    # training generator
    training_generator = TrainingGenerator(cfg, device=device)
    if cfg.dataset == "hd_epic":
        training_generator.set_sample_dir(video_loader.get_folder_suffix())

    # ---- Reduce to ~10 FPS ----
    last_process_ts = -float('inf')
    target_interval = 1.0 / 4.0  # seconds between frames


    SEGMENT_OUTPUT_DIR = "segmented_frames"
    if cfg.pipeline.record_seg:
        os.makedirs(SEGMENT_OUTPUT_DIR, exist_ok=True)

    time_analysis = {}
    time_analysis["frame_load"] = []
    time_analysis["hand_detection"] = []
    time_analysis["segmentation_detection"] = []
    time_analysis["embed_generation"] = []
    time_analysis["graph_generation"] = []
    time_analysis["training_generator"] = []
    time_analysis["total_time"] = []
    
    testing_it = 10


    itr = 0
    # processing loop
    while True:
        itr+=1
        start_time = time.perf_counter()
        frame_rgb, depth, pose, timestamp = video_loader.next_frame()
        load_time = time.perf_counter() - start_time
        time_analysis["frame_load"].append(load_time)

        if frame_rgb is None or depth is None or pose is None:
            print("No more frames available, exiting.")
            break


        pixel_indexed_pcd = video_loader.generate_pixel_indexed_pcd(frame_rgb, depth, pose)
        if pixel_indexed_pcd is None:
            print("No pixel indexed point cloud available, skipping frame.")
            continue

        # if itr >= testing_it:
        #     break


        # Tag this log with an integer timeline for easy scrubbing
        rr.set_time("time", duration=timestamp)

        # Optionally downscale if needed:
        # img = cv2.resize(frame_rgb, (frame_rgb.shape[1]//2, frame_rgb.shape[0]//2))
        img = frame_rgb

        # raw frame
        rr.log("raw_video/frame", rr.Image(img).compress(jpeg_quality=85))

        # if hd_epic, log the psuedo depth map
        if cfg.dataset == "hd_epic":
            pseudo_depth = video_loader.pseudo_depth
            # nan to zero
            pseudo_depth = np.nan_to_num(pseudo_depth, nan=0.0)

            pseudo_depth_vis = img.copy()
            nonzero_indices = np.nonzero(pseudo_depth)
            for y, x in zip(*nonzero_indices):
                # apply colormap to pseudo depth
                depth_val = np.uint8(pseudo_depth[y, x] * 255)
                color = cv2.applyColorMap(np.array([[depth_val]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0].tolist()
                cv2.circle(pseudo_depth_vis, (x, y), 6, color, -1)


            rr.log("PseudoDepth", rr.Image(pseudo_depth_vis))


        # Hand detection
        hand_detection_start_time = time.perf_counter()
        hand_data, hd_img = hand_detection.detect_hands(
            img, pixel_indexed_pcd, timestamp_ms=int(timestamp * 1000)
        )
        time_analysis["hand_detection"].append(time.perf_counter() - hand_detection_start_time)

        rr.log("hand_detection/annotated_image", rr.Image(hd_img))
        


        # Segmentation
        objects, seg_img, hand_data = segmentation.segment(
            img, pixel_indexed_pcd, hand_data, timestamp_ms=int(timestamp * 1000), iteration=itr
        )
        time_analysis["segmentation_detection"].append(segmentation.det_seg_time)
        time_analysis["embed_generation"].append(segmentation.embedding_generation_time)


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




        graph_generator_start_time = time.perf_counter()
        graph = graph_generator.generate_graph(img, objects, hand_data)
        time_analysis["graph_generation"].append(time.perf_counter() - graph_generator_start_time)

        time_analysis["total_time"].append(time.perf_counter() - start_time)

        # Training generator
        training_generator_start_time = time.perf_counter()
        training_generator.add_sequence(img, graph)
        time_analysis["training_generator"].append(time.perf_counter() - training_generator_start_time)

        end_time = time.perf_counter()
        print(f"Processing time for frame {itr}: {end_time - start_time:.3f} seconds")
        # Log the processing time

    # print the average processing time
    for key, times in time_analysis.items():
        print(f"Average time for {key}: {np.mean(times):.3f} seconds over {len(times)} frames")

import argparse



training_path = f"dataset/graph_samples_radio_epic_hd_new"


import copy
def process_vrs_hd(vrs_file, device):
    GlobalHydra.instance().clear()
    with initialize(config_path="conf", version_base=None):
        cfg = compose(config_name="config")
        cfg.training_generator.path = training_path
        cfg.dataset = "hd_epic"
        cfg.vrs_loader.path = vrs_file
        main(cfg, start_rerun=False, device=device)
        rr.Clear(recursive=True)
        torch.cuda.empty_cache()
        gc.collect()


def mp_process_vrs_hd(args):
    vrs_file, device = args
    return process_vrs_hd(vrs_file, device)

if __name__ == "__main__":
    GlobalHydra.instance().clear()

    parser = argparse.ArgumentParser(description="Run the video processing pipeline.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="red",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        default=False,
        help="Run the full pipeline on all of the recorded videos.",
    )

    args = parser.parse_args()

    if args.dataset == "red":
        print("+++++++++++++++++++ USING RED")
        if args.full:
            # find all the folder in the videos directory
            train_path = "dataset/recordings/train"
            val_path = "dataset/recordings/val"

            train_folders = []
            for folder in os.listdir(train_path):
                train_folders.append(os.path.join(train_path, folder))

            val_folders = []
            for folder in os.listdir(val_path):
                val_folders.append(os.path.join(val_path, folder))

            print("Train folders:", train_folders)
            print("Val folders:", val_folders)
            # get the config from hydra
            with initialize(config_path="conf", version_base=None):
                cfg = compose(config_name="config")

                # create a randoom folder in dataset/graph_samplesXXXXX
                cfg.training_generator.path = f"dataset/graph_samples{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                cfg.dataset = "red"
                # first process the training videos
                for folder in train_folders:
                    cfg.video.path = folder
                    print(f"Processing training folder: {folder}")
                    main(cfg=cfg)
                    rr.Clear(recursive=True)
                    # empty the cuda cache and free up cuda memory
                    torch.cuda.empty_cache()



                # then process the validation videos
                for folder in val_folders:
                    cfg.video.path = folder
                    print(f"Processing validation folder: {folder}")
        
        else:
            # Run the pipeline on a single video diectly from the config
            from hydra import compose, initialize
            with initialize(config_path="conf", version_base=None):
                rr.init("video_stream", spawn=True)  # spawn=True ⇒ open viewer
                rr.serve_web_viewer(open_browser=False)

                cfg = compose(config_name="config")
                cfg.dataset = "red"
                main(cfg, start_rerun=True)

    elif args.dataset == "hd_epic":
        if args.full:
            from hydra.utils import instantiate
            from omegaconf import OmegaConf
            from hydra import compose, initialize
            import time

            train_path = "dataset/HD-EPIC/VRS/"

            vrs_files = []
            for folder in os.listdir(train_path):
                for vrs_file in os.listdir(train_path + folder):
                    if vrs_file.endswith(".vrs"):
                        vrs_files.append(train_path + folder + "/"+ vrs_file)
            
            print("Found vrs files", len(vrs_files), " in the directories")

            gpus_available = list(range(torch.cuda.device_count()))

            mp.set_start_method("spawn", force=True)
            
            # Create an argument list for each vrs file; assign devices round robin.
            # Initialize a dictionary to track the process running on each device.
            device_process = {"cuda:" + str(g): None for g in gpus_available}
            # Create a copy of the list of vrs_files to process.
            vrs_queue = list(vrs_files)

            # Loop until all vrs files are processed and all devices are idle.
            while vrs_queue or any(proc is not None for proc in device_process.values()):
                for device in list(device_process.keys()):
                    proc = device_process[device]
                    if proc is not None:
                        # Check if the previously assigned process is finished.
                        if not proc.is_alive():
                            proc.join()
                            device_process[device] = None

                    # If the device is idle, start processing the next vrs_file.
                    if device_process[device] is None and vrs_queue:
                        vrs_file = vrs_queue.pop(0)
                        process = mp.Process(target=mp_process_vrs_hd, args=((vrs_file, device),))
                        process.start()
                        device_process[device] = process
                # Sleep a little before checking again.
                time.sleep(1)
        else:
            # Run the pipeline on a single video diectly from the config
            from hydra import compose, initialize
            import copy
            with initialize(config_path="conf", version_base=None):
                rr.init("video_stream", spawn=True)  # spawn=True ⇒ open viewer
                rr.serve_web_viewer(open_browser=False)

                cfg = compose(config_name="config")
                cfg.dataset = "hd_epic"
                main(cfg, start_rerun=True)


