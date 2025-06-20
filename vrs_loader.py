import open3d as o3d
from dataclasses import dataclass
import numpy as np
import os
import cv2
import rerun as rr
import torch
import json
# aria imports
from projectaria_tools.core import data_provider
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core import mps # IO for Aria MPS assets
from projectaria_tools.core.mps.utils import ( # Aria MPS utilities
    filter_points_from_confidence,
    filter_points_from_count,
)
from projectaria_tools.core.calibration import (
    CameraCalibration,
    distort_by_calibration,
)
from projectaria_tools.core import calibration
from points_and_observation_manager import PointsAndObservationsManager, OnlineRgbCameraHelper
from datetime import timedelta

# depth estimate
import depth_pro
from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
import zipfile
import shutil
from omni_dc.omni_dc_wrapper import DepthPredictor

LEFT_SLAM_STREAM_ID = StreamId("1201-1")
RIGHT_SLAM_STREAM_ID = StreamId("1201-2")

class VRSLoader:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        # this is the path to the VRS data
        self.vrs_path = cfg.path
        self.ego_exo_project_path = os.path.dirname(self.vrs_path)

        print("VRs path", self.vrs_path)
        print("Device is", device)

        # get which P01, P02, etc. this is
        self.person_id = os.path.basename(os.path.dirname(self.vrs_path))


        self.vrs_slam_mapping_json = self.cfg.vrs_slam_mapping_json + self.person_id +"/SLAM/multi/vrs_to_multi_slam.json"
        if not os.path.exists(self.vrs_slam_mapping_json):
            raise ValueError(f"VRS SLAM mapping JSON file does not exist: {self.vrs_slam_mapping_json}")
        
        
        with open(self.vrs_slam_mapping_json, 'r') as f:
            self.vrs_slam_mapping = json.load(f)
            print(f"Loaded VRS SLAM mapping from {self.vrs_slam_mapping}")

        # get the last part of the path to use as the project name like "P01/P01-20240202-110250.vrs"
        # from the path like "data/HD-EPIC/VRS/P01/P01-20240202-110250_anonymized.vrs"
        self.project_name = os.path.join(os.path.basename(os.path.dirname(self.vrs_path)), os.path.basename(self.vrs_path))
        # remove the "_anonymized.vrs" part from the project name
        self.project_name = self.project_name.replace("_anonymized.vrs", ".vrs")

        # check if the project name is in the mapping
        if self.project_name not in self.vrs_slam_mapping:
            raise ValueError(f"Project name {self.project_name} not found in the VRS SLAM mapping JSON file: {self.vrs_slam_mapping_json}")
        # get the project name from the mapping
        self.slam_name = self.vrs_slam_mapping[self.project_name]
        print(f"Project name: {self.project_name}, SLAM name: {self.slam_name}")

        # generate the slam root directory
        self.slam_root_dir = os.path.dirname(self.vrs_slam_mapping_json)
        self.slam_dir = os.path.join(self.slam_root_dir, self.slam_name)

        # if slamdir does not exist but there is a zip file instead unzip and use that
        self.unzipped = False
        if not os.path.exists(self.slam_dir):
            zip_path = self.slam_dir + ".zip"
            if os.path.exists(zip_path):
                print(f"Slam directory not found. Extracting from {zip_path}...")
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(self.slam_root_dir)
                self.unzipped = True
                print(f"Extraction complete. Slam directory available at {self.slam_dir}")
            else:
                raise ValueError(f"Slam directory does not exist and no zip file found at: {zip_path}")
            
        # HAND TRACKING

        # base_name = os.path.splitext(os.path.basename(self.project_name))[0]
        # self.hand_tracking_folder = self.cfg.vrs_slam_mapping_json + self.person_id + "/GAZE_HAND/" + "mps_" + base_name + "_vrs.zip"
        # self.hand_tracking_folder_unzipped = self.cfg.vrs_slam_mapping_json + self.person_id + "/GAZE_HAND/" + "mps_" + base_name + "_vrs"

        # # Check if the hand‐tracking zip exists
        # if os.path.isfile(self.hand_tracking_folder):
        #     # Compute extraction directory (remove .zip suffix)
        #     self.extract_dir = os.path.dirname(self.hand_tracking_folder) 
        #     # Make sure it exists
        #     os.makedirs(self.extract_dir, exist_ok=True)
        #     # Extract all files
        #     with zipfile.ZipFile(self.hand_tracking_folder, "r") as zf:
        #         zf.extractall(self.extract_dir)
        #     # Update the folder path to the extracted directory
        #     self.hand_tracking_folder = self.extract_dir
        #     print(f"Extracted hand‐tracking data to {self.extract_dir}")

        # self.hand_tracking_result_path = self.hand_tracking_folder_unzipped + "/hand_tracking/wrist_and_palm_poses.csv"

        # self.hand_tracking_results = mps.hand_tracking.read_wrist_and_palm_poses(
        #     self.hand_tracking_result_path
        # )

        # print(f"Number of hand tracking results: {len(self.hand_tracking_results)}")

        
        # paramters
        self.time_domain = TimeDomain.DEVICE_TIME  # query data based on host time
        self.option = TimeQueryOptions.CLOSEST # get data whose time [in TimeDomain] is CLOSEST to query time
        self.threshold_invdep = 5e-4
        self.threshold_dep = 5e-4
        self.max_points_per_pcd = 500_000  # maximum number of points per point cloud
        self.crop_size = 60  # crop the image by this many pixels on each side
        self.blur_threshold = 18.0 # threshold for skipping due to blur
        self.decimation_factor = 8  # decimation factor for the point cloud

        # get the vrs data
        self.vrs_data_provider = data_provider.create_vrs_data_provider(self.vrs_path)
        if self.vrs_data_provider is None:
            raise ValueError(f"Could not create VRS data provider for path: {self.vrs_path}")
        self.rgb_stream_id = StreamId("214-1")
        self.rgb_stream_label = self.vrs_data_provider.get_label_from_stream_id(self.rgb_stream_id)
        self.device_calibration = self.vrs_data_provider.get_device_calibration()
        self.start_time = self.vrs_data_provider.get_first_time_ns(self.rgb_stream_id, self.time_domain)
        self.end_time = self.vrs_data_provider.get_last_time_ns(self.rgb_stream_id, self.time_domain)
        image_config = self.vrs_data_provider.get_image_configuration(self.rgb_stream_id)

        self.uncropped_width = image_config.image_width
        self.uncropped_height = image_config.image_height
        self.width = self.uncropped_width - 2 * self.crop_size
        self.height = self.uncropped_height - 2 * self.crop_size

        self.num_images = self.vrs_data_provider.get_num_data(self.rgb_stream_id)


        # mps config for pose estimates and point cloud
        mps_data_paths_provider = mps.MpsDataPathsProvider(self.slam_dir)
        mps_data_paths = mps_data_paths_provider.get_data_paths()
        self.mps_data_provider = mps.MpsDataProvider(mps_data_paths)
        trajectory_data = mps.read_closed_loop_trajectory(mps_data_paths.slam.closed_loop_trajectory)

        self.device_trajectory = [it.transform_world_device.translation()[0] for it in trajectory_data][0::80]
        self.point_cloud = self.mps_data_provider.get_semidense_point_cloud()
        # self.point_observations = self.mps_data_provider.get_semidense_observations()

        # get directly from camera visible keypoints
        self.points_and_observations_manager = PointsAndObservationsManager.from_mps_data_provider(
            self.mps_data_provider
        )
        self.camera_serial_left_slam = (
            self.vrs_data_provider.get_configuration(LEFT_SLAM_STREAM_ID)
            .image_configuration()
            .sensor_serial
        )
        self.camera_serial_right_slam = (
            self.vrs_data_provider.get_configuration(LEFT_SLAM_STREAM_ID)
            .image_configuration()
            .sensor_serial
        )


        # filter the pointcloud
        # filtered_point_cloud = filter_points_from_confidence(self.point_cloud, self.threshold_invdep, self.threshold_dep)
        # downsampled_points_cloud = filter_points_from_count(filtered_point_cloud, self.max_points_per_pcd)
        # Retrieve point positions
        # self.points_position = np.stack([it.position_world for it in downsampled_points_cloud])
        # N = self.points_position.shape[0]
        # self.points_world_h = np.hstack([self.points_position, np.ones((N, 1), dtype=np.float32)])  # (N,4)

        print(f"VRS data loaded from {self.vrs_path}")
        print(f"Number of images in the VRS data: {self.num_images}")
        print(f"Start time: {self.start_time}, End time: {self.end_time}")
        # print(f"Point cloud contains {len(self.points_position)} points")


        self.omni_dcnet = DepthPredictor(checkpoint_path="conf/checkpoints/omni_dc/modelv1.1_best_72epochs.pt",
                                          da_path="conf/checkpoints/omni_dc/depth_anything_v2_vitl.pth",
                                          device=self.device)


        print("Finished loading the depth pro file")

        if self.unzipped:
            print(f"Unzipped slam data is no longer needed, deleting {self.slam_dir}...")
            shutil.rmtree(self.slam_dir)

        self.idx = 0  # index for the next frame to be processed

        print("Finished the vrs loader")



    def get_camera_calibration(self, stream_id: StreamId) -> CameraCalibration:
        device_calibration = self.vrs_data_provider.get_device_calibration()
        
        stream_label = self.vrs_data_provider.get_label_from_stream_id(stream_id)
        camera_calibration = device_calibration.get_camera_calib(stream_label)
        return camera_calibration

    def get_folder_suffix(self) -> str:
        # Get the folder suffix to save the results
        return self.person_id + "/" + self.slam_name
    



    def get_palm_and_wrist_pose(self, time_ns: int):
        # tolerance of 100 milliseconds in nanoseconds
        tol_us = 1000 * 1e3 # has to be high otherise they are not alligned at least we get some tracking this way
        time_us = time_ns * 1e-3
        # find the hand‐tracking result with the smallest time difference
        best = min(
            self.hand_tracking_results,
            key=lambda r: abs(r.tracking_timestamp - timedelta(microseconds=time_us))
        )
        # Check if the closest timestamp is within tolerance using timedelta
        if abs(best.tracking_timestamp - timedelta(microseconds=time_us)) > timedelta(microseconds=tol_us):
            print(f"No valid hand tracking result found, best timedelta {best.tracking_timestamp - timedelta(microseconds=time_us).microseconds * 1e3} ms")
            return None, None
        # return (palm_pose, wrist_pose)
        print(f"Valid hand tracking recors is found, best timedelta {(best.tracking_timestamp - timedelta(microseconds=time_us)).microseconds * 1e-3} ms")
        return best
    

    def get_hand_positions_in_world(self, hand, T_world_device: np.ndarray):
        # return the latest hand positions in world coordinates
        if hand is not None:
            left_hand = hand.left_hand
            right_hand = hand.right_hand

            if left_hand is not None:
                # check if the confidence is larger than 0.5
                if left_hand.confidence > 0.5:
                    left_palm_pos = T_world_device @ np.append(left_hand.palm_position_device, 1)
                    left_wrist_pos = T_world_device @ np.append(left_hand.wrist_position_device, 1)

                    left_palm_pos = left_palm_pos[:3]  # remove the homogeneous coordinate
                    left_wrist_pos = left_wrist_pos[:3]  # remove the homogeneous coordinate
                else:
                    left_palm_pos = None
                    left_wrist_pos = None
            else:
                left_palm_pos = None
                left_wrist_pos = None


            if right_hand is not None:
                # check if the confidence is larger than 0.5
                if right_hand.confidence > 0.5:
                    right_palm_pos = T_world_device @ np.append(right_hand.palm_position_device, 1)
                    right_wrist_pos = T_world_device @ np.append(right_hand.wrist_position_device, 1)

                    right_palm_pos = right_palm_pos[:3]
                    right_wrist_pos = right_wrist_pos[:3]
                else:
                    right_palm_pos = None
                    right_wrist_pos = None
            else:
                right_palm_pos = None
                right_wrist_pos = None


            hand_pos = {}
            hand_pos["left_palm"] = left_palm_pos
            hand_pos["left_wrist"] = left_wrist_pos
            hand_pos["right_palm"] = right_palm_pos
            hand_pos["right_wrist"] = right_wrist_pos
            return hand_pos


    def get_undistorted_image(self, stream_id: StreamId, index: int) -> np.array:
        # Retrieving the image
        image_tuple = self.vrs_data_provider.get_image_data_by_index(stream_id, index)
        
        # Retrieve the camera calibration attached to the Image (stream_id)
        camera_calibration = self.get_camera_calibration(stream_id)

        # Building the target calibration (Pinhole camera) to get the undistorted image
        focal_lengths = camera_calibration.get_focal_lengths()
        image_size = camera_calibration.get_image_size()
        pinhole_calib = calibration.get_linear_camera_calibration(
            image_size[0], image_size[1], focal_lengths[0]
        )

        # Compute the actual undistorted image (pixel sampling by using ray projection/reprojection)
        try:
            undistorted_image = distort_by_calibration(
                image_tuple[0].to_numpy_array(), pinhole_calib, camera_calibration
            )
        except:
            return None, None

        return undistorted_image, image_tuple[1].capture_timestamp_ns


    def get_pose(self, time_ns: int) -> np.ndarray:
        camera_calibration = self.get_camera_calibration(self.rgb_stream_id)
        pose_info = self.mps_data_provider.get_closed_loop_pose(time_ns)
        T_device_camera = camera_calibration.get_transform_device_camera()
        T_world_device = pose_info.transform_world_device # _core_pybinds.sophus.SE3
        T_world_camera = T_world_device @ T_device_camera

        return T_world_camera.to_matrix(), T_world_device.to_matrix()
    
    def get_focal_lengths(self, stream_id: StreamId) -> tuple:
        camera_calibration = self.get_camera_calibration(stream_id)
        f_px = camera_calibration.get_focal_lengths()[0]
        # scale the focal lengths to the cropped image size
        return f_px * (self.uncropped_width / (self.uncropped_width - 2 * self.crop_size))


    def get_depth_estimate_ml_pro(self, image):
        f_px = self.get_focal_lengths(self.rgb_stream_id)

        image_transformed = self.transform_depth_pro(image.copy()).to(device=self.device).to(torch.bfloat16)  # Transform the image for Depth Pro model
        prediction = self.model_depth_pro.infer(image_transformed, f_px=f_px)
        depth = prediction["depth"]  # Depth in [m].
        depth = depth.squeeze().to(torch.float32).cpu().numpy()  # Remove batch dimension and move to CPU

        return depth
    
    def get_depth_estimate_depth_anything(self, image):
        # Preprocess the image for Depth Anything V2
        depth = self.model_da2.infer_image(image)
        print(f"Max depth: {np.nanmax(depth):.2f}, Min depth: {np.nanmin(depth):.2f}")
        return depth.squeeze()  


    def get_psuedo_depth(self, pose: np.ndarray):
        T_camera_world = np.linalg.inv(pose)
        # Transform the points from world to camera coordinates
        points_cam_h = (T_camera_world @ self.points_world_h.T).T  
        points_cam   = points_cam_h[:, :3]
        # filter points that are behind the camera
        mask_front = points_cam[:, 2] > 0
        points_cam = points_cam[mask_front]

        f_px = self.get_focal_lengths(self.rgb_stream_id)
        x_pix =  (points_cam[:, 0] / points_cam[:, 2]) * f_px
        y_pix =  (points_cam[:, 1] / points_cam[:, 2]) * f_px
        cx, cy = self.width * 0.5, self.height * 0.5
        u_crop = x_pix + cx
        v_crop = y_pix + cy
        mask_in = (u_crop >= 0) & (u_crop < self.width) & (v_crop >= 0) & (v_crop < self.height)
        u_final = u_crop[mask_in]
        v_final = v_crop[mask_in]

        dist_xyz = np.linalg.norm(points_cam[mask_in], axis=1)
        depth_z  = points_cam[:, 2][mask_in]

        pseudo_depth = np.full((self.height, self.width), np.nan, dtype=np.float32)
        u_int = u_final.astype(np.int32)
        v_int = v_final.astype(np.int32)
        # If multiple 3-D points fall on the same pixel, keep the *nearest*
        for uu, vv, dz in zip(u_int, v_int, dist_xyz):   #  here actually the distace be better
            old = pseudo_depth[vv, uu]
            if np.isnan(old) or dz < old:
                pseudo_depth[vv, uu] = dz

        print(f"Max pseudo depth: {np.nanmax(pseudo_depth):.2f}, Min pseudo depth: {np.nanmin(pseudo_depth):.2f}, Valid points: {np.sum(~np.isnan(pseudo_depth))}")

        return pseudo_depth


    def correct_depth_with_psuedo_depth(self, depth_np: np.ndarray, pseudo_depth: np.ndarray):
        valid_mask = ~np.isnan(pseudo_depth)
        if np.sum(valid_mask) > 0:
            # Solve for parameters a (scale) and b (bias) in: pseudo_depth ≈ a * depth_np + b
            depth_valid = depth_np[valid_mask]
            pseudo_valid = pseudo_depth[valid_mask]

            mean_depth = np.mean(depth_valid)
            mean_pseudo = np.mean(pseudo_valid)
            numerator = np.sum((depth_valid - mean_depth) * (pseudo_valid - mean_pseudo))
            denominator = np.sum((depth_valid - mean_depth) ** 2)

            if denominator > 0:
                scale_factor = numerator / denominator
                bias_term = mean_pseudo - scale_factor * mean_depth
                depth_np = depth_np * scale_factor + bias_term
                print(f"Depth corrected with scale factor: {scale_factor:.4f} and bias: {bias_term:.4f} valid points: {np.sum(valid_mask)}")
                return depth_np
            else:
                print("Denominator for scale factor calculation is zero, skipping depth correction.")
                return depth_np
        else:
            print("No valid pseudo depth points found, skipping depth correction.")
            depth_vis = depth_np


    def next_frame(self):
        # get the next frame of the point cloud and the image
        if self.idx >= self.num_images:
            return None, None, None, None

        raw_image, timestamp_ns = self.get_undistorted_image(self.rgb_stream_id, self.idx)

        if raw_image is None:
            print("starting to look for valid sequence")
            for i in range(self.idx, self.num_images):
                raw_image, timestamp_ns = self.get_undistorted_image(self.rgb_stream_id, i)
                if raw_image is not None:
                    print("found valid id", i)
                    self.idx = i
                    break
            if raw_image is None:
                return None, None, None, None
            

        # check the blur
        blur_metric = cv2.Laplacian(raw_image, cv2.CV_64F).var()
        if blur_metric < self.blur_threshold:
            print(f"Skipping frame at time {timestamp_ns} due to high blur (metric: {blur_metric:.2f}).")
            self.idx += 1
            return self.next_frame()  # skip this frame

        ############ camera keypoints dynamix
        frame_set_uids = {}
        # get the left slam 
        print(timestamp_ns * 1e-9)
        uvs, uids = self.points_and_observations_manager.get_slam_observations(timestamp_ns, self.camera_serial_left_slam)
        frame_set_uids[str(LEFT_SLAM_STREAM_ID)] = uids
        uvs, uids = self.points_and_observations_manager.get_slam_observations(timestamp_ns, self.camera_serial_right_slam)
        frame_set_uids[str(RIGHT_SLAM_STREAM_ID)] = uids

        all_uids = set(frame_set_uids[str(LEFT_SLAM_STREAM_ID)]).union(
            frame_set_uids[str(RIGHT_SLAM_STREAM_ID)]
        )

        camera_calibration, device_pose = OnlineRgbCameraHelper(
            self.vrs_data_provider, self.mps_data_provider, timestamp_ns
        )
        visible_uvs, uids = self.points_and_observations_manager.get_rgb_observations(
            all_uids, camera_calibration, device_pose
        )
        # apply crop to the visible uvs and their uids
        # Adjust the visible keypoints for the cropped image.
        cropped_uvs = []
        cropped_pos_w = []
        for uv, uid in zip(visible_uvs, uids):
            cropped_uv = uv - self.crop_size
            # Keep only valid keypoints within the cropped image boundaries
            if (cropped_uv[0] < 0 or cropped_uv[0] >= self.width or
                cropped_uv[1] < 0 or cropped_uv[1] >= self.height):
                continue
            cropped_uvs.append(cropped_uv)
            cropped_pos_w.append(self.points_and_observations_manager.points[uid].position_world)

        print("valid keypoints:", len(cropped_uvs))

        # crop the borders by 50 pixels
        cropped_image = raw_image[self.crop_size:-self.crop_size, self.crop_size:-self.crop_size]


        # get the pose
        T_world_camera, T_world_device = self.get_pose(timestamp_ns)

        T_cam_world = np.linalg.inv(T_world_camera)
        pseudo_depth = np.full((self.height, self.width), np.nan, dtype=np.float32)
        for uv, pos_w in zip(cropped_uvs, cropped_pos_w):
            pos_w_h = np.append(pos_w, 1)
            pos_cam = T_cam_world @ pos_w_h
            depth_val = np.linalg.norm(pos_cam[:3])
            u, v = int(round(uv[0])), int(round(uv[1]))
            if 0 <= u < self.width and 0 <= v < self.height:
                current = pseudo_depth[v, u]
                if np.isnan(current) or depth_val < current:
                    pseudo_depth[v, u] = depth_val
        print(f"Created pseudo depth: max {np.nanmax(pseudo_depth):.2f}, min {np.nanmin(pseudo_depth):.2f}, valid {np.sum(~np.isnan(pseudo_depth))}")
        self.pseudo_depth = pseudo_depth

        psuedo_depth_zero = np.nan_to_num(pseudo_depth, nan=0)
        with torch.autocast(self.device, torch.bfloat16):
            depth = self.omni_dcnet(cropped_image, psuedo_depth_zero)

        self.idx += self.decimation_factor  # skip some frames to reduce the number of frames processed
        return cropped_image, depth, T_world_camera, timestamp_ns / 1000_000  # convert to ms
    

    def generate_pcd(self, color, depth_np, T_wc):
        
        h, w = depth_np.shape
        u = np.arange(w)
        v = np.arange(h)
        uu, vv = np.meshgrid(u, v)

        f_px = self.get_focal_lengths(self.rgb_stream_id)

        # Assume the principal point is located at the center of the cropped image
        cx, cy = w / 2.0, h / 2.0
        X = (uu - cx) * depth_np / f_px
        Y = (vv - cy) * depth_np / f_px
        Z = depth_np
        points_camera = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

        # Convert the 3D points from camera to world coordinates using the camera pose
        ones = np.ones((points_camera.shape[0], 1))
        points_camera_hom = np.concatenate((points_camera, ones), axis=1)
        points_world_hom = (T_wc @ points_camera_hom.T).T

        valid_mask = depth_np > 0

        # Apply the validity mask to the 3D points and color image
        points_camera_valid = np.stack((X, Y, Z), axis=-1)[valid_mask]
        ones = np.ones((points_camera_valid.shape[0], 1))
        points_camera_hom = np.concatenate((points_camera_valid, ones), axis=1)
        points_world_hom = (T_wc @ points_camera_hom.T).T
        pts = points_world_hom[:, :3]
        col = color[valid_mask]

        return pts, col
    


    def generate_pixel_indexed_pcd(self, color, depth, cam2world):
        
        h, w = depth.shape
        u = np.arange(w)
        v = np.arange(h)
        uu, vv = np.meshgrid(u, v)

        f_px = self.get_focal_lengths(self.rgb_stream_id)
        cx, cy = w / 2.0, h / 2.0

        # Compute 3D coordinates in camera frame for each pixel
        X = (uu - cx) * depth / f_px
        Y = (vv - cy) * depth / f_px
        Z = depth
        points_cam = np.stack((X, Y, Z), axis=-1)  # (H,W,3)

        # Convert to homogeneous coordinates
        ones = np.ones((h, w, 1), dtype=points_cam.dtype)
        points_cam_hom = np.concatenate([points_cam, ones], axis=-1)  # (H,W,4)

        # Reshape and transform points to world coordinates
        points_flat = points_cam_hom.reshape(-1, 4).T  # (4, H*W)
        points_world_flat = (cam2world @ points_flat).T  # (H*W,4)
        points_world = points_world_flat[:, :3].reshape(h, w, 3)

        # Combine world coordinates with color to form a pixel-indexed point cloud (H, W, 6)
        pixel_indexed_pcd = np.concatenate([points_world, color.astype(np.float32)], axis=-1)


        return pixel_indexed_pcd
    





if __name__ == "__main__":

    @dataclass
    class Config:
        path: str = "/path/to/vrs/data"
        vrs_slam_mapping_json: str = "vrs_slam_mapping.json"
        ml_depth_pro_checkpoint_uri: str = "conf/checkpoints/depthpro/depth_pro.pt"


    cfg = Config(path="dataset/HD-EPIC/VRS/P01/P01-20240203-184045_anonymized.vrs", vrs_slam_mapping_json="dataset/HD-EPIC/SLAM-and-Gaze/")
    vrs_loader = VRSLoader(cfg, device="cuda")

    rr.init("Aria Glasses", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP)
    rr.log("world/keypoints", rr.Points3D(vrs_loader.points_position, colors=[255, 0, 0], radii=0.001), static=True)

    while True:
        cropped_image, depth, T_wc, timestamp_ns = vrs_loader.next_frame()
        if cropped_image is None:
            break
        
        print(f"Processing frame at time {timestamp_ns} (index: {vrs_loader.idx})")

        rr.set_time("time", duration=timestamp_ns)

        # Log the image
        rr.log("Image", rr.Image(cropped_image))

        

        # Log the depth
        normalized_depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        rr.log("Depth", rr.Image(normalized_depth))

        # Log the psuedo depth
        pseudo_depth = vrs_loader.pseudo_depth
        # overlay the pseudo depth on the image
        pseudo_depth_vis = cv2.applyColorMap(
            cv2.normalize(pseudo_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        rr.log("PseudoDepth", rr.Image(pseudo_depth_vis))

        # # Log the pose
        # rr.log("Pose", rr.RigidTransform(T_wc))

        # Generate and log the point cloud
        pts, col = vrs_loader.generate_pcd(cropped_image, depth, T_wc)

        rr.log("world/point_cloud", rr.Points3D(pts, colors=col))


        # log the palm and wrist pose
        hand = vrs_loader.hand
        if hand is not None:
            if hand["left_palm"] is not None:
                rr.log("world/left_palm", rr.Points3D(hand["left_palm"], colors=[0, 255, 0], radii=0.05))
            if hand["left_wrist"] is not None:
                rr.log("world/left_wrist", rr.Points3D(hand["left_wrist"], colors=[0, 255, 0], radii=0.05))
            if hand["right_palm"] is not None:
                rr.log("world/right_palm", rr.Points3D(hand["right_palm"], colors=[255, 0, 0], radii=0.05))
            if hand["right_wrist"] is not None:
                rr.log("world/right_wrist", rr.Points3D(hand["right_wrist"], colors=[255, 0, 0], radii=0.05))

