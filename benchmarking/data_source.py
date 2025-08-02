import glob
import json
import os
import cv2
import liblzfse
import numpy as np
import png
import tyro
import yaml
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm, trange
from typing import List, Tuple, Union
from natsort import natsorted
from pathlib import Path
from scipy.spatial.transform import Rotation
import rerun as rr
import time
import open3d as o3d
import torch


def load_depth(filepath, is_rotated, desired_width=960, desired_height=720):
    with open(filepath, 'rb') as depth_fh:
        raw_bytes = depth_fh.read()
        decompressed_bytes = liblzfse.decompress(raw_bytes)
        depth_img = np.frombuffer(decompressed_bytes, dtype=np.float32)
        if is_rotated:
            depth_img = depth_img.reshape((256, 192))
            # rotate the image 90 degrees counterclockwise
            depth_img = np.rot90(depth_img, k=1)
        else:
            depth_img = depth_img.reshape((192, 256))  # Original resolution
        depth_img = cv2.resize(depth_img, (desired_width, desired_height), interpolation=cv2.INTER_LINEAR)
        
    return depth_img

def load_color(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print the shape of the image
    # if shape is 960x720 rotate it 90 degrees clockwise
    is_rotated = False
    if img.shape[:2] == (960, 720):
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        is_rotated = True
    
    return img, is_rotated


def get_poses(metadata_dict: dict) -> int:
    """Converts Record3D's metadata dict into pose matrices needed by nerfstudio
    Args:
        metadata_dict: Dict containing Record3D metadata
    Returns:
        np.array of pose matrices for each image of shape: (num_images, 4, 4)
    """

    poses_data = np.array(metadata_dict["poses"])  

    pos = poses_data[:, -3:]  # x, y, z
    quat = poses_data[:, :4]  # qx, qy, qz, qw
    rot = Rotation.from_quat(quat).as_matrix()  # Convert quaternion to rotation matrix

    camera_to_worlds = np.zeros((len(pos), 4, 4))
    camera_to_worlds[:, :3, :3] = rot
    camera_to_worlds[:, :3, 3] = pos
    camera_to_worlds[:, 3, 3] = 1.0

    return camera_to_worlds


def get_intrinsics(metadata_dict: dict, downscale_factor: float = 1.0) -> int:
    """Converts Record3D metadata dict into intrinsic info needed by nerfstudio
    Args:
        metadata_dict: Dict containing Record3D metadata
        downscale_factor: factor to scale RGB image by (usually scale factor is
            set to 7.5 for record3d 1.8 or higher -- this is the factor that downscales
            RGB images to lidar)
    Returns:
        dict with camera intrinsics keys needed by nerfstudio
    """

    # Camera intrinsics
    K = np.array(metadata_dict["K"]).reshape((3, 3)).T
    K = K / downscale_factor
    K[2, 2] = 1.0
    focal_length = K[0, 0]

    H = metadata_dict["h"]
    W = metadata_dict["w"]

    H = int(H / downscale_factor)
    W = int(W / downscale_factor)

    # # TODO(akristoffersen): The metadata dict comes with principle points,
    # # but caused errors in image coord indexing. Should update once that is fixed.
    # cx, cy = W / 2, H / 2
    cx, cy = K[0, 2], K[1, 2]

    intrinsics_dict = {
        "fx": focal_length,
        "fy": focal_length,
        "cx": cx,
        "cy": cy,
        "w": W,
        "h": H,
    }

    return intrinsics_dict

class R3D_loaderConfig:
    def __init__(self, path: str, decimation_factor: int = 1):
        self.path = path
        self.decimation_factor = decimation_factor

class R3D_loader:
    def __init__(self, cfg: R3D_loaderConfig):
        self.cfg = cfg
        self.datapath = cfg.path
        
        # get the metadata
        with open(os.path.join(self.datapath, "metadata"), "r") as f:
            metadata = json.load(f)

        # get the intrinsics dict
        self.intrinsics_dict = get_intrinsics(metadata) 
        self.frame_idx = 0
        # get the color, depth, conf and pose paths
        self.color_paths = natsorted(glob.glob(os.path.join(self.datapath, "rgbd", "*.jpg")))
        self.depth_paths = natsorted(glob.glob(os.path.join(self.datapath, "rgbd", "*.depth")))

        self.poses = get_poses(metadata)
        self.decimation_factor = cfg.decimation_factor


    def num_frames(self):
        return len(self.color_paths) // self.decimation_factor


    def get_intrinsics(self):
        return self.intrinsics_dict
    

    def next_frame(self):

        if self.frame_idx >= len(self.color_paths):
            return None, None, None, None

        color, is_rotated = load_color(self.color_paths[self.frame_idx])
        depth = load_depth(self.depth_paths[self.frame_idx], is_rotated)
        pose = self.poses[self.frame_idx]
        self.frame_idx += self.decimation_factor

        return color, depth, pose, self.frame_idx / 30.0  # assuming 30 fps
    
    def generate_pcd(self, color, depth, cam2world_hom):
        # create an Open3D RGBD image
        intr = o3d.camera.PinholeCameraIntrinsic(
            self.intrinsics_dict["w"], self.intrinsics_dict["h"],
            self.intrinsics_dict["fx"], self.intrinsics_dict["fy"],
            self.intrinsics_dict["cx"], self.intrinsics_dict["cy"],
        )
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color),
            o3d.geometry.Image(depth * 1000),  # convert depth to mm
            convert_rgb_to_intensity=False,
        )


        # backproject to a point cloud and transform into world coords
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)
        flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        pcd.transform(flip_transform)

        # transform the point cloud into world coordinates
        pcd.transform(cam2world_hom)
        
        # extract numpy arrays and log to rerun
        pts = np.asarray(pcd.points)
        cols = np.asarray(pcd.colors)
        return pts, cols



    def generate_pixel_indexed_pcd(self, color, depth, cam2world_hom, intrinsics=None):
        # create an Open3D RGBD image
        intr = None
        if intrinsics is None:
            intr = o3d.camera.PinholeCameraIntrinsic(
                self.intrinsics_dict["w"], self.intrinsics_dict["h"],
                self.intrinsics_dict["fx"], self.intrinsics_dict["fy"],
                self.intrinsics_dict["cx"], self.intrinsics_dict["cy"],
            )
        else:
            intr = o3d.camera.PinholeCameraIntrinsic(
                intrinsics["w"], intrinsics["h"],
                intrinsics["fx"], intrinsics["fy"],
                intrinsics["cx"], intrinsics["cy"],
            )

        if color.shape[0] * color.shape[1] != depth.shape[0] * depth.shape[1]:
            return None
        print(f"Depth 02: {depth.shape}")

        # Ensure depth is float and has no invalid values
        depth = np.nan_to_num(depth, nan=0.0)
        depth[depth <= 0] = 0.001  # Set very small positive values instead of zero
        
        # Set a very large depth_trunc to avoid clipping
        depth_trunc = 1e6  # 1,000,000 mm = 1,000 meters

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color),
            o3d.geometry.Image(depth * 1000),  # convert depth to mm
            convert_rgb_to_intensity=False,
            depth_trunc= 10000.0

        )

        # backproject to a point cloud and transform into world coords
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)
        print(f"PCD: ")
        flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        pcd.transform(flip_transform)

        # transform the point cloud into world coordinates
        pcd.transform(cam2world_hom)
        
        # extract numpy arrays and log to rerun
        pts = np.asarray(pcd.points)
        cols = np.asarray(pcd.colors)

        # create an array of pixel indices
        h, w = color.shape[:2]
        h_depth, w_depth = depth.shape[:2]

        if len(pts) != h * w:
            print(f"Warning: number of points {len(pts)} does not match number of pixels {h * w}.")
            return None

        pixel_indexed_pcd = np.zeros((h, w, 6), dtype=np.float32)
        pixel_indexed_pcd[:, :, :3] = pts.reshape(h, w, 3)
        pixel_indexed_pcd[:, :, 3:] = cols.reshape(h, w, 3)

        return pixel_indexed_pcd



if __name__ == "__main__":
    
    @dataclass
    class Config:
        path: str  # path to the unzipped .r3d folder
        decimation_factor: int = 1  # decimation factor for the frames, default is 1 (no decimation)

    # Example usage
    conf = Config(path="dataset/recordings/train/01", decimation_factor=1)
    loader = R3D_loader(conf)
    rr.init("3d_vision_demo", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    while True:
        color, depth, cam2world_hom, timestamp = loader.next_frame()
        if color is None:
            break

        rr.set_time("time", timestamp=timestamp)
        rr.log("rgb", rr.Image(color).compress(jpeg_quality=95))
        rr.log("depth", rr.DepthImage(depth))


        intr = o3d.camera.PinholeCameraIntrinsic(
            loader.intrinsics_dict["w"], loader.intrinsics_dict["h"],
            loader.intrinsics_dict["fx"], loader.intrinsics_dict["fy"],
            loader.intrinsics_dict["cx"], loader.intrinsics_dict["cy"],
        )
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color),
            o3d.geometry.Image(depth * 1000),  # convert depth to mm
            convert_rgb_to_intensity=False,
        )


        # backproject to a point cloud and transform into world coords
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)
        flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        pcd.transform(flip_transform)

        # transform the point cloud into world coordinates
        pcd.transform(cam2world_hom)
        
        # extract numpy arrays and log to rerun
        pts = np.asarray(pcd.points)
        cols = np.asarray(pcd.colors)
        rr.log("world/point_cloud", rr.Points3D(pts, colors=cols))




