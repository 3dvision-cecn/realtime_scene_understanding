"""
Preprocess an unzipped .r3d file to the Record3DDataset format with adjusted image resolution.
"""

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

# from promptda.promptda import PromptDA


def load_depth(filepath, desired_width=960, desired_height=720):
    with open(filepath, 'rb') as depth_fh:
        raw_bytes = depth_fh.read()
        decompressed_bytes = liblzfse.decompress(raw_bytes)
        depth_img = np.frombuffer(decompressed_bytes, dtype=np.float32)
        depth_img = depth_img.reshape((192, 256))  # Original resolution
        depth_img = cv2.resize(depth_img, (desired_width, desired_height), interpolation=cv2.INTER_LINEAR)
    return depth_img

def load_color(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


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


class R3D_loader:
    def __init__(self, cfg):
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

        # self.promptda = PromptDA(encoder = "vits", ckpt_path = "conf/checkpoints/promptda/model(1).ckpt").to("cuda").eval()



    def get_intrinsics(self):
        return self.intrinsics_dict
    

    def next_frame(self):

        if self.frame_idx >= len(self.color_paths):
            return None, None, None, None

        color = load_color(self.color_paths[self.frame_idx])
        depth = load_depth(self.depth_paths[self.frame_idx])

        # neural_depth = self.neural_depth(color, depth)


        # pose
        pose = self.poses[self.frame_idx]

        self.frame_idx += 1

        return color, depth, pose, self.frame_idx / 30.0  # assuming 30 fps
    

    # def neural_depth(self, color, depth):
    #     # max_size // 14 = 0
    #     # ensure color and depth max size is multiple of 14
    #     h, w = color.shape[:2]
    #     new_h = h - (h % 14)
    #     new_w = w - (w % 14)
    #     color = color[:new_h, :new_w]
    #     depth = depth[:new_h, :new_w]

    #     print(f"mean depth: {np.mean(depth)}, min depth: {np.min(depth)}, max depth: {np.max(depth)}")

    #     # neural depth prediction
    #     color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)  # convert to BGR for OpenCV
    #     color_t = torch.tensor(color).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    #     color_t = color_t.to("cuda")
    #     depth_t = torch.tensor(depth).unsqueeze(0).unsqueeze(0).float()
    #     depth_t = depth_t.to("cuda")
    #     print(f"Color shape: {color_t.shape}, Depth shape: {depth_t.shape}")
    #     depth_neural = self.promptda.predict(color_t, depth_t)
    #     print(f"Depth neural shape: {depth_neural.shape}")

    #     print(f"mean neural depth: {torch.mean(depth_neural)}, min neural depth: {torch.min(depth_neural)}, max neural depth: {torch.max(depth_neural)}")

    #     # convert to numpy and squeeze
    #     depth_neural = depth_neural.squeeze().cpu().numpy()
    #     # resize to original size
    #     depth_neural = cv2.resize(depth_neural, (w, h), interpolation=cv2.INTER_LINEAR)

    #     return depth_neural



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

        # select only 1000 points for logging
        if len(pts) > 50000:
            idx = np.random.choice(len(pts), 50000, replace=False)
            pts = pts[idx]
            cols = cols[idx]

        return pts, cols



    def generate_pixel_indexed_pcd(self, color, depth, cam2world_hom):
        # create an Open3D RGBD image
        intr = o3d.camera.PinholeCameraIntrinsic(
            self.intrinsics_dict["w"], self.intrinsics_dict["h"],
            self.intrinsics_dict["fx"], self.intrinsics_dict["fy"],
            self.intrinsics_dict["cx"], self.intrinsics_dict["cy"],
        )

        if color.shape[0] * color.shape[1] != depth.shape[0] * depth.shape[1]:
            print(f"Warning: color shape {color.shape} does not match depth shape {depth.shape}.")
            return None

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

    # Example usage
    conf = Config(path="/home/eongan/dropbox/recordings/efe_kitchen_tea/2025-05-04--19-55-15")
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


        # log the camera pose in world coordinates
        # create an Open3D RGBD image
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

        # select only 1000 points for logging
        # if len(pts) > 1000:
        #     idx = np.random.choice(len(pts), 1000, replace=False)
        #     pts = pts[idx]
        #     cols = cols[idx]

        rr.log("world/point_cloud", rr.Points3D(pts, colors=cols))






