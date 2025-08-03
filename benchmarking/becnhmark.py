from data_source import R3D_loader, R3D_loaderConfig
import tyro
from dataclasses import dataclass, field
from typing import List
from tqdm import tqdm
import models
import torch
import rerun as rr
import cv2
import numpy as np

@dataclass
class BenchmarkingConfig:
    red_folders: List[str] = field(default_factory=lambda: ["dataset/recordings/train/01"])
    decimation_factor: int = 1  # decimation factor for the frames, default is 1 (no decimation)
    model: str = "Metric3DONNXEstimatorLarge"  # models to benchmark, default is MLDepthEstimator
    spawn_rerun: bool = True  # whether to spawn rerun or not


def main(config: BenchmarkingConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using folders: {config.red_folders}")
    print(f"Models to benchmark: {config.model}")

    # Initialize rerun
    rr.init("3d_vision_benchmark", spawn=config.spawn_rerun)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)


    # initialize the depth estimator
    model_class = getattr(models, config.model, None)
    depth_estimator = model_class(device=device)

    for red_folder in config.red_folders:
        red_loader_conf = R3D_loaderConfig(
            path=red_folder,
            decimation_factor=config.decimation_factor
        )
        data_source = R3D_loader(red_loader_conf)
        num_frames = data_source.num_frames()
        
        for _ in tqdm(range(num_frames)):
            image, depth, pose, frame = data_source.next_frame()
            print(f"Depth shape: {depth.shape}")

            # get the point cloud from the RGBD image
            pixel_indexed_pcd_gt = data_source.generate_pixel_indexed_pcd(image, depth, pose)
            gt_points = pixel_indexed_pcd_gt[:, :, :3]
            gt_colors = pixel_indexed_pcd_gt[:, :, 3:]

            if image is None:
                break
            # get the depth map using the models
            predicted_depth, pred_intr = depth_estimator.process_image(image)
            print(f"Predicted Depth shape: {predicted_depth.shape}")


            pixel_indexed_pcd_pred = data_source.generate_pixel_indexed_pcd(image, predicted_depth, pose, pred_intr)
            if pixel_indexed_pcd_pred is None:
                print(f"Warning: Skipping frame {frame} due to point cloud generation failure")
                continue
            pred_points = pixel_indexed_pcd_pred[:, :, :3]
            pred_colors = pixel_indexed_pcd_pred[:, :, 3:]
            
            # Filter out NaN values for point cloud visualization
            valid_pred_mask = np.isfinite(pred_points).all(axis=2)
            
            # l1 error between predicted and ground truth depth
            l1_error = torch.nn.functional.l1_loss(
                torch.tensor(predicted_depth, device=device, dtype=torch.float32),
                torch.tensor(depth, device=device, dtype=torch.float32)
            )
            # abs relative error
            abs_rel_error = np.mean(np.abs(predicted_depth - depth) / (depth + 1e-6))


            print(f"Frame {frame}: L1 error: {l1_error.item()}, Absolute Relative Error: {abs_rel_error.item()}")

            # apply jet colormap to the depth maps for visualization
            # Normalize the predicted depth for visualization
            predicted_norm = cv2.normalize(predicted_depth, None, 0, 255, cv2.NORM_MINMAX)
            predicted_norm = predicted_norm.astype('uint8')
            predicted_color = cv2.applyColorMap(predicted_norm, cv2.COLORMAP_JET)

            # Normalize the ground truth depth for visualization
            depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
            depth_norm = depth_norm.astype('uint8')
            depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

            # create a l1 error image
            l1_error_image = np.abs(predicted_depth - depth)
            l1_error_norm = cv2.normalize(l1_error_image, None, 0, 255, cv2.NORM_MINMAX)
            l1_error_norm = l1_error_norm.astype('uint8')


            ## Rerun logging
            rr.set_time("time", timestamp=frame)

            rr.log("l1_error", rr.Image(l1_error_norm).compress(jpeg_quality=95))
            rr.log("predicted_depth_colormap", rr.Image(predicted_color).compress(jpeg_quality=95))
            rr.log("ground_truth_depth_colormap", rr.Image(depth_color).compress(jpeg_quality=95))
            rr.log("rgb", rr.Image(image).compress(jpeg_quality=95))

            gt_points = gt_points.reshape(-1, 3)
            gt_colors = gt_colors.reshape(-1, 3)
            
            # Filter out NaN values from predicted points
            pred_points_flat = pred_points.reshape(-1, 3)
            pred_colors_flat = pred_colors.reshape(-1, 3)
            valid_pred_indices = np.isfinite(pred_points_flat).all(axis=1)
            pred_points_valid = pred_points_flat[valid_pred_indices]
            pred_colors_valid = pred_colors_flat[valid_pred_indices]

            if len(gt_points) > 30000:
                # Sample from ground truth
                gt_indices = np.random.choice(len(gt_points), size=min(100000, len(gt_points)), replace=False)
                gt_points = gt_points[gt_indices]
                gt_colors = gt_colors[gt_indices]
                
                # Sample from valid predicted points
                if len(pred_points_valid) > 30000:
                    pred_indices = np.random.choice(len(pred_points_valid), size=min(100000, len(pred_points_valid)), replace=False)
                    pred_points_valid = pred_points_valid[pred_indices]
                    pred_colors_valid = pred_colors_valid[pred_indices]

            print(f"Logging {len(gt_points)} GT points and {len(pred_points_valid)} predicted points")
            rr.log("world/point_cloud_gt", rr.Points3D(gt_points, colors=gt_colors))
            rr.log("world/point_cloud_pred", rr.Points3D(pred_points_valid, colors=pred_colors_valid))

if __name__ == "__main__":
    config = tyro.cli(BenchmarkingConfig)
    main(config)
