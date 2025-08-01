from data_source import R3D_loader, R3D_loaderConfig
import tyro
from dataclasses import dataclass, field
from typing import List
from tqdm import tqdm
import models
import torch
import rerun as rr
import cv2

@dataclass
class BenchmarkingConfig:
    red_folders: List[str] = field(default_factory=lambda: ["dataset/recordings/train/01"])
    decimation_factor: int = 1  # decimation factor for the frames, default is 1 (no decimation)
    model: str = "MLDepthEstimator"  # models to benchmark, default is MLDepthEstimator



def main(config: BenchmarkingConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using folders: {config.red_folders}")
    print(f"Models to benchmark: {config.model}")

    # Initialize rerun
    rr.init("3d_vision_benchmark", spawn=True)
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
            image, depth, _, frame = data_source.next_frame()
            if image is None:
                break
            # get the depth map using the models
            predicted_depth = depth_estimator.process_image(image)
            
            l1_loss = torch.nn.functional.l1_loss(
                torch.tensor(predicted_depth, device=device, dtype=torch.float32),
                torch.tensor(depth, device=device, dtype=torch.float32)
            )

            print(f"Frame {frame}: L1 Loss = {l1_loss.item()}")
            print(f"min(predicted_depth): {predicted_depth.min()}, max(predicted_depth): {predicted_depth.max()}")
            print(f"min(depth): {depth.min()}, max(depth): {depth.max()}")

            # apply jet colormap to the depth maps for visualization
            # Normalize the predicted depth for visualization
            predicted_norm = cv2.normalize(predicted_depth, None, 0, 255, cv2.NORM_MINMAX)
            predicted_norm = predicted_norm.astype('uint8')
            predicted_color = cv2.applyColorMap(predicted_norm, cv2.COLORMAP_JET)

            # Normalize the ground truth depth for visualization
            depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
            depth_norm = depth_norm.astype('uint8')
            depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

            # Log the colorized depth maps
            rr.log("predicted_depth_colormap", rr.Image(predicted_color).compress(jpeg_quality=95))
            rr.log("ground_truth_depth_colormap", rr.Image(depth_color).compress(jpeg_quality=95))

            ## Rerun logging
            rr.set_time("time", timestamp=frame)
            rr.log("rgb", rr.Image(image).compress(jpeg_quality=95))


if __name__ == "__main__":
    config = tyro.cli(BenchmarkingConfig)
    main(config)
