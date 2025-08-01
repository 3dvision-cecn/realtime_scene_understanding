from data_source import R3D_loader, R3D_loaderConfig
import tyro
from dataclasses import dataclass, field
from typing import List
from tqdm import tqdm
import models
import torch

@dataclass
class BenchmarkingConfig:
    red_folders: List[str] = field(default_factory=lambda: ["dataset/recordings/train/01"])
    decimation_factor: int = 1  # decimation factor for the frames, default is 1 (no decimation)
    model: str = "MLDepthEstimator"  # models to benchmark, default is MLDepthEstimator



def main(config: BenchmarkingConfig):
    print(f"Using folders: {config.red_folders}")
    print(f"Models to benchmark: {config.model}")

    # initialize the depth estimator
    model_class = getattr(models, config.model, None)
    depth_estimator = model_class(device="cuda" if torch.cuda.is_available() else "cpu")

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
            depth_map = depth_estimator.process_image(image)
            




if __name__ == "__main__":
    config = tyro.cli(BenchmarkingConfig)
    main(config)
