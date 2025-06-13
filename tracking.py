import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from hydra.core.global_hydra import GlobalHydra
import hydra
from omegaconf import DictConfig
from transformers import AutoProcessor, AutoModelForVision2Seq
import rerun as rr

import torch
import open3d as o3d
from scipy.spatial.transform import Rotation





# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def prepare_sam(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = build_sam2(cfg.model_cfg, "conf/" + cfg.model_path, device=device, apply_postprocessing=False)
    sam = torch.compile(sam, mode="reduce-overhead", fullgraph=True)

    return SAM2AutomaticMaskGenerator(
        sam,
        points_per_side=32,
        pred_iou_thresh=0.9,
        stability_score_thresh=0.9,
        min_mask_region_area=250,
        points_per_batch=256,
    )


def remove_outliers_pcd(pcd: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    # create a o3d point cloud
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd[:, :3])
    pcd_o3d.colors = o3d.utility.Vector3dVector(pcd[:, 3:6])
    # knn search for neighbors
    labels = np.asarray(pcd_o3d.cluster_dbscan(eps=threshold, min_points=10, print_progress=False))
    labels_noise = np.where(labels == -1, 1, 0)
    labels_clean = np.where(labels == 0, 1, 0)   
    num_clean = np.sum(labels_clean)

    pcd_np = np.zeros(((num_clean), 6), dtype=np.float32)
    pcd_np[:, :3] = np.asarray(pcd_o3d.points)[labels_clean == 1]
    pcd_np[:, 3:6] = np.asarray(pcd_o3d.colors)[labels_clean == 1]
    return pcd_np


# -----------------------------------------------------------------------------
# Vision-Language Model (caption/identify objects)
# -----------------------------------------------------------------------------

class VLM:
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.processor = AutoProcessor.from_pretrained('HuggingFaceTB/SmolVLM-Instruct')
        self.model = AutoModelForVision2Seq.from_pretrained(
            'HuggingFaceTB/SmolVLM-Instruct', torch_dtype=torch.bfloat16
        ).to(self.device)

    def ask(self, frame: np.ndarray, center: tuple[int,int], radius: int = 64) -> str:
        x, y = center
        # draw a red circle on the image
        # draw a red circle on a copy of the frame
        circ = frame.copy()
        cv2.circle(circ, (x, y), radius, (0, 0, 255), 2)
        # crop the image to the circle
        offset = int(radius + 5)
        x0, y0 = max(x - offset, 0), max(y - offset, 0)
        x1, y1 = min(x + offset, frame.shape[1]), min(y + offset, frame.shape[0])
        circ = circ[y0:y1, x0:x1]

        # uf cropping the image, the circle is not centered return unknown
        if circ.shape[0] != circ.shape[1]:
            cv2.putText(circ, "unknown", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Circle", circ)
            cv2.waitKey(100)
            return "unknown"

        # show the image with the circle

        messages = [
            {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is the inanimate object in the middle of the red circle, in maximum two words !"},
            ]
            },
        ]
        # convert the image to RGB
        circ = cv2.cvtColor(circ, cv2.COLOR_BGR2RGB)
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[circ], return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=20)
        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )


        asssitant = generated_texts[0].split("Assistant:")[-1]

        cv2.putText(circ, asssitant.strip(), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Circle", circ)
        cv2.waitKey(100)

        return asssitant.strip()

def mask_radius(mask: np.ndarray) -> int:
    """
    Compute the radius of the smallest circle centered at the mask centroid
    that covers all mask pixels.
    """
    # find centroid
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0
    cx, cy = xs.mean(), ys.mean()
    # compute distances from centroid to every mask pixel
    dists = np.sqrt((xs - cx)**2 + (ys - cy)**2) + 10
    # round up to an integer radius
    return int(np.ceil(dists.max()))


# -----------------------------------------------------------------------------
# Database for Objects
# -----------------------------------------------------------------------------

class Object:
    def __init__(self, unqiue_id: int, name: str, voxels: np.ndarray, centroids: np.ndarray):
        self.unique_id = unqiue_id
        self.name = name
        self.voxels = voxels
        self.centroids = centroids




class ObjectDatabase:
    def __init__(self):
        self.objects = {}
        self.next_id = 0

    def add_object(self, name: str, voxels: np.ndarray, centroids: np.ndarray):
        # Check for existing object by voxel overlap
        match_id = self.find_matching_object(voxels, centroids, 0.15)
        if match_id is not None:
            # Optionally update the object voxels here if you want
            unique_voxels_map = {tuple(v.grid_index): v for v in self.objects[match_id].voxels}
            # Add/update with the new voxels passed to this method
            for new_voxel in voxels: 
                unique_voxels_map[tuple(new_voxel.grid_index)] = new_voxel
            self.objects[match_id].voxels = np.asarray(list(unique_voxels_map.values()))
            return match_id
        # Add as new object
        obj = Object(self.next_id, name, voxels, centroids)
        self.objects[self.next_id] = obj
        self.next_id += 1
        return self.next_id - 1

    def get_neighbors(self, index):
        x, y, z = index
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbors.append((x+dx, y+dy, z+dz))
        return neighbors

    def expand_voxel_set(self, vox_set):
        expanded = set(vox_set)
        for idx in vox_set:
            expanded.update(self.get_neighbors(idx))
        return expanded

    def find_matching_object(self, voxels: np.ndarray, centeroids: np.ndarray, overlap_thresh: float = 0.3):
        """Return object id if overlap with any object is above threshold, else None.
        Considers neighboring voxels in overlap calculation."""
        if len(voxels) == 0:
            return None

        # Convert to set of tuples for fast overlap computation
        vox_set = set(tuple(v.grid_index) for v in voxels)
        # Expand with neighbors
        vox_set_expanded = self.expand_voxel_set(vox_set)

        for obj_id, obj in self.objects.items():
            # if centroid distance is too far, skip
            dist = np.linalg.norm(obj.centroids - centeroids)
            if dist > 0.2:
                continue

            obj_vox_set = set(tuple(v.grid_index) for v in obj.voxels)
            obj_vox_set_expanded = self.expand_voxel_set(obj_vox_set)

            # Compute overlap on expanded sets
            intersection = len(vox_set_expanded & obj_vox_set_expanded)
            union = len(vox_set_expanded | obj_vox_set_expanded)
            if union == 0:
                continue
            overlap = intersection / union
            print(f"Overlap with object {obj_id}: {overlap:.2f} Num inersection: {intersection} Num union: {union}")
            if overlap > overlap_thresh:
                return obj_id
        return None


    def get_object(self, unique_id: int):
        return self.objects.get(unique_id)
    

    def remove_object(self, unique_id: int):
        if unique_id in self.objects:
            del self.objects[unique_id]
            


# -----------------------------------------------------------------------------
# Main with VLM object labeling
# -----------------------------------------------------------------------------

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    maskgen = prepare_sam(cfg.segmentation)
    vlm = VLM(device)

    reid_db = ObjectDatabase()
    
    prev_segs, id_to_color, id_to_label = {}, {}, {}


    from red_loader import R3D_loader
    video_loader = R3D_loader(cfg.video)

    rr.init("video_stream", spawn=True)  # spawn=True ⇒ open viewer
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    frames = []

    decimate = 4
    couter = 0

    while True:
        frame, depth, pose, timestamp = video_loader.next_frame()
        if frame is None:
            print("No more frames to process.")
            break
        # pixel indexed pcd has shape (h, w, 6) 3d points + 3d colors
        pixel_indexed_pcd = video_loader.generate_pixel_indexed_pcd(frame, depth, pose)
        if pixel_indexed_pcd is None:
            continue

        couter += 1
        if couter % decimate != 0:
            continue

        rr.set_time("time", duration=timestamp)
        rr.log("raw_video/frame", rr.Image(frame).compress(jpeg_quality=85))

        # generate masks
        masks = maskgen.generate(frame)

        # seperate the pcd into segments of masks
        pcd_segments = []
        for mask in masks:
            mask = mask['segmentation']
            args = np.where(mask)
            pcd_segment = pixel_indexed_pcd[args]
            pcd_segments.append(pcd_segment)

            

        in_frame_ids = []
        # for each segment, compute the centroid and radius
        for i, pcd_segment in enumerate(pcd_segments):
            filtered_pcd = remove_outliers_pcd(pcd_segment, threshold=0.005)
            if len(filtered_pcd) <= 100 or len(filtered_pcd) >= 30000:
                continue

            points = filtered_pcd[:, :3]
            colors = filtered_pcd[:, 3:]
            # compute the centroid
            centroid = points.mean(axis=0)
            # filter the pointcloud with
            # random color
            colors[:,:] = np.random.rand(3)

            # fit a bounding box to the pcd segment
            pcd_o3d = o3d.geometry.PointCloud()
            pcd_o3d.points = o3d.utility.Vector3dVector(points)
            # Fit an oriented bounding box
            obb = pcd_o3d.get_oriented_bounding_box()
            # Log the OBB to Rerun
            # Convert rotation matrix to quaternion (xyzw)
            rotation_matrix = obb.R
            quat_xyzw = Rotation.from_matrix(rotation_matrix).as_quat()

            rr.log(f"world/pcd/segment_{i}/obb", rr.Boxes3D(
                centers=obb.center,
                half_sizes=obb.extent / 2.0,
                rotations=quat_xyzw, # Pass quaternion here
                colors=colors.mean(axis=0), # Use mean color of the segment
                labels=f"ID: {i}"
            ))

            # voxelize the pcd segment
            voxel_size = 0.02  # Define the size of a voxel
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_o3d, voxel_size=voxel_size)
            # Get the voxel centers and colors
            voxels = voxel_grid.get_voxels()
            voxel_centers = np.array([voxel_grid.get_voxel_center_coordinate(voxel.grid_index) for voxel in voxels])
            
            # For simplicity, let's assign a uniform color or average color to voxels
            # If you have per-voxel color, you can extract it similarly
            voxel_colors = np.tile(colors.mean(axis=0), (len(voxel_centers), 1))

            # if len(voxel_centers) > 0:
            #     rr.log(f"world/pcd/segment_{i}/voxels", rr.Points3D(
            #         positions=voxel_centers,
            #         colors=voxel_colors,
            #         radii=voxel_size / 2.0 # Approximate radius for visualization
            #     ))

            # add the segment to database
            voxels_np = np.asarray(voxels)
            obj_id = reid_db.add_object("unknown", voxels_np, centroid)
            in_frame_ids.append(obj_id)

            # add it to rerun
            rr.log(f"world/pcd/centroid_{obj_id}", rr.Points3D(positions=centroid.reshape(1, 3), 
                                                               colors=colors.mean(axis=0).reshape(1, 3), 
                                                               radii=0.05, 
                                                               labels=f"ID: {obj_id}",
                                                               show_labels=True))
            # add the pcd segment to rerun
            rr.log(f"world/pcd/segment_{obj_id}", rr.Points3D(positions=points, colors=colors, radii=0.01))





if __name__=="__main__":
    GlobalHydra.instance().clear(); main()
