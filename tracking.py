import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from hydra.core.global_hydra import GlobalHydra
import hydra
from omegaconf import DictConfig
from transformers import AutoProcessor, AutoModelForVision2Seq
from ultralytics import FastSAM

VIDEO_SRC = "video.mp4"  # 0 for webcam
MODEL_CFG = "checkpoints/sam/sam2.1_hiera_l.yaml"
CHECKPOINT_PATH = "conf/checkpoints/sam/sam2.1_hiera_large.pt"
MASK_ALPHA = 0.8
EDGE_OFFSET = 5
EMBED_SIM_THRESH = 0.8  # cosine threshold for ResNet embeddings
IOU_THRESH = 0.1        # minimum IoU for accepting new segmentation
FRAME_BLUR_THRESH = 15.0  # variance threshold for whole-frame blur
MAX_RADIUS = 250  # maximum radius for VLM asking
MIN_RADIUS = 50  # minimum radius for VLM asking
# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def prepare_sam(cfg_path: str, ckpt_path: str, device: str):
    sam = build_sam2(cfg_path, ckpt_path, device=device, apply_postprocessing=False)
    sam = torch.compile(sam, mode="reduce-overhead", fullgraph=True)

    return SAM2AutomaticMaskGenerator(
        sam,
        points_per_side=32,
        pred_iou_thresh=0.9,
        stability_score_thresh=0.9,
        min_mask_region_area=250,
        points_per_batch=256,
    )


def dilate_mask(mask: np.ndarray, offset: int) -> np.ndarray:
    if offset <= 0:
        return mask
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (offset * 2 + 1, offset * 2 + 1)
    )
    return cv2.dilate(mask, kernel)


def mask_centroid(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return (int(xs.mean()), int(ys.mean()))


def blend_masks(image: np.ndarray, masks: list, alpha: float = 0.45) -> np.ndarray:
    overlay = np.zeros_like(image, dtype=np.uint8)
    for m in masks:
        overlay[m['segmentation']] = m.get('color', (0, 255, 0))
    return cv2.addWeighted(image, 1.0, overlay, alpha, 0)


def compute_hsv_hist(frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1,2], mask, [8,8,8], [0,180,0,256,0,256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist.astype(np.float32)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union > 0 else 0.0

# -----------------------------------------------------------------------------
# ResNet-50 embedder (RGB, motion-blur tolerant)
# -----------------------------------------------------------------------------

def build_embedder(device: str):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval().to(device)
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return model, preprocess


def extract_embedding(model, preprocess, patch: np.ndarray, device: str) -> np.ndarray:
    with torch.no_grad():
        t = preprocess(patch).unsqueeze(0).to(device)
        feat = model(t).squeeze()
        feat = F.normalize(feat.float(), dim=0)
    return feat.cpu().numpy()

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
# Re-identification DB: ResNet embeddings + HSV histograms
# -----------------------------------------------------------------------------

class ReIDDatabase:
    def __init__(self):
        self._db_embed, self._db_hist = {}, {}
        self._next_id = 0
    def _new_id(self):
        oid = self._next_id; self._next_id+=1; return oid
    def _cosine(self,a,b): return float(np.dot(a,b))
    def assign_id(self,embed, hist):
        if not self._db_embed: return self._new_id()
        best_id,best_s=None,-1
        for oid in self._db_embed:
            se=self._cosine(embed,self._db_embed[oid])
            sh=(cv2.compareHist(hist,self._db_hist[oid],cv2.HISTCMP_CORREL)+1)/2
            sc=0.5*se+0.5*sh
            if sc>best_s: best_id,best_s=oid,sc
        return best_id if best_s>=EMBED_SIM_THRESH else self._new_id()
    def update(self,oid,embed, hist):
        if oid in self._db_embed:
            e=0.0*self._db_embed[oid]+1.0*embed
            self._db_embed[oid]=F.normalize(torch.tensor(e),dim=0).cpu().numpy()
            self._db_hist[oid]=0.0*self._db_hist[oid]+1.0*hist
        else:
            self._db_embed[oid]=embed; self._db_hist[oid]=hist

# -----------------------------------------------------------------------------
# Main with VLM object labeling
# -----------------------------------------------------------------------------

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    maskgen = prepare_sam(MODEL_CFG, CHECKPOINT_PATH, device)
    embedder, preprocess = build_embedder(device)
    vlm = VLM(device)

    reid_db = ReIDDatabase()
    prev_segs, id_to_color, id_to_label = {}, {}, {}

    # cap = cv2.VideoCapture(VIDEO_SRC)
    cv2.namedWindow("SAM2 + VLM", cv2.WINDOW_NORMAL)


    from red_loader import R3D_loader
    video_loader = R3D_loader(cfg.video)


    frames = []
    cv2.resizeWindow("SAM2 + VLM", 1280, 720)

    decimate = 4
    couter = 0

    while True:
        frame, depth, pose, timestamp = video_loader.next_frame()
        # rgb to bgr
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if frame is None:
            print("No more frames to process.")
            break

        couter += 1
        if couter % decimate != 0:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_var = float(np.var(cv2.Laplacian(gray, cv2.CV_64F)))
        if frame_var < FRAME_BLUR_THRESH:
            print("Frame is too blurry, skipping...")
            # add text to the frame
            cv2.putText(frame, "Frame is too blurry, skipping...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            frames.append(frame)
            cv2.imshow("SAM2 + VLM", frame)
            cv2.waitKey(1)
            continue
        # use bfloat 16 for SAM2
        masks = maskgen.generate(frame_rgb)
        disp=[]
        for m in masks:
                
            mask_bin=(m['segmentation'].astype(np.uint8))*255
            roi=dilate_mask(mask_bin, EDGE_OFFSET)
            ys,xs=np.where(roi)
            if len(xs)==0: continue
            x0,x1,y0,y1=xs.min(),xs.max(),ys.min(),ys.max()
            patch=frame[y0:y1+1,x0:x1+1]
            emb=extract_embedding(embedder, preprocess, patch, device)
            hist=compute_hsv_hist(frame,roi)
            oid=reid_db.assign_id(emb,hist)
            reid_db.update(oid,emb,hist)
            if oid not in id_to_color:
                id_to_color[oid]=tuple(map(int,np.random.randint(0,255,3)))
            seg = prev_segs.get(oid, m['segmentation'])
            seg=m['segmentation']; prev_segs[oid]=seg
            center = mask_centroid(seg)
            if center and oid not in id_to_label:
                # compute a tight circle radius from this mask
                radius = mask_radius(seg)
                if radius < MAX_RADIUS and radius > MIN_RADIUS:
                    id_to_label[oid] = vlm.ask(frame, center, radius)
                else:
                    id_to_label[oid] = "unknown"
                    # remove the object from the database
                    reid_db.update(oid, np.zeros_like(emb), np.zeros_like(hist))

            disp.append({'segmentation':seg,'color':id_to_color[oid],'center':center,'id':oid})
        vis=blend_masks(frame,disp,MASK_ALPHA)
        # overlay ID and VLM label
        for obj in disp:
            c=obj['center']; oid=obj['id']
            if c:
                if id_to_label[oid]!="unknown":
                    cv2.circle(vis,c,10,obj['color'],-1)
                    txt=f"ID {oid}: {id_to_label.get(oid,'...')}"
                    cv2.putText(vis,txt,(c[0]+5,c[1]-5),cv2.FONT_HERSHEY_SIMPLEX,1.0,obj['color'],1)
        cv2.imshow("SAM2 + VLM",vis)
        frames.append(vis)
        if cv2.waitKey(1)&0xFF==ord('q'): break
    cv2.destroyAllWindows()
    # Save the frames as a video
    if frames:
        print("Saving frame_count:", len(frames))
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (w, h))
        for frm in frames:
            out.write(frm)
        out.release()
        print("Saved video to output.mp4")

if __name__=="__main__":
    GlobalHydra.instance().clear(); main()
