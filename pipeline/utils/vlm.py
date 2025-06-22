from transformers import AutoProcessor, AutoModelForVision2Seq
import cv2
import torch
import numpy as np


class VLM:
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.processor = AutoProcessor.from_pretrained('HuggingFaceTB/SmolVLM-Instruct')
        self.model = AutoModelForVision2Seq.from_pretrained(
            'HuggingFaceTB/SmolVLM-Instruct', torch_dtype=torch.bfloat16,
        ).to(self.device)

    def ask(self, frame: np.ndarray, center: tuple[int,int], radius: int = 64) -> str:
        x, y = center
        # draw a red circle on the image
        # draw a red circle on a copy of the frame
        circ = frame.copy()
        # conver rgb to bgr
        circ = cv2.cvtColor(circ, cv2.COLOR_RGB2BGR)
        cv2.circle(circ, (x, y), radius, (0, 0, 255), 2)
        # crop the image to the circle
        offset = int(radius + 10)
        x0, y0 = max(x - offset, 0), max(y - offset, 0)
        x1, y1 = min(x + offset, frame.shape[1]), min(y + offset, frame.shape[0])
        circ = circ[y0:y1, x0:x1]
        # uf cropping the image, the circle is not centered return unknown
        if circ.shape[0] != circ.shape[1]:
            # cv2.putText(circ, "unknown", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # cv2.imshow("Circle", circ)
            # cv2.waitKey(100)
            return "unknown"
        # show the image with the circle
        messages = [
            {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is the object in the middle of the red circle, in maximum two words !"},
            ]
            },
        ]
        # convert the image to RGB
        circ = cv2.cvtColor(circ, cv2.COLOR_BGR2RGB)
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[circ], return_tensors="pt").to(self.device)
        # bfloat16 is needed for the model to run on GPU
        with torch.autocast("cuda", dtype=torch.bfloat16):
            generated_ids = self.model.generate(**inputs, max_new_tokens=20)
        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        asssitant = generated_texts[0].split("Assistant:")[-1]
        # for debugging, show the circle with the text
        # cv2.putText(circ, asssitant.strip(), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # cv2.imshow("Circle", circ)
        # cv2.waitKey(0)
        return asssitant.strip()