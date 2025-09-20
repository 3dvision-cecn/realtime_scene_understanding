#!/usr/bin/env python3
"""
Download SmolVLM-Instruct model for offline use.
This will cache the model in ~/.cache/huggingface/
"""

from transformers import AutoProcessor, AutoModelForVision2Seq
import torch

print("Downloading SmolVLM-Instruct model...")

try:
    # Download and cache the processor
    print("Downloading processor...")
    processor = AutoProcessor.from_pretrained('HuggingFaceTB/SmolVLM-Instruct')
    print("✓ Processor downloaded and cached successfully!")

    # Download and cache the model
    print("Downloading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        'HuggingFaceTB/SmolVLM-Instruct',
        torch_dtype=torch.bfloat16
    )
    print("✓ Model downloaded and cached successfully!")

    print(f"✓ Models cached in: ~/.cache/huggingface/")
    print("✓ You can now rsync this cache to your cluster for offline use")

    # Test the model works
    print("Testing model...")
    # Create a dummy RGB image (224x224x3)
    import numpy as np
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What do you see?"},
        ]
    }]

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[dummy_image], return_tensors="pt")

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=10)

    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    print(f"✓ Model test successful - generated text length: {len(generated_texts[0])}")

except Exception as e:
    print(f"✗ Error downloading model: {e}")