#!/usr/bin/env python3
"""
Download nvidia/RADIO-L model for offline use.
This will cache the model in ~/.cache/huggingface/
"""

from transformers import AutoImageProcessor, AutoModel
import torch
import numpy as np

print("Downloading nvidia/RADIO-L model...")

try:
    # Download and cache the processor
    print("Downloading processor...")
    processor = AutoImageProcessor.from_pretrained("nvidia/RADIO-L", trust_remote_code=True)
    print("✓ Processor downloaded and cached successfully!")

    # Download and cache the model
    print("Downloading model...")
    model = AutoModel.from_pretrained("nvidia/RADIO-L", trust_remote_code=True)
    print("✓ Model downloaded and cached successfully!")

    print(f"✓ Models cached in: ~/.cache/huggingface/")
    print("✓ You can now rsync this cache to your cluster for offline use")

    # Test the model works
    print("Testing model...")
    # Create a dummy RGB image (224x224x3)
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    # Get supported resolution and resize
    next_support_size = model.get_nearest_supported_resolution(224, 224)
    print(f"✓ Model supports resolution: {next_support_size}")

    # Test processing
    inputs = processor(images=dummy_image, return_tensors="pt")

    with torch.no_grad():
        outputs, raw = model(inputs.pixel_values)

    print(f"✓ Model test successful - output shape: {outputs.shape}")

except Exception as e:
    print(f"✗ Error downloading model: {e}")