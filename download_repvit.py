#!/usr/bin/env python3
"""
Download RepViT backbone model for EdgeTAM offline use.
This will cache the model in ~/.cache/huggingface/
"""

import timm
import torch

print("Downloading RepViT model for EdgeTAM...")

try:
    # This downloads and caches the model
    model = timm.create_model('repvit_m1.dist_in1k', pretrained=True)
    print("✓ RepViT model downloaded and cached successfully!")
    print(f"✓ Model cached in: ~/.cache/huggingface/")
    print("✓ You can now rsync this cache to your cluster for offline use")

    # Test the model works
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"✓ Model test successful - output shape: {output.shape}")

except Exception as e:
    print(f"✗ Error downloading model: {e}")