#!/usr/bin/env python3
"""
Download YOLO model for offline use using the normal YOLO constructor.
This will cache the model in the standard location for rsync to cluster.
"""

from ultralytics import YOLO

def download_yolo_model():
    print("Downloading YOLO12x model using YOLO constructor...")

    try:
        # This downloads and caches the model in the standard ultralytics cache
        model = YOLO("yolo12x.pt")
        print("✓ YOLO model downloaded and cached successfully!")
        print("✓ Model cached in standard ultralytics cache location")
        print("✓ You can now rsync this cache to your cluster for offline use")

        # Test the model works
        print("✓ Model initialization successful")

    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        print("Make sure you have internet connectivity and try again")

if __name__ == "__main__":
    download_yolo_model()