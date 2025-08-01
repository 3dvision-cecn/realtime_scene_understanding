import h5py
import numpy as np

file = "/home/can/Dropbox/Can.sacan.cs team folder/dataset/graph_samples_clip_concat/train/04/sample_000.h5"

with h5py.File(file, 'r') as f:
    feats = f['frames']['0005']['features']
    print("Shape:", feats.shape)
    print("Dtype:", feats.dtype)
    print("Sample:", feats[0] if feats.shape[0] > 0 else "Empty")
    edge_labels = f['frames']['0005']['edge_labels'][:]
    print("Edge labels shape:", edge_labels.shape)
    print("Edge labels dtype:", edge_labels.dtype)
    print("First 5 edge labels:", edge_labels[:5])
    pos = f['frames']['0005']['pos'][:]
    print("Position shape:", pos.shape)  # should be (N, 3)
    print("First node 3D position:", pos[0])
    labels = f['frames']['0005']['labels'][:]
    print("Node labels:", labels)
        # Find indices where label == 2
    indices = np.where(labels == 2)[0]
    print("Indices of label 2:", indices)

    if len(indices) > 0:
        for i in indices:
            print(f"Node {i} has label 2, 3D position: {pos[i]}")
    else:
        print("No node with label 2 in this frame.")