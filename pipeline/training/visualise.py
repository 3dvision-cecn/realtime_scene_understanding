import json
import os
import torch
import random
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import (GATConv, TransformerConv, HeteroConv, global_mean_pool)
from torch_geometric.loader import DataLoader
from omegaconf import OmegaConf
import wandb
import time
import numpy as np
import argparse
from sklearn.metrics import confusion_matrix, top_k_accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

from tgnn_model import GraphClassifier
from data_loader import GraphDataset
from vn_mappings import generate_label_map, load_vn_mappings
from tqdm import tqdm
import umap
from datetime import datetime

def create_graph_embeddings(dataset, model, device):
    model.eval()
    embeddings = []
    labels = {}
    labels['is_iphone'] = []
    labels['kitche_num'] = []
    labels['action'] = []
    labels['pred_action'] = []
    
    with torch.no_grad():
        for data in tqdm(dataset):
            data = data.to(device)
            out = model.get_graph_embedding(data.x_dict, 
                        data.edge_index_dict,
                        {'relation': data['object', 'relation', 'object'].edge_attr},
                        data
                        )
            # flatten the batch dimension
            out = out.cpu().numpy()
            embeddings.append(out)

            # is iphone
            is_iphone = data['is_iphone'].cpu().numpy()
            labels['is_iphone'].append(is_iphone)
            # kitchen number
            kitchen_num = data['kitchen_num'].cpu().numpy()
            labels['kitche_num'].append(kitchen_num)
            # action
            action = data.y.argmax(dim=1).cpu().numpy()
            labels['action'].append(action)

            pred = model(data.x_dict, 
            data.edge_index_dict,
            {'relation': data['object', 'relation', 'object'].edge_attr},
            data
            )
            pred = pred.argmax(dim=1).cpu().numpy()
            labels['pred_action'].append(pred)
    
    # concatante label lists
    labels['is_iphone'] = np.concatenate(labels['is_iphone'], axis=0)
    labels['kitche_num'] = np.concatenate(labels['kitche_num'], axis=0)
    labels['action'] = np.concatenate(labels['action'], axis=0)
    labels['pred_action'] = np.concatenate(labels['pred_action'], axis=0)

    return np.concatenate(embeddings, axis=0), labels


argparser = argparse.ArgumentParser(description="Visualise Graph Samples")
argparser.add_argument("--dataset_path", type=str, default="dataset/graph_samples", help="Path to the dataset directory")
argparser.add_argument("--network_path", type=str, default="network.pth", help="Path to the trained model")
argparser.add_argument("--batch_size", type=int, default=32, help="Batch size for data loading")


args = argparser.parse_args()

# load the dataset
train_dataset = GraphDataset(
    data_dir=args.dataset_path,
    node_drop_p=0.0,
    is_train=True
)
val_dataset = GraphDataset(
    data_dir=args.dataset_path,
    node_drop_p=0.0,
    is_train=False
)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)


train_csv = "conf/ek100/EPIC_100_train.csv"
val_csv = "conf/ek100/EPIC_100_train.csv"

label, mapping_vn2act = generate_label_map('ek100_cls')
mapping_act2v = {i: int(vn.split(':')[0]) for (vn, i) in mapping_vn2act.items()}
mapping_act2n = {i: int(vn.split(':')[1]) for (vn, i) in mapping_vn2act.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = GraphClassifier(3072, 128, 4, len(mapping_vn2act)).to(device)
model.load_state_dict(torch.load(args.network_path, map_location=device, weights_only=True))


embeddings_train, labels_train = create_graph_embeddings(train_loader, model, device)
embeddings_val, labels_val = create_graph_embeddings(val_loader, model, device)
print("Embeddings train shape:", embeddings_train.shape)
print("Embeddings val shape:", embeddings_val.shape)

# print is_iphone labels
# number of is_iphone == Treue in train and val
print("Number of is_iphone == True in train:", np.sum(labels_train['is_iphone']))
print("Number of is_iphone == True in val:", np.sum(labels_val['is_iphone']))

combined_embeddings = np.concatenate((embeddings_train, embeddings_val), axis=0)

# UMAP for dimensionality reduction
reducer = umap.UMAP()
reducer.fit_transform(combined_embeddings)
train_embeddings_umap = reducer.transform(embeddings_train)
val_embeddings_umap = reducer.transform(embeddings_val)

# label all iphone blue and all non-iphone red
# visualize embeddings with blue for iPhone and red for non-iPhone
plt.figure(figsize=(10, 5))

# plot train set
# plt.subplot(1, 2, 1)
colors_train = ['blue' if flag else 'red' for flag in labels_train['is_iphone']]
plt.scatter(train_embeddings_umap[:, 0], train_embeddings_umap[:, 1],
            c=colors_train, alpha=0.6, s=10)
plt.title('UMAP (iPhone vs HD-Epic)')
plt.xlabel('UMAP axis 1')
plt.ylabel('UMAP axis 2')
plt.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', label='iPhone',
               markerfacecolor='blue', markersize=6),
    plt.Line2D([0], [0], marker='o', color='w', label='HD-Epic',
               markerfacecolor='red', markersize=6)
])

# # plot validation set
# plt.subplot(1, 2, 2)
# colors_val = ['blue' if flag else 'red' for flag in labels_val['is_iphone']]
# plt.scatter(val_embeddings_umap[:, 0], val_embeddings_umap[:, 1],
#             c=colors_val, alpha=0.6, s=10)
# plt.title('Val UMAP (iPhone vs Non-iPhone)')
# plt.xlabel('UMAP 1')
# plt.ylabel('UMAP 2')
# plt.legend(handles=[
#     plt.Line2D([0], [0], marker='o', color='w', label='iPhone',
#                markerfacecolor='blue', markersize=6),
#     plt.Line2D([0], [0], marker='o', color='w', label='Non-iPhone',
#                markerfacecolor='red', markersize=6)
# ])


plt.figure(figsize=(10, 5))

# plot the error in the train set
plt.subplot(1, 2, 1)
# highlight correct vs incorrect predictions on train set
error_flags = labels_train['action'] != labels_train['pred_action']
colors_error = ['blue' if not e else 'red' for e in error_flags]

plt.scatter(
    train_embeddings_umap[:, 0],
    train_embeddings_umap[:, 1],
    c=colors_error,
    alpha=0.6,
    s=10
)
plt.title('Train UMAP (Prediction Errors)')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', label='Correct',
               markerfacecolor='blue', markersize=6),
    plt.Line2D([0], [0], marker='o', color='w', label='Incorrect',
               markerfacecolor='red', markersize=6)
])

# plot the error in the validation set
plt.subplot(1, 2, 2)
error_flags_val = labels_val['action'] != labels_val['pred_action']
colors_error_val = ['blue' if not e else 'red' for e in error_flags_val]

plt.scatter(
    val_embeddings_umap[:, 0],
    val_embeddings_umap[:, 1],
    c=colors_error_val,
    alpha=0.6,
    s=10
)
plt.title('Val UMAP (Prediction Errors)')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', label='Correct',
               markerfacecolor='blue', markersize=6),
    plt.Line2D([0], [0], marker='o', color='w', label='Incorrect',
               markerfacecolor='red', markersize=6)
])



# # plot kitchen numbers
# plt.subplot(1, 2, 2)
# # get unique kitchen IDs and a discrete colormap
# unique_kitchens = np.unique(labels_train['kitche_num'])
# cmap = plt.get_cmap('tab10', len(unique_kitchens))
# # map each sample’s kitchen_num to a color
# kitchen_idx = {k: i for i, k in enumerate(unique_kitchens)}
# colors_kitchen = [cmap(kitchen_idx[k]) for k in labels_train['kitche_num']]

# plt.scatter(
#     train_embeddings_umap[:, 0],
#     train_embeddings_umap[:, 1],
#     c=colors_kitchen,
#     alpha=1.0,
#     s=10
# )
# plt.title('Train UMAP (Kitchen Number)')
# plt.xlabel('UMAP 1')
# plt.ylabel('UMAP 2')

# # build legend entries
# handles = [
#     plt.Line2D([0], [0], marker='o', color='w', label=f'Kitchen {k}',
#                markerfacecolor=cmap(i), markersize=6)
#     for i, k in enumerate(unique_kitchens)
# ]
# plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')

# plot actions
print("Number of unique actions in train set:", len(np.unique(labels_train['action'])))
# plot actions
plt.figure(figsize=(6, 5))
unique_actions = np.unique(labels_train['action'])
cmap_actions = plt.get_cmap('hsv', len(unique_actions))
action_to_idx = {a: i for i, a in enumerate(unique_actions)}
colors_actions = [cmap_actions(action_to_idx[a]) for a in labels_train['action']]

plt.scatter(
    train_embeddings_umap[:, 0],
    train_embeddings_umap[:, 1],
    c=colors_actions,
    alpha=1.0,
    s=10
)
plt.title('Train UMAP (Actions)')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')

# build legend for actions
handles = [
    plt.Line2D([0], [0], marker='o', color='w', label=f'Action {a}',
               markerfacecolor=cmap_actions(idx), markersize=6)
    for idx, a in enumerate(unique_actions)
]
plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig("umap_visualization.png", bbox_inches='tight')
plt.show()





