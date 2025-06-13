import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (GATConv, TransformerConv, HeteroConv, global_mean_pool, global_max_pool)

    
# TGNN Model
class GraphClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, 
                 edge_feat_dim, out_channels, feat_dim=128):
        super().__init__()


        self.fc1 = nn.Linear(hidden_channels, 128)
        self.fc2 = nn.Linear(128, out_channels)

        # create MLP for edge features
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_feat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, feat_dim)
        )

        # create a mlp for node features
        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, feat_dim)
        )


        self.conv = HeteroConv({
            ('object', 'relation', 'object'): GATConv(feat_dim, hidden_channels, edge_dim=feat_dim),
            ('object', 'temporal', 'object'): TransformerConv(feat_dim, hidden_channels),
        }, aggr='sum')


        # Aggregate node features
        self.pool = global_mean_pool # Try GAT Pooling

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, batch):
        # apply MLP to node features
        for key in x_dict.keys():
            if 'object' in key:
                x_dict[key] = self.node_mlp(x_dict[key])
            
        # apply MLP to edge features
        for key in edge_attr_dict.keys():
            if 'relation' in key:
                edge_attr_dict[key] = self.edge_mlp(edge_attr_dict[key])


        x_dict = self.conv(x_dict, edge_index_dict, edge_attr_dict)
        
        # Handle single graph case in inference or batched graphs in training
        if 'batch' in batch['object']:
            graph_embedding = self.pool(x_dict['object'], batch['object'].batch)
        else:
            # Take mean for single graph
            graph_embedding = x_dict['object'].mean(dim=0, keepdim=True)  
            
        graph_embedding = F.relu(self.fc1(graph_embedding))
        return self.fc2(graph_embedding)