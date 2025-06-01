import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (GATConv, TransformerConv, HeteroConv, global_mean_pool)

    
# TGNN Model
class GraphClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, 
                 edge_feat_dim, out_channels):
        super().__init__()

        self.conv = HeteroConv({
            ('object', 'relation', 'object'): GATConv(in_channels, hidden_channels, edge_dim=edge_feat_dim),
            ('object', 'temporal', 'object'): TransformerConv(in_channels, hidden_channels),
        }, aggr='sum')

        self.fc1 = nn.Linear(hidden_channels, 128)
        self.fc2 = nn.Linear(128, out_channels)

        # Aggregate node features
        self.pool = global_mean_pool # Try GAT Pooling

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, batch):
        x_dict = self.conv(x_dict, edge_index_dict, edge_attr_dict)
        
        # Handle single graph case in inference or batched graphs in training
        if 'batch' in batch['object']:
            graph_embedding = self.pool(x_dict['object'], batch['object'].batch)
        else:
            # Take mean for single graph
            graph_embedding = x_dict['object'].mean(dim=0, keepdim=True)  
            
        graph_embedding = F.relu(self.fc1(graph_embedding))
        return self.fc2(graph_embedding)