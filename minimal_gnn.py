# ============================================================
# Minimal GNN that consumes KG subgraph tensors
# ============================================================

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv


class MinimalGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()

        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)

        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        """
        x: [num_nodes, in_dim]
        edge_index: [2, num_edges]
        """
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        return x
