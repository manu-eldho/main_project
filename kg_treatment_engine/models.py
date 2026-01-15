import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class TripletGAT(nn.Module):
    def __init__(self, in_dim=128, out_dim=64):
        super().__init__()
        self.gat1 = GATConv(in_dim, 64, heads=4, concat=True)
        self.gat2 = GATConv(64 * 4, out_dim, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        return self.gat2(x, edge_index)


class ContextAttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, Z):
        w = F.softmax(self.attn(Z), dim=0)
        return (w * Z).sum(dim=0)


class TreatmentScorer(nn.Module):
    def __init__(self, num_treatments, dim):
        super().__init__()
        self.emb = nn.Embedding(num_treatments, dim)

    def forward(self, disease_emb):
        return torch.matmul(self.emb.weight, disease_emb)
