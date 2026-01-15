# ============================================================
# Treatment Scoring Head
# ============================================================

import torch
import torch.nn as nn


class TreatmentScoringHead(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.scorer = nn.Linear(embedding_dim, 1)

    def forward(self, node_embeddings):
        """
        node_embeddings: [num_nodes, embedding_dim]
        returns scores: [num_nodes]
        """
        scores = self.scorer(node_embeddings).squeeze(-1)
        return scores
