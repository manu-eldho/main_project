import torch
import torch.nn as nn


class TripletEncoder(nn.Module):
    def __init__(self, entity2id, relation2id, emb_dim=128):
        super().__init__()

        self.entity2id = entity2id
        self.relation2id = relation2id

        self.entity_emb = nn.Embedding(len(entity2id), emb_dim)
        self.relation_emb = nn.Embedding(len(relation2id), emb_dim)

        self.proj = nn.Sequential(
            nn.Linear(emb_dim * 3, emb_dim),
            nn.ReLU()
        )


    def _safe_lookup(self, vocab, key):
        """
        Returns ID for key if present, otherwise returns ID for [MASK]
        """
        return vocab[key] if key in vocab else vocab["[MASK]"]

    def forward(self, triplets):
        h_ids = torch.tensor(
            [self._safe_lookup(self.entity2id, h) for h, _, _ in triplets],
            dtype=torch.long
        )

        r_ids = torch.tensor(
            [self._safe_lookup(self.relation2id, r) for _, r, _ in triplets],
            dtype=torch.long
        )

        t_ids = torch.tensor(
            [self._safe_lookup(self.entity2id, t) for _, _, t in triplets],
            dtype=torch.long
        )

        h = self.entity_emb(h_ids)
        r = self.relation_emb(r_ids)
        t = self.entity_emb(t_ids)

        x = torch.cat([h, r, t], dim=1)
        return self.proj(x)
