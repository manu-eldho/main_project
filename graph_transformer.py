import torch
import torch.nn as nn


class GraphLevelTransformer(nn.Module):
    """
    Graph-level Transformer encoder.
    Operates on node embeddings produced by a GNN.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_cls_token: bool = True
    ):
        super().__init__()

        self.use_cls_token = use_cls_token

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, node_embeddings):
        """
        node_embeddings: (N, D)
        returns:
            refined_node_embeddings: (N, D)
            graph_embedding: (D)
        """

        x = node_embeddings.unsqueeze(0)  # (1, N, D)

        if self.use_cls_token:
            cls = self.cls_token.expand(1, -1, -1)  # (1, 1, D)
            x = torch.cat([cls, x], dim=1)          # (1, N+1, D)

        x = self.encoder(x)
        x = self.norm(x)

        if self.use_cls_token:
            graph_embedding = x[:, 0]        # (1, D)
            refined_nodes = x[:, 1:]         # (1, N, D)
        else:
            graph_embedding = x.mean(dim=1)  # (1, D)
            refined_nodes = x

        return refined_nodes.squeeze(0), graph_embedding.squeeze(0)
