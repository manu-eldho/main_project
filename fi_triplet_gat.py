import torch
from torch_geometric.data import Data

def build_triplet_graph(masked_triplets):
    """
    Each masked triplet is a node.
    Fully connect all triplet nodes (same anchor entity).
    """

    num_nodes = len(masked_triplets)

    # Fully connected (excluding self-loops)
    edge_index = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edge_index.append([i, j])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return edge_index

def initialize_triplet_embeddings(num_nodes, emb_dim=128):
    """
    One learnable embedding per triplet node.
    """
    return torch.randn((num_nodes, emb_dim))

from torch_geometric.nn import GATConv
import torch.nn.functional as F

class TripletGAT(torch.nn.Module):
    def __init__(self, in_dim=128, hidden_dim=64, out_dim=64, heads=4):
        super().__init__()

        self.gat1 = GATConv(
            in_dim,
            hidden_dim,
            heads=heads,
            concat=True
        )

        self.gat2 = GATConv(
            hidden_dim * heads,
            out_dim,
            heads=1,
            concat=False
        )

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        return x

if __name__ == "__main__":
    import pandas as pd
    from fi_role_aware_context import extract_role_aware_context
    from fi_masked_triplets import mask_context_triplets

    # Load KG
    df = pd.read_csv("cleaned_kg.csv", header=None)
    df.columns = ["head", "relation", "tail"]

    anchor = "Pepper__bell___Bacterial_spot"

    # Role-aware context
    role_context = extract_role_aware_context(df, anchor)

    # Masked triplets
    masked_triplets = mask_context_triplets(role_context)

    # Build graph
    edge_index = build_triplet_graph(masked_triplets)

    # Initialize embeddings
    x = initialize_triplet_embeddings(len(masked_triplets))

    # Create PyG data
    data = Data(x=x, edge_index=edge_index)

    # Run GAT
    model = TripletGAT()
    out = model(data.x, data.edge_index)

    print("Input shape :", data.x.shape)
    print("Output shape:", out.shape)
