import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextAttentionPooling(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.attn = nn.Linear(emb_dim, 1)

    def forward(self, Z):
        """
        Z: [num_triplets, emb_dim]
        returns: [emb_dim] disease embedding
        """
        weights = F.softmax(self.attn(Z), dim=0)   # [num_triplets, 1]
        pooled = (weights * Z).sum(dim=0)
        return pooled

if __name__ == "__main__":
    from fi_triplet_gat import TripletGAT, build_triplet_graph, initialize_triplet_embeddings
    from fi_role_aware_context import extract_role_aware_context
    from fi_masked_triplets import mask_context_triplets
    import pandas as pd
    from torch_geometric.data import Data

    # Load KG
    df = pd.read_csv("cleaned_kg.csv", header=None)
    df.columns = ["head", "relation", "tail"]

    anchor = "Pepper__bell___Bacterial_spot"

    role_context = extract_role_aware_context(df, anchor)
    masked_triplets = mask_context_triplets(role_context)

    edge_index = build_triplet_graph(masked_triplets)
    x = initialize_triplet_embeddings(len(masked_triplets))

    data = Data(x=x, edge_index=edge_index)

    gat = TripletGAT()
    Z = gat(data.x, data.edge_index)

    pool = ContextAttentionPooling(Z.size(1))
    h_disease = pool(Z)

    print("Disease embedding shape:", h_disease.shape)
