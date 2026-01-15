import torch
import random
import pandas as pd
from torch_geometric.data import Data

from fi_triplet_gat import TripletGAT, build_triplet_graph, initialize_triplet_embeddings
from fi_context_pooling import ContextAttentionPooling
from fi_treatment_scoring import TreatmentScorer
from fi_training_step import ranking_loss
from fi_role_aware_context import extract_role_aware_context
from fi_masked_triplets import mask_context_triplets

# Load KG
df = pd.read_csv("cleaned_kg.csv", header=None)
df.columns = ["head", "relation", "tail"]

# Build treatment vocab
treatments = sorted(df[df["relation"] == "treated_by"]["tail"].unique())
treatment2id = {t: i for i, t in enumerate(treatments)}

# Models
gat = TripletGAT()
pool = ContextAttentionPooling(64)
scorer = TreatmentScorer(64)

optimizer = torch.optim.Adam(
    list(gat.parameters()) +
    list(pool.parameters()) +
    list(scorer.parameters()),
    lr=1e-3
)

# Training for a few steps
for step in range(10):
    optimizer.zero_grad()

    row = df[df["relation"] == "treated_by"].sample(1).iloc[0]
    disease = row["head"]
    pos_treatment = row["tail"]

    neg_treatment = random.choice(
        [t for t in treatments if t != pos_treatment]
    )

    # Forward
    role_context = extract_role_aware_context(df, disease)
    masked = mask_context_triplets(role_context)

    edge_index = build_triplet_graph(masked)
    x = initialize_triplet_embeddings(len(masked))

    data = Data(x=x, edge_index=edge_index)

    Z = gat(data.x, data.edge_index)
    disease_emb = pool(Z)

    pos_id = torch.tensor([treatment2id[pos_treatment]])
    neg_id = torch.tensor([treatment2id[neg_treatment]])

    pos_score = scorer(disease_emb, pos_id)
    neg_score = scorer(disease_emb, neg_id)

    loss = ranking_loss(pos_score, neg_score)
    loss.backward()
    optimizer.step()

    print(f"Step {step} | Loss: {loss.item():.4f}")
