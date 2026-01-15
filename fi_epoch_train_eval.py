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
from fi_evaluation import evaluate_model


# ======================
# Load KG
# ======================
df = pd.read_csv("cleaned_kg.csv", header=None)
df.columns = ["head", "relation", "tail"]

# Treatment vocabulary
treatments = sorted(df[df["relation"] == "treated_by"]["tail"].unique())
treatment2id = {t: i for i, t in enumerate(treatments)}

treated_df = df[df["relation"] == "treated_by"]


# ======================
# Models
# ======================
gat = TripletGAT()
pool = ContextAttentionPooling(64)
scorer = TreatmentScorer(64)

optimizer = torch.optim.Adam(
    list(gat.parameters()) +
    list(pool.parameters()) +
    list(scorer.parameters()),
    lr=1e-3
)


# ======================
# Training + Evaluation
# ======================
EPOCHS = 10
STEPS_PER_EPOCH = 50

print("\nEpoch | Hits@1 | Hits@3 | Hits@5 | MRR")
print("-" * 45)

for epoch in range(1, EPOCHS + 1):

    # ---- Training ----
    gat.train()
    pool.train()
    scorer.train()

    for _ in range(STEPS_PER_EPOCH):
        optimizer.zero_grad()

        row = treated_df.sample(1).iloc[0]
        disease = row["head"]
        pos_treatment = row["tail"]

        neg_treatment = random.choice(
            [t for t in treatments if t != pos_treatment]
        )

        # Build disease embedding
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

    # ---- Evaluation ----
    gat.eval()
    pool.eval()
    scorer.eval()

    hits, mrr = evaluate_model(
        df,
        gat,
        pool,
        scorer,
        treatment2id,
        Ks=(1, 3, 5)
    )

    print(
        f"{epoch:5d} | "
        f"{hits[1]:.4f} | "
        f"{hits[3]:.4f} | "
        f"{hits[5]:.4f} | "
        f"{mrr:.4f}"
    )
