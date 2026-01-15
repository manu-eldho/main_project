import torch
import pandas as pd
from torch_geometric.data import Data

from fi_triplet_gat import TripletGAT, build_triplet_graph, initialize_triplet_embeddings
from fi_context_pooling import ContextAttentionPooling
from fi_treatment_scoring import TreatmentScorer
from fi_role_aware_context import extract_role_aware_context
from fi_masked_triplets import mask_context_triplets


def predict_treatments(
    disease,
    df,
    gat,
    pool,
    scorer,
    treatment2id,
    top_k=5
):
    """
    Predict top-K treatments for a given disease.
    """

    gat.eval()
    pool.eval()
    scorer.eval()

    # -------- Build disease embedding --------
    role_context = extract_role_aware_context(df, disease)
    masked = mask_context_triplets(role_context)

    if len(masked) == 0:
        raise ValueError(f"No context found for disease: {disease}")

    edge_index = build_triplet_graph(masked)
    x = initialize_triplet_embeddings(len(masked))

    data = Data(x=x, edge_index=edge_index)

    with torch.no_grad():
        Z = gat(data.x, data.edge_index)
        disease_emb = pool(Z)

        # -------- Score all treatments --------
        all_ids = torch.tensor(list(treatment2id.values()))
        scores = scorer(disease_emb, all_ids)

        # Rank treatments
        scores, indices = torch.sort(scores, descending=True)

    id2treatment = {v: k for k, v in treatment2id.items()}

    predictions = [
        (id2treatment[all_ids[i].item()], scores[i].item())
        for i in range(top_k)
    ]

    return predictions

if __name__ == "__main__":
    # Load KG
    df = pd.read_csv("cleaned_kg.csv", header=None)
    df.columns = ["head", "relation", "tail"]

    # Treatment vocabulary
    treatments = sorted(df[df["relation"] == "treated_by"]["tail"].unique())
    treatment2id = {t: i for i, t in enumerate(treatments)}

    # Load models (same ones you trained)
    gat = TripletGAT()
    pool = ContextAttentionPooling(64)
    scorer = TreatmentScorer(64)

    # Example disease
    disease = "Pepper__bell___Bacterial_spot"

    preds = predict_treatments(
        disease,
        df,
        gat,
        pool,
        scorer,
        treatment2id,
        top_k=5
    )

    print(f"\nPredicted treatments for: {disease}\n")
    for rank, (treatment, score) in enumerate(preds, 1):
        print(f"{rank}. {treatment}  (score={score:.4f})")
