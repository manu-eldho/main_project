import torch
import pandas as pd
from torch_geometric.data import Data

from fi_triplet_gat import TripletGAT, build_triplet_graph, initialize_triplet_embeddings
from fi_context_pooling import ContextAttentionPooling
from fi_role_aware_context import extract_role_aware_context
from fi_masked_triplets import mask_context_triplets
from fi_treatment_scoring import TreatmentScorer
from fi_baseline_pooling import mean_pooling



def get_disease_embedding(df, disease, gat, pool):
    role_context = extract_role_aware_context(df, disease)
    masked = mask_context_triplets(role_context)

    edge_index = build_triplet_graph(masked)
    x = initialize_triplet_embeddings(len(masked))

    data = Data(x=x, edge_index=edge_index)

    Z = gat(data.x, data.edge_index)
    h_disease = mean_pooling(Z)

    return h_disease

def evaluate_model(
    df,
    gat,
    pool,
    scorer,
    treatment2id,
    Ks=(1, 3, 5)
):
    hits = {k: 0 for k in Ks}
    mrr = 0.0
    count = 0

    treated_df = df[df["relation"] == "treated_by"]

    for _, row in treated_df.iterrows():
        disease = row["head"]
        true_treatment = row["tail"]

        if true_treatment not in treatment2id:
            continue

        # Disease embedding
        h_disease = get_disease_embedding(df, disease, gat, pool)

        # Score all treatments
        all_ids = torch.tensor(list(treatment2id.values()))
        scores = scorer(h_disease, all_ids)

        # Rank treatments
        sorted_scores, indices = torch.sort(scores, descending=True)
        ranked_ids = all_ids[indices]

        true_id = treatment2id[true_treatment]
        rank = (ranked_ids == true_id).nonzero(as_tuple=True)[0].item() + 1

        # MRR
        mrr += 1.0 / rank

        # Hits@K
        for k in Ks:
            if rank <= k:
                hits[k] += 1

        count += 1

    # Normalize
    hits = {k: hits[k] / count for k in hits}
    mrr /= count

    return hits, mrr

if __name__ == "__main__":
    # Load KG
    df = pd.read_csv("cleaned_kg.csv", header=None)
    df.columns = ["head", "relation", "tail"]

    # Treatment vocabulary
    treatments = sorted(df[df["relation"] == "treated_by"]["tail"].unique())
    treatment2id = {t: i for i, t in enumerate(treatments)}

    # Models
    gat = TripletGAT()
    pool = ContextAttentionPooling(64)
    scorer = TreatmentScorer(64)

    hits, mrr = evaluate_model(
        df,
        gat,
        pool,
        scorer,
        treatment2id,
        Ks=(1, 3, 5)
    )

    print("Evaluation Results")
    print("Hits@1:", hits[1])
    print("Hits@3:", hits[3])
    print("Hits@5:", hits[5])
    print("MRR   :", mrr)
