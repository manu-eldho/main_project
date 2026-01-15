import os
import torch
import pandas as pd
from torch_geometric.data import Data

from kg_utils import extract_role_aware_context, mask_context
from models import TripletGAT, ContextAttentionPooling, TreatmentScorer


class TreatmentPredictionEngine:

    def __init__(self, kg_path):
        # ======================
        # Load KG
        # ======================
        self.df = pd.read_csv(kg_path, header=None)
        self.df.columns = ["head", "relation", "tail"]

        # ======================
        # Treatment vocabulary
        # ======================
        self.treatments = sorted(
            self.df[self.df["relation"] == "treated_by"]["tail"].unique()
        )
        self.treatment2id = {t: i for i, t in enumerate(self.treatments)}

        # ======================
        # Models
        # ======================
        self.gat = TripletGAT()
        self.pool = ContextAttentionPooling(64)
        self.scorer = TreatmentScorer(len(self.treatments), 64)

        # ======================
        # Robust model loading
        # ======================
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "trained_model.pt")

        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location="cpu")

            # ---- Load GAT & Pool (exact match) ----
            self.gat.load_state_dict(ckpt["gat"])
            self.pool.load_state_dict(ckpt["pool"])

            # ---- Load scorer safely (vocab size mismatch fix) ----
            scorer_state = ckpt["scorer"]

            if "treatment_embeddings.weight" in scorer_state:
                pretrained_weight = scorer_state["treatment_embeddings.weight"]
                current_weight = self.scorer.treatment_embeddings.weight

                min_size = min(
                    pretrained_weight.size(0),
                    current_weight.size(0)
                )

                current_weight.data[:min_size] = pretrained_weight[:min_size]

                print(f"✔ Loaded trained KG model (treatments copied: {min_size})")
            else:
                raise RuntimeError(
                    "Checkpoint scorer does not contain treatment_embeddings.weight"
                )
        else:
            print("⚠ Using untrained KG model")

        self.gat.eval()
        self.pool.eval()
        self.scorer.eval()

    # ======================
    # Build fully-connected context graph
    # ======================
    def _build_graph(self, n):
        if n <= 1:
            return torch.empty((2, 0), dtype=torch.long)
        edges = [[i, j] for i in range(n) for j in range(n) if i != j]
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    # ======================
    # Prediction
    # ======================
    def predict(self, disease, top_k=5):

        if disease not in set(self.df["head"]).union(set(self.df["tail"])):
            raise ValueError(f"Disease not found in KG: {disease}")

        # -------- Context extraction --------
        context = extract_role_aware_context(self.df, disease)
        masked = mask_context(context)

        if len(masked) == 0:
            raise ValueError(f"No context found for disease: {disease}")

        # -------- Triplet embeddings --------
        x = torch.randn(len(masked), 128)
        edge_index = self._build_graph(len(masked))

        data = Data(x=x, edge_index=edge_index)

        # -------- Inference --------
        with torch.no_grad():
            Z = self.gat(data.x, data.edge_index)
            disease_emb = self.pool(Z)
            scores = self.scorer(disease_emb)

        ranked_idx = torch.argsort(scores, descending=True)

        return [
            (self.treatments[i], scores[i].item())
            for i in ranked_idx[:top_k]
        ]
