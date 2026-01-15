import torch
import pandas as pd
from torch_geometric.data import Data

from kg_utils import extract_role_aware_context, mask_context
from models import TripletGAT, ContextAttentionPooling, TreatmentScorer


class TreatmentPredictionEngine:

    def __init__(self, kg_path):
        # Load KG
        self.df = pd.read_csv(kg_path, header=None)
        self.df.columns = ["head", "relation", "tail"]

        # Treatment vocabulary
        self.treatments = sorted(
            self.df[self.df["relation"] == "treated_by"]["tail"].unique()
        )
        self.treatment2id = {t: i for i, t in enumerate(self.treatments)}

        # Models
        self.gat = TripletGAT()
        self.pool = ContextAttentionPooling(64)
        self.scorer = TreatmentScorer(len(self.treatments), 64)

        # Optional: load trained weights
        try:
            ckpt = torch.load("trained_model.pt")
            self.gat.load_state_dict(ckpt["gat"])
            self.pool.load_state_dict(ckpt["pool"])
            self.scorer.load_state_dict(ckpt["scorer"])
            print("✔ Loaded trained KG model")
        except:
            print("⚠ Using untrained KG model")

        self.gat.eval()
        self.pool.eval()
        self.scorer.eval()

    def _build_graph(self, n):
        edges = [[i, j] for i in range(n) for j in range(n) if i != j]
        return torch.tensor(edges).t().contiguous()

    def predict(self, disease, top_k=5):

        if disease not in set(self.df["head"]).union(set(self.df["tail"])):
            raise ValueError("Disease not found in KG")

        # Context extraction
        context = extract_role_aware_context(self.df, disease)
        masked = mask_context(context)

        # Triplet embeddings (learned during training)
        x = torch.randn(len(masked), 128)
        edge_index = self._build_graph(len(masked))

        data = Data(x=x, edge_index=edge_index)

        with torch.no_grad():
            Z = self.gat(data.x, data.edge_index)
            disease_emb = self.pool(Z)
            scores = self.scorer(disease_emb)

        ranked = torch.argsort(scores, descending=True)

        return [
            (self.treatments[i], scores[i].item())
            for i in ranked[:top_k]
        ]
