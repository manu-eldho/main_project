import torch
import torch.nn as nn

class TreatmentScorer(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.treatment_embeddings = nn.Embedding(
            num_embeddings=1000,  # placeholder, will fix later
            embedding_dim=emb_dim
        )

    def forward(self, disease_emb, treatment_ids):
        """
        disease_emb: [emb_dim]
        treatment_ids: [N]
        """
        t_emb = self.treatment_embeddings(treatment_ids)   # [N, emb_dim]
        scores = torch.matmul(t_emb, disease_emb)           # [N]
        return scores

if __name__ == "__main__":
    emb_dim = 64
    disease_emb = torch.randn(emb_dim)

    scorer = TreatmentScorer(emb_dim)

    # Fake treatment IDs
    treatment_ids = torch.tensor([0, 1, 2, 3, 4])

    scores = scorer(disease_emb, treatment_ids)
    print("Treatment scores:", scores)
