import torch
import torch.nn.functional as F

def ranking_loss(pos_score, neg_score, margin=1.0):
    """
    pos_score: scalar
    neg_score: scalar
    """
    return F.relu(margin - pos_score + neg_score)

if __name__ == "__main__":
    pos = torch.tensor(0.5, requires_grad=True)
    neg = torch.tensor(0.8, requires_grad=True)

    loss = ranking_loss(pos, neg)
    loss.backward()

    print("Loss:", loss.item())
