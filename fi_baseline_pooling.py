import torch

def mean_pooling(Z):
    """
    Z: [num_triplets, emb_dim]
    """
    return Z.mean(dim=0)
