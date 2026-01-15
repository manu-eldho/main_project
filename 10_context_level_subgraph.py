# ============================================================
# Disease-Centric KG → GNN → Graph Transformer → TuckER Matching
# FINAL LEAK-FREE, RELATION-AWARE VERSION
# ============================================================

import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

from subgraph_to_gnn_data import subgraph_to_tensors
from minimal_gnn import MinimalGNN
from graph_transformer import GraphLevelTransformer

# ------------------------------------------------------------
# GLOBALS
# ------------------------------------------------------------

EMB_DIM = 32
RELATIONS = {"treatedby": 0}   # extensible later

# ------------------------------------------------------------
# STEP 1: LOAD + NORMALIZE
# ------------------------------------------------------------

def normalize(text):
    return str(text).strip().replace(" ", "_").lower()

def load_triples(path):
    df = pd.read_csv(path)
    for c in ["head", "relation", "tail"]:
        df[c] = df[c].apply(normalize)
    return list(df.itertuples(index=False, name=None))

# ------------------------------------------------------------
# STEP 2: BUILD KG
# ------------------------------------------------------------

VALID_TREATMENTS = {
    "copper_spray", "pruning", "sanitation", "crop_rotation",
    "soil_solarization", "resistant_variety",
    "remove_infected_plants", "biological_control",
    "proper_irrigation"
}

def build_entity_graph(triples):
    G = nx.DiGraph()
    for h, r, t in triples:
        if r == "treatedby" and t not in VALID_TREATMENTS:
            continue
        G.add_edge(h, t, relation=r)
    return G

# ------------------------------------------------------------
# STEP 3: DISEASE-CENTRIC SUBGRAPH
# ------------------------------------------------------------

class DiseaseSubgraphExtractor:
    def __init__(self, G):
        self.G = G

    def extract(self, disease):
        SG = nx.DiGraph()
        SG.add_node(disease)

        for nbr in self.G.successors(disease):
            rel = self.G[disease][nbr]["relation"]
            if rel in {"treatedby", "hassymptom", "showssymptom"}:
                SG.add_edge(disease, nbr, relation=rel)

        for nbr in self.G.successors(disease):
            if self.G[disease][nbr]["relation"] == "similarto":
                for x in self.G.successors(nbr):
                    if self.G[nbr][x]["relation"] == "treatedby":
                        SG.add_edge(disease, nbr, relation="similarto")
                        SG.add_edge(nbr, x, relation="treatedby")

        return SG

# ------------------------------------------------------------
# STEP 4: HELPERS
# ------------------------------------------------------------

def get_all_diseases(G):
    return sorted({u for u, v, d in G.edges(data=True) if d["relation"] == "treatedby"})

def get_all_treatments(G):
    return {v for u, v, d in G.edges(data=True) if d["relation"] == "treatedby"}

def remove_treatedby_edges(G, disease):
    Gc = G.copy()
    for _, v, d in list(Gc.edges(disease, data=True)):
        if d["relation"] == "treatedby":
            Gc.remove_edge(disease, v)
    return Gc

# ------------------------------------------------------------
# STEP 5: HARD NEGATIVES
# ------------------------------------------------------------

def hard_negative_sampling(G, disease, positives, k=3):
    negatives = set()
    for nbr in G.successors(disease):
        if G[disease][nbr]["relation"] == "similarto":
            for x in G.successors(nbr):
                if G[nbr][x]["relation"] == "treatedby":
                    negatives.add(x)
    negatives -= positives
    return set(random.sample(list(negatives), min(k, len(negatives))))

# ------------------------------------------------------------
# STEP 6: TUCKER MATCHER
# ------------------------------------------------------------

class TuckER(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Parameter(torch.randn(dim, dim, dim) * 0.1)

    def forward(self, es, er, eo):
        x = torch.tensordot(self.W, es, dims=([0], [0]))
        x = torch.tensordot(x, er, dims=([0], [0]))
        return torch.sum(x * eo, dim=-1)

# ------------------------------------------------------------
# STEP 7: METRICS
# ------------------------------------------------------------

def evaluate(scores, positives, k_vals=(1,3,5)):
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ranks = [i+1 for i,(n,_) in enumerate(ranked) if n in positives]
    mrr = np.mean([1/r for r in ranks]) if ranks else 0.0
    hits = {k: int(any(p in [n for n,_ in ranked[:k]] for p in positives))
            for k in k_vals}
    return mrr, hits

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

if __name__ == "__main__":

    random.seed(42)
    torch.manual_seed(42)

    G = build_entity_graph(load_triples("plant_kg_triples.csv"))
    extractor = DiseaseSubgraphExtractor(G)

    diseases = get_all_diseases(G)
    treatments = get_all_treatments(G)

    sample = subgraph_to_tensors(extractor.extract(diseases[0]), G)

    gnn = MinimalGNN(sample["x"].shape[1], 64, EMB_DIM)
    transformer = GraphLevelTransformer(EMB_DIM, 4, 2)
    matcher = TuckER(EMB_DIM)

    relation_emb = nn.Embedding(len(RELATIONS), EMB_DIM)

    for p in gnn.parameters():
        p.requires_grad = False

    optimizer = optim.Adam(
        list(transformer.parameters()) +
        list(matcher.parameters()) +
        list(relation_emb.parameters()),
        lr=1e-3
    )

    criterion = nn.BCEWithLogitsLoss()

    # ---------------- TRAINING ----------------

    for epoch in range(50):
        total, steps = 0, 0

        for d in diseases:
            SG = extractor.extract(d)
            data = subgraph_to_tensors(SG, G)

            positives = {v for u,v,r in SG.edges(data="relation") if r=="treatedby"}
            if not positives:
                continue

            negatives = hard_negative_sampling(G, d, positives)

            node_emb = gnn(data["x"], data["edge_index"])
            refined,_ = transformer(node_emb)

            es = refined[data["node2id"][d]]
            er = relation_emb(torch.tensor(RELATIONS["treatedby"]))

            scores, labels = [], []
            for t in positives | negatives:
                if t not in data["node2id"]:
                    continue
                eo = refined[data["node2id"][t]]
                scores.append(matcher(es, er, eo))
                labels.append(1.0 if t in positives else 0.0)

            loss = criterion(torch.stack(scores), torch.tensor(labels))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item()
            steps += 1

        print(f"Epoch {epoch:02d} | Avg Loss {total/max(1,steps):.4f}")

    # ---------------- LEAK-FREE EVALUATION ----------------

    mrrs, h1, h3, h5 = [], [], [], []

    for d in diseases:
        true_pos = {v for u,v,r in G.edges(data="relation") if u==d and r=="treatedby"}
        if not true_pos:
            continue

        Gt = remove_treatedby_edges(G, d)
        SG = DiseaseSubgraphExtractor(Gt).extract(d)
        data = subgraph_to_tensors(SG, Gt)

        node_emb = gnn(data["x"], data["edge_index"])
        refined,_ = transformer(node_emb)

        es = refined[data["node2id"][d]]
        er = relation_emb(torch.tensor(RELATIONS["treatedby"]))

        scores = {}
        for t in treatments:
            if t in data["node2id"]:
                eo = refined[data["node2id"][t]]
                scores[t] = matcher(es, er, eo).item()

        mrr, hits = evaluate(scores, true_pos)
        mrrs.append(mrr)
        h1.append(hits[1]); h3.append(hits[3]); h5.append(hits[5])

    print("\n==== FINAL LEAK-FREE EVALUATION ====")
    print(f"MRR    : {np.mean(mrrs):.4f}")
    print(f"Hits@1 : {np.mean(h1):.4f}")
    print(f"Hits@3 : {np.mean(h3):.4f}")
    print(f"Hits@5 : {np.mean(h5):.4f}")
