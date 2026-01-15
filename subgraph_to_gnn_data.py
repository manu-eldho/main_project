import torch
import torch.nn as nn
import networkx as nx

# ------------------------------------------------------------
# NODE TYPE DEFINITIONS
# ------------------------------------------------------------
# 0 → disease
# 1 → treatment
# 2 → symptom
# 3 → other (crop, practice, etc.)

def get_node_type(node, G: nx.DiGraph):
    # Incoming hasDisease → disease
    for _, _, d in G.in_edges(node, data=True):
        if d["relation"] == "hasdisease":
            return 0

    # Outgoing treatedBy → treatment
    for _, _, d in G.out_edges(node, data=True):
        if d["relation"] == "treatedby":
            return 1

    # Outgoing symptom relations → symptom
    for _, _, d in G.out_edges(node, data=True):
        if d["relation"] in {"hassymptom", "showssymptom"}:
            return 2

    return 3


# ------------------------------------------------------------
# SUBGRAPH → GNN TENSORS
# ------------------------------------------------------------

def subgraph_to_tensors(subgraph: nx.DiGraph, G: nx.DiGraph):
    """
    Converts a NetworkX subgraph into tensors for GNN input
    with node-type embeddings.
    """

    # ---- Node mapping ----
    nodes = list(subgraph.nodes())
    node2id = {node: i for i, node in enumerate(nodes)}

    # ---- Edge index and edge types ----
    edge_index = []
    edge_types = []
    relation2id = {}

    for u, v, data in subgraph.edges(data=True):
        edge_index.append([node2id[u], node2id[v]])

        rel = data["relation"]
        if rel not in relation2id:
            relation2id[rel] = len(relation2id)
        edge_types.append(relation2id[rel])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_types = torch.tensor(edge_types, dtype=torch.long)

    # --------------------------------------------------------
    # NODE FEATURES = RANDOM + NODE-TYPE EMBEDDING
    # --------------------------------------------------------

    EMB_DIM = 32
    TYPE_DIM = 8
    NUM_TYPES = 4

    # Base random features (break symmetry)
    base_x = torch.randn((len(nodes), EMB_DIM - TYPE_DIM))

    # Node type ids
    type_ids = torch.tensor(
        [get_node_type(node, G) for node in nodes],
        dtype=torch.long
    )

    # Learnable type embeddings
    type_embedding = nn.Embedding(NUM_TYPES, TYPE_DIM)
    type_x = type_embedding(type_ids)

    # Final node features
    x = torch.cat([base_x, type_x], dim=1)

    return {
        "x": x,
        "edge_index": edge_index,
        "edge_type": edge_types,
        "node2id": node2id,
        "relation2id": relation2id
    }
