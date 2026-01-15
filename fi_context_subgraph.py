def extract_context_subgraph(df, anchor_entity, max_triplets=None):
    """
    Extract context-level subgraph for a given co-occurring entity.
    Each triplet is treated as a node (TGformer Section III-A).
    """

    # Find co-occurring triplets
    mask = (df["head"] == anchor_entity) | (df["tail"] == anchor_entity)
    sub_df = df[mask].copy()

    # Optional: limit number of contextual triplets
    if max_triplets is not None and len(sub_df) > max_triplets:
        sub_df = sub_df.sample(
            n=max_triplets,
            random_state=42
        )

    # Convert to list of triplets
    triplets = list(
        sub_df[["head", "relation", "tail"]]
        .itertuples(index=False, name=None)
    )

    return triplets

if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("cleaned_kg.csv", header=None)
    df.columns = ["head", "relation", "tail"]

    anchor = "Pepper__bell___Bacterial_spot"
    subgraph = extract_context_subgraph(df, anchor)

    print(f"Anchor entity: {anchor}")
    print(f"Contextual triplets ({len(subgraph)}):")
    for t in subgraph:
        print("  ", t)
