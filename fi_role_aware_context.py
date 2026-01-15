def extract_role_aware_context(df, anchor_entity, max_triplets=None):
    """
    Extract role-aware context-level subgraph for a given anchor entity.

    Returns a list of dicts:
    {
        "triplet": (head, relation, tail),
        "anchor_role": "SUBJECT" | "OBJECT"
    }
    """

    context = []

    for _, row in df.iterrows():
        h, r, t = row["head"], row["relation"], row["tail"]

        if h == anchor_entity:
            context.append({
                "triplet": (h, r, t),
                "anchor_role": "SUBJECT"
            })

        elif t == anchor_entity:
            context.append({
                "triplet": (h, r, t),
                "anchor_role": "OBJECT"
            })

    # Optional: sample contextual triplets (as TGformer does)
    if max_triplets is not None and len(context) > max_triplets:
        import random
        random.seed(42)
        context = random.sample(context, max_triplets)

    return context

if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("cleaned_kg.csv", header=None)
    df.columns = ["head", "relation", "tail"]

    anchor = "Pepper__bell___Bacterial_spot"
    context = extract_role_aware_context(df, anchor)

    print(f"Anchor entity: {anchor}")
    print(f"Role-aware contextual triplets ({len(context)}):\n")

    for item in context:
        print(item["anchor_role"], "→", item["triplet"])
