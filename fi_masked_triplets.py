MASK_TOKEN = "[MASK]"

def mask_context_triplets(role_aware_context):
    """
    Apply TGformer-style masking based on anchor role.

    Input:
        role_aware_context: list of dicts with keys:
            - triplet (h, r, t)
            - anchor_role ("SUBJECT" | "OBJECT")

    Output:
        list of masked triplets (h, r, t)
    """

    masked = []

    for item in role_aware_context:
        h, r, t = item["triplet"]
        role = item["anchor_role"]

        if role == "SUBJECT":
            masked.append((MASK_TOKEN, r, t))

        elif role == "OBJECT":
            masked.append((h, r, MASK_TOKEN))

        else:
            raise ValueError(f"Unknown role: {role}")

    return masked

if __name__ == "__main__":
    import pandas as pd
    from fi_role_aware_context import extract_role_aware_context

    df = pd.read_csv("cleaned_kg.csv", header=None)
    df.columns = ["head", "relation", "tail"]

    anchor = "Pepper__bell___Bacterial_spot"
    role_context = extract_role_aware_context(df, anchor)

    masked = mask_context_triplets(role_context)

    print("Masked contextual triplets:\n")
    for m in masked[:10]:
        print(m)
