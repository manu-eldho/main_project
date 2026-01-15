def extract_role_aware_context(df, anchor):
    context = []
    for _, row in df.iterrows():
        h, r, t = row["head"], row["relation"], row["tail"]
        if h == anchor:
            context.append((h, r, t, "SUBJECT"))
        elif t == anchor:
            context.append((h, r, t, "OBJECT"))
    return context


def mask_context(context):
    masked = []
    for h, r, t, role in context:
        if role == "SUBJECT":
            masked.append(("[MASK]", r, t))
        else:
            masked.append((h, r, "[MASK]"))
    return masked
