import pandas as pd

def validate_kg_csv(
    csv_path,
    sep=",",
    allow_self_loops=True,
    verbose=True
):
    """
    Validates a knowledge graph CSV with NO HEADER.
    Expected format per row:
        subject, relation, object
    """

    # Load CSV safely
    try:
        df = pd.read_csv(csv_path, header=None, sep=sep)
    except Exception as e:
        raise ValueError(f"Failed to read CSV: {e}")

    # 1. Column count check
    if df.shape[1] != 3:
        raise ValueError(
            f"Invalid column count: expected 3, found {df.shape[1]}"
        )

    df.columns = ["head", "relation", "tail"]

    # 2. Missing / empty value check
    if df.isnull().any().any():
        bad_rows = df[df.isnull().any(axis=1)]
        raise ValueError(
            f"Missing values found in rows:\n{bad_rows}"
        )

    # Strip whitespace
    df = df.applymap(lambda x: str(x).strip())

    # 3. Empty string check
    empty_mask = (df == "")
    if empty_mask.any().any():
        bad_rows = df[empty_mask.any(axis=1)]
        raise ValueError(
            f"Empty strings found in rows:\n{bad_rows}"
        )

    # 4. Relation leakage check
    entities = set(df["head"]).union(set(df["tail"]))
    relations = set(df["relation"])

    leakage = entities.intersection(relations)
    if leakage:
        raise ValueError(
            f"Relation/entity overlap detected: {leakage}"
        )

    # 5. Self-loop check (optional)
    if not allow_self_loops:
        self_loops = df[df["head"] == df["tail"]]
        if not self_loops.empty:
            raise ValueError(
                f"Self-loops detected:\n{self_loops}"
            )

    # 6. Duplicate triplet check
    duplicates = df[df.duplicated()]
    if not duplicates.empty:
        raise ValueError(
            f"Duplicate triplets found:\n{duplicates}"
        )

    if verbose:
        print("✅ CSV validation PASSED")
        print(f"   Triplets      : {len(df)}")
        print(f"   Entities      : {len(entities)}")
        print(f"   Relations     : {len(relations)}")

    return df

df = validate_kg_csv("cleaned_kg.csv", allow_self_loops=False)
