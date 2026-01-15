import pandas as pd

# Load raw CSV
df = pd.read_csv("updated_final_kg.csv", header=None)
df.columns = ["head", "relation", "tail"]

# 1. Drop fully empty rows
df = df.dropna(how="all")

# 2. Drop rows with missing head / relation / tail
df = df.dropna(subset=["head", "relation", "tail"])

# 3. (Optional but recommended)
# Remove pathogen/treatment relations for healthy entities
invalid_relations = {"caused_by_pathogen", "treated_by"}

df = df[~(
    df["head"].str.contains("healthy", case=False) &
    df["relation"].isin(invalid_relations)
)]

# Save cleaned KG
df.to_csv("cleaned_kg.csv", index=False, header=False)

print("✅ KG cleaned and saved as cleaned_kg.csv")
