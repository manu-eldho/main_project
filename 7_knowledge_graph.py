import random
import csv
from itertools import combinations

random.seed(42)

# =========================
# PLANTVILLAGE CLASSES
# =========================
PLANT_DISEASES = {
    "Apple": ["Apple_Scab", "Black_Rot", "Cedar_Apple_Rust"],
    "Blueberry": ["Blueberry_Healthy"],
    "Cherry": ["Powdery_Mildew"],
    "Corn": ["Gray_Leaf_Spot", "Common_Rust", "Northern_Leaf_Blight"],
    "Grape": ["Black_Rot", "Esca", "Leaf_Blight"],
    "Orange": ["Citrus_Greening"],
    "Peach": ["Bacterial_Spot"],
    "Pepper": ["Bacterial_Spot"],
    "Potato": ["Early_Blight", "Late_Blight"],
    "Raspberry": ["Healthy"],
    "Soybean": ["Healthy"],
    "Squash": ["Powdery_Mildew"],
    "Strawberry": ["Leaf_Scorch"],
    "Tomato": [
        "Bacterial_Spot", "Early_Blight", "Late_Blight",
        "Leaf_Mold", "Septoria_Leaf_Spot",
        "Spider_Mites", "Target_Spot",
        "Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato_Mosaic_Virus"
    ]
}

# Remove healthy-only entries
PLANT_DISEASES = {
    p: [d for d in ds if "Healthy" not in d]
    for p, ds in PLANT_DISEASES.items()
    if any("Healthy" not in d for d in ds)
}

# =========================
# SYMPTOMS (EXPANDED)
# =========================
SYMPTOMS = [
    "Brown_Leaf_Spots", "Yellowing_Leaves", "Water_Soaked_Lesions",
    "Leaf_Wilting", "Dark_Margins", "White_Powdery_Growth",
    "Curling_Leaves", "Necrotic_Tissue", "Leaf_Scorch",
    "Mosaic_Patterns", "Stunted_Growth", "Vein_Yellowing"
]

# =========================
# TREATMENTS (EXPANDED)
# =========================
TREATMENTS = [
    "Fungicide", "Copper_Spray", "Crop_Rotation",
    "Resistant_Variety", "Remove_Infected_Plants",
    "Proper_Irrigation", "Soil_Solarization",
    "Biological_Control", "Pruning", "Sanitation"
]

PLANT_PARTS = ["Leaf", "Stem", "Fruit"]

# =========================
# BASE TRIPLE GENERATION
# =========================
triples = []

# Plant → Disease
for plant, diseases in PLANT_DISEASES.items():
    for disease in diseases:
        triples.append((plant, "hasDisease", disease))

# Disease → Symptoms / Treatments / PlantParts
disease_list = [d for ds in PLANT_DISEASES.values() for d in ds]

for disease in disease_list:
    for symptom in random.sample(SYMPTOMS, k=4):
        triples.append((disease, "hasSymptom", symptom))
    for treatment in random.sample(TREATMENTS, k=3):
        triples.append((disease, "treatedBy", treatment))
    triples.append((disease, "affectsPart", "Leaf"))

# =========================
# DERIVED RELATIONS
# =========================

# Symptom co-occurrence
for s1, s2 in random.sample(list(combinations(SYMPTOMS, 2)), 40):
    triples.append((s1, "coOccursWith", s2))

# Disease similarity (shared symptoms)
disease_symptoms = {
    d: set(
        t[2] for t in triples
        if t[0] == d and t[1] == "hasSymptom"
    )
    for d in disease_list
}

for d1, d2 in combinations(disease_list, 2):
    if len(disease_symptoms[d1] & disease_symptoms[d2]) >= 2:
        triples.append((d1, "similarTo", d2))

# Treatment applicability
for treatment in TREATMENTS:
    applicable = random.sample(disease_list, k=4)
    for disease in applicable:
        triples.append((treatment, "applicableTo", disease))

# Plant → Symptom (derived)
for plant, diseases in PLANT_DISEASES.items():
    seen = set()
    for d in diseases:
        for t in triples:
            if t[0] == d and t[1] == "hasSymptom":
                if t[2] not in seen:
                    triples.append((plant, "showsSymptom", t[2]))
                    seen.add(t[2])

# =========================
# NEGATIVE SAMPLING
# =========================
positive_set = set(triples)
entities = list(
    set([h for h,_,_ in triples] + [t for _,_,t in triples])
)

relations = list(set(r for _, r, _ in triples))
negative_triples = []

while len(negative_triples) < len(triples):
    h = random.choice(entities)
    r = random.choice(relations)
    t = random.choice(entities)
    if (h, r, t) not in positive_set:
        negative_triples.append((h, r, t))

triples.extend(negative_triples)

# =========================
# SAVE TO CSV
# =========================
with open("plant_kg_triples.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["head", "relation", "tail"])
    writer.writerows(triples)

print(f"Total triples generated: {len(triples)}")
