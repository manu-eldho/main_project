from kg_engine import TreatmentPredictionEngine

if __name__ == "__main__":
    engine = TreatmentPredictionEngine("cleaned_kg.csv")

    predictions = engine.predict(
        disease="Pepper__bell___Bacterial_spot",
        top_k=5
    )

    print("\nRecommended treatments:\n")
    for i, (t, s) in enumerate(predictions, 1):
        print(f"{i}. {t} (score={s:.3f})")
