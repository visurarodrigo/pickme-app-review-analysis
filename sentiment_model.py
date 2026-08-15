from pathlib import Path
import pandas as pd
from transformers import pipeline

DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "pickme_reviews_clean.csv"
OUTPUT_FILE = DATA_DIR / "pickme_reviews_with_sentiment.csv"

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

def main():
    print("Loading cleaned PickMe reviews...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Loaded {len(df)} reviews.")
    print("Loading Hugging Face sentiment model...")

    classifier = pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
    )

    reviews = df["review_text"].fillna("").astype(str).tolist()

    print("Running sentiment predictions...")

    results = classifier(
        reviews,
        batch_size=16,
        truncation=True,
    )

    sentiments = []
    confidences = []

    for result in results:
        label = result["label"].lower()
        score = result["score"]

        if label == "positive":
            sentiments.append("Positive")
        elif label == "negative":
            sentiments.append("Negative")
        else:
            sentiments.append(label.title())

        confidences.append(round(score, 4))

    df["sentiment"] = sentiments
    df["confidence"] = confidences

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print("Sentiment prediction completed.")
    print("\nSentiment distribution:")
    print(df["sentiment"].value_counts())
    print(f"\nSaved file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()