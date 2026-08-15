from pathlib import Path
import pandas as pd
from transformers import pipeline

DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "pickme_reviews_clean.csv"
OUTPUT_FILE = DATA_DIR / "pickme_reviews_with_sentiment.csv"

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


def detect_language(text):
    text = str(text)

    for char in text:
        # Sinhala Unicode range
        if "\u0D80" <= char <= "\u0DFF":
            return "Sinhala"

        # Tamil Unicode range
        if "\u0B80" <= char <= "\u0BFF":
            return "Tamil"

    return "English"


def main():
    print("Loading cleaned PickMe reviews...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Loaded {len(df)} reviews.")

    print("Detecting languages...")

    df["language"] = df["review_text"].apply(detect_language)

    print("\nLanguage distribution:")
    print(df["language"].value_counts())

    english_mask = df["language"] == "English"

    df["sentiment"] = "Not Analyzed"
    df["confidence"] = None

    english_reviews = df.loc[english_mask, "review_text"].fillna("").astype(str).tolist()

    print(f"\nEnglish reviews to analyze: {len(english_reviews)}")
    print("Non-English reviews kept but not analyzed:", len(df) - len(english_reviews))

    if len(english_reviews) > 0:
        print("Loading Hugging Face sentiment model...")

        classifier = pipeline(
            "sentiment-analysis",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME,
        )

        print("Running sentiment predictions...")

        results = classifier(
            english_reviews,
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

        df.loc[english_mask, "sentiment"] = sentiments
        df.loc[english_mask, "confidence"] = confidences

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print("\nSentiment distribution for English reviews:")
    print(df[df["sentiment"] != "Not Analyzed"]["sentiment"].value_counts())

    print(f"\nSaved file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()