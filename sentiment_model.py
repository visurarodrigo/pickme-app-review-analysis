# Import pathlib for filesystem path handling in a cross-platform way
from pathlib import Path
# Import pandas for loading and manipulating tabular review data
import pandas as pd
# Import Hugging Face's pipeline utility for easy use of pre-trained models
from transformers import pipeline

# Directory where input/output CSV files are stored
DATA_DIR = Path("data")
# Path to the cleaned reviews file produced by the earlier preprocessing step
INPUT_FILE = DATA_DIR / "pickme_reviews_clean.csv"
# Path where the sentiment-enriched output will be written
OUTPUT_FILE = DATA_DIR / "pickme_reviews_with_sentiment.csv"

# Pre-trained DistilBERT model fine-tuned on the Stanford Sentiment Treebank (SST-2)
# It outputs "POSITIVE" / "NEGATIVE" labels with a confidence score
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

def main():
    # Notify the user that we are beginning to load the input data
    print("Loading cleaned PickMe reviews...")

    # Read the cleaned PickMe reviews CSV into a pandas DataFrame
    df = pd.read_csv(INPUT_FILE)

    # Show how many reviews were successfully loaded
    print(f"Loaded {len(df)} reviews.")
    # Inform the user that the (potentially heavy) sentiment model is being downloaded/loaded
    print("Loading Hugging Face sentiment model...")

    # Create a ready-to-use sentiment-analysis pipeline that wraps the model + tokenizer
    classifier = pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
    )

    # Extract the review text column:
    # - fillna("") replaces any missing reviews with empty strings (model can't handle NaN)
    # - astype(str) ensures every entry is treated as a string
    # - tolist() converts the column into a Python list expected by the pipeline
    reviews = df["review_text"].fillna("").astype(str).tolist()

    # Inform the user that the (potentially slow) inference phase is starting
    print("Running sentiment predictions...")

    # Run the model on all reviews in batches of 16;
    # truncation=True safely cuts off reviews longer than the model's max input length
    results = classifier(
        reviews,
        batch_size=16,
        truncation=True,
    )

    # Parallel lists to store the cleaned sentiment label and confidence score per review
    sentiments = []
    confidences = []

    # Iterate through each model's prediction and normalize the label + score
    for result in results:
        # The pipeline returns labels like "POSITIVE"/"NEGATIVE"; lowercase for comparison
        label = result["label"].lower()
        # Confidence score (probability the model assigned to the predicted label)
        score = result["score"]

        # Map the model's raw label to a clean display label
        if label == "positive":
            sentiments.append("Positive")
        elif label == "negative":
            sentiments.append("Negative")
        else:
            # Fallback for any unexpected labels: title-case them
            sentiments.append(label.title())

        # Round the confidence score to 4 decimal places for tidy storage
        confidences.append(round(score, 4))

    # Add the new columns to the original DataFrame alongside the existing review data
    df["sentiment"] = sentiments
    df["confidence"] = confidences

    # Write the enriched DataFrame to disk as UTF-8 CSV (no pandas index column)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    # Final status messages summarizing the run
    print("Sentiment prediction completed.")
    print("\nSentiment distribution:")
    # Show how many reviews fell into each sentiment category
    print(df["sentiment"].value_counts())
    # Tell the user where the resulting file was saved
    print(f"\nSaved file: {OUTPUT_FILE}")


# Standard Python idiom: only run main() when this file is executed directly
# (not when it is imported as a module)
if __name__ == "__main__":
    main()