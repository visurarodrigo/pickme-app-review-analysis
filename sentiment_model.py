# Importing necessary libraries
from pathlib import Path
import pandas as pd
from transformers import pipeline

# Defining the directory where the data is stored
DATA_DIR = Path("data")

# Defining the input and output file paths
INPUT_FILE = DATA_DIR / "pickme_reviews_clean.csv"
OUTPUT_FILE = DATA_DIR / "pickme_reviews_with_sentiment.csv"

# Defining the model name for sentiment analysis
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

# Function to detect the language of a text
def detect_language(text):
    # Converting the text to string
    text = str(text)

    # Checking if the text contains Sinhala characters
    for char in text:
        if "\u0D80" <= char <= "\u0DFF":
            return "Sinhala"

    # Checking if the text contains Tamil characters
    for char in text:
        if "\u0B80" <= char <= "\u0BFF":
            return "Tamil"

    # If no special characters found, assuming it's English
    return "English"

# Main function
def main():
    # Printing a message to indicate that the program is loading the cleaned PickMe reviews
    print("Loading cleaned PickMe reviews...")

    # Loading the cleaned PickMe reviews from the CSV file
    df = pd.read_csv(INPUT_FILE)

    # Printing the number of reviews loaded
    print(f"Loaded {len(df)} reviews.")

    # Detecting the language of each review
    df["language"] = df["review_text"].apply(detect_language)

    # Printing the distribution of languages in the reviews
    print("\nLanguage distribution:")
    print(df["language"].value_counts())

    # Creating a mask for English reviews
    english_mask = df["language"] == "English"

    # Initializing sentiment and confidence columns
    df["sentiment"] = "Not Analyzed"
    df["confidence"] = None

    # Extracting English reviews
    english_reviews = df.loc[english_mask, "review_text"].fillna("").astype(str).tolist()

    # Printing the number of English reviews to analyze
    print(f"\nEnglish reviews to analyze: {len(english_reviews)}")
    print("Non-English reviews kept but not analyzed:", len(df) - len(english_reviews))

    # If there are English reviews to analyze
    if len(english_reviews) > 0:
        # Loading the Hugging Face sentiment analysis model
        print("Loading Hugging Face sentiment model...")
        classifier = pipeline(
            "sentiment-analysis",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME,
        )

        # Running sentiment predictions on the English reviews
        print("Running sentiment predictions...")
        results = classifier(
            english_reviews,
            batch_size=16,
            truncation=True,
        )

        # Initializing lists for sentiments and confidences
        sentiments = []
        confidences = []

        # Extracting sentiments and confidences from the results
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

        # Updating the sentiment and confidence columns in the dataframe
        df.loc[english_mask, "sentiment"] = sentiments
        df.loc[english_mask, "confidence"] = confidences

    # Saving the dataframe with sentiment analysis results to a CSV file
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    # Printing the distribution of sentiments for English reviews
    print("\nSentiment distribution for English reviews:")
    print(df[df["sentiment"] != "Not Analyzed"]["sentiment"].value_counts())

    # Printing the path of the saved file
    print(f"\nSaved file: {OUTPUT_FILE}")

# Running the main function if the script is executed directly
if __name__ == "__main__":
    main()