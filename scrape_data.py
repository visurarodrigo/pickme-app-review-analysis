"""
scrape_data.py
---------------
Scrapes Google Play Store reviews for the PickMe passenger app and saves them
as raw and cleaned CSV files inside the `data/` directory.
"""

from pathlib import Path
import pandas as pd
from google_play_scraper import Sort, reviews

# Target app package name on Google Play (PickMe passenger app)
APP_ID = "com.pickme.passenger"

# Directory where scraped CSVs will be stored
DATA_DIR = Path("data")
# Ensure the data directory exists before writing files
DATA_DIR.mkdir(exist_ok=True)

# Number of reviews to request per fetch call
COUNT = 2000


def fetch_reviews(country_code):
    """
    Fetches the latest reviews for APP_ID from a specific country store.

    Args:
        country_code (str): Two-letter ISO country code (e.g. "lk", "us").

    Returns:
        list[dict]: A list of review dictionaries returned by google_play_scraper.
    """
    print(f"Trying country code: {country_code}")

    # Request reviews from the Google Play Store:
    # - lang: language of reviews to fetch
    # - country: which Play Store country catalog to use
    # - sort: order by newest first
    # - count: maximum number of reviews to return
    result, _ = reviews(
        APP_ID,
        lang="en",
        country=country_code,
        sort=Sort.NEWEST,
        count=COUNT,
    )

    return result


def main():
    """Orchestrates scraping, cleaning, and saving the reviews."""
    print("Starting PickMe Google Play review collection...")

    # First attempt: fetch from Sri Lanka's Play Store (app is primarily used there)
    result = fetch_reviews("lk")

    # Fallback: if no reviews are returned for Sri Lanka, try the US store
    if len(result) == 0:
        result = fetch_reviews("us")

    # Convert the list of review dicts into a pandas DataFrame for easier processing
    df = pd.DataFrame(result)

    # If the DataFrame is empty, abort gracefully
    if df.empty:
        print("No reviews found. Check your internet connection or try again.")
        return

    # Keep only the columns relevant for our analysis
    df = df[["content", "score", "at"]].copy()

    # Rename columns to friendlier, more descriptive names
    df.rename(
        columns={
            "content": "review_text",
            "score": "rating",
            "at": "date",
        },
        inplace=True,
    )

    # Save the raw (unmodified) dataset first to preserve original data
    raw_path = DATA_DIR / "pickme_reviews_raw.csv"
    df.to_csv(raw_path, index=False, encoding="utf-8")
    print(f"Raw data saved: {raw_path}")

    # Basic cleaning steps:
    # 1. Ensure review_text is a string and trim leading/trailing whitespace
    df["review_text"] = df["review_text"].astype(str).str.strip()
    # 2. Remove very short reviews (less than 3 characters) as they carry little meaning
    df = df[df["review_text"].str.len() >= 3]
    # 3. Drop duplicate reviews based on identical text
    df = df.drop_duplicates(subset=["review_text"])

    # Parse the date column to datetime; invalid dates become NaT
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # Drop rows missing any of the essential fields
    df = df.dropna(subset=["review_text", "rating", "date"])

    # Sort by date in descending order (newest first) and reset the index
    df = df.sort_values("date", ascending=False).reset_index(drop=True)

    # Save the cleaned dataset, ready for analysis
    clean_path = DATA_DIR / "pickme_reviews_clean.csv"
    df.to_csv(clean_path, index=False, encoding="utf-8")

    print(f"Clean data saved: {clean_path}")
    print(f"Total clean reviews: {len(df)}")
    # Display a quick preview of the cleaned data
    print(df.head())


if __name__ == "__main__":
    main()
