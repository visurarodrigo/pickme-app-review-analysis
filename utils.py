import re
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS


SENTIMENT_COLORS = {
    "Positive": "#16a34a",
    "Negative": "#dc2626",
}


def load_data(path):
    """
    Loads the sentiment analysis data from a CSV file.

    Args:
        path (str): Path to the CSV file containing the sentiment analysis data.

    Returns:
        pd.DataFrame: The sentiment analysis data loaded from the CSV file.
    """
    df = pd.read_csv(path)

    # Convert date column to datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Convert rating and confidence columns to numeric
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

    # Drop rows with missing dates
    df = df.dropna(subset=["date"])

    return df


def get_analyzed_reviews(df):
    """
    Returns a DataFrame containing only the reviews with sentiment analysis results.

    Args:
        df (pd.DataFrame): The DataFrame containing the sentiment analysis data.

    Returns:
        pd.DataFrame: A DataFrame containing only the reviews with sentiment analysis results.
    """
    # Filter the DataFrame to only include reviews with sentiment analysis results
    analyzed_reviews = df[df["sentiment"].isin(["Positive", "Negative"])].copy()

    return analyzed_reviews


def sentiment_distribution_chart(df):
    """
    Returns a pie chart showing the distribution of sentiment analysis results.

    Args:
        df (pd.DataFrame): The DataFrame containing the sentiment analysis data.

    Returns:
        plotly.graph_objs.Figure: A pie chart showing the distribution of sentiment analysis results.

    """
    if df.empty:
        return None

    counts = df["sentiment"].value_counts().reset_index()
    counts.columns = ["sentiment", "count"]

    # Create a pie chart
    fig = px.pie(
        counts,
        names="sentiment",
        values="count",
        hole=0.45,
        color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
    )

    # Add percent labels to the chart
    fig.update_traces(textinfo="percent+label")

    # Update the chart layout
    fig.update_layout(
        margin=dict(t=20, b=20, l=20, r=20),  # Add some space around the chart
        height=320,  # Set a fixed height for the chart
        showlegend=True,  # Show the legend
    )

    return fig


def sentiment_trend_chart(df):
    """
    Returns an area chart showing the trend of sentiment analysis results over time.

    Args:
        df (pd.DataFrame): The DataFrame containing the sentiment analysis data.

    Returns:
        plotly.graph_objs.Figure: An area chart showing the trend of sentiment analysis results over time.
    """
    if df.empty:
        return None

    # Resample the DataFrame to compute the weekly sum of Positive and Negative reviews
    trend = df.set_index("date").resample("W").agg(
        # Use a lambda function to compute the sum of Positive reviews
        Positive=("sentiment", lambda x: (x == "Positive").sum()),
        # Use a lambda function to compute the sum of Negative reviews
        Negative=("sentiment", lambda x: (x == "Negative").sum()),
    ).reset_index()

    # Melt the DataFrame to create a single column for the sentiment
    trend = trend.melt(
        id_vars="date",
        var_name="sentiment",
        value_name="count",
    )

    # Create an area chart
    fig = px.area(
        trend,
        x="date",
        y="count",
        color="sentiment",
        # Add custom labels to the x-axis and y-axis
        labels={
            "date": "Date",
            "count": "Reviews",
        },
        # Use a discrete color map to differentiate between Positive and Negative reviews
        color_discrete_map=SENTIMENT_COLORS,
    )

    # Update the chart layout
    fig.update_layout(
        # Add some space around the chart
        margin=dict(t=20, b=20, l=20, r=20),
        # Set a fixed height for the chart
        height=350,
        # Set custom titles for the x-axis and y-axis
        xaxis_title="Date",
        yaxis_title="Reviews",
    )

    return fig


def get_top_reviews(df, sentiment, n=5):
    """
    Returns the top n reviews for a given sentiment type.

    Args:
        df (pd.DataFrame): The DataFrame containing the sentiment analysis data.
        sentiment (str): The sentiment type to filter by (Positive or Negative).
        n (int): The number of reviews to return.

    Returns:
        pd.DataFrame: A DataFrame containing the top n reviews for the given sentiment type.
    """
    filtered = df[df["sentiment"] == sentiment].copy()

    # Calculate the length of each review to filter out very short ones
    filtered["text_length"] = filtered["review_text"].astype(str).str.len()

    # Filter out reviews with less than 15 characters
    filtered = filtered[filtered["text_length"] >= 15]

    # Sort the filtered DataFrame by confidence and text length in descending order
    filtered = filtered.sort_values(
        ["confidence", "text_length"],
        ascending=[False, False],
    )

    # Return the top n reviews
    return filtered[["review_text", "rating", "date", "confidence"]].head(n)


def prepare_text_for_wordcloud(df, sentiment=None):
    """
    Prepares the text for generating a word cloud.

    Args:
        df (pd.DataFrame): The DataFrame containing the sentiment analysis data.
        sentiment (str): Optional sentiment type to filter by (Positive or Negative).

    Returns:
        str: The prepared text for generating a word cloud.
    """
    text_df = df.copy()

    if sentiment:
        text_df = text_df[text_df["sentiment"] == sentiment]

    # Join all review texts together
    text = " ".join(text_df["review_text"].dropna().astype(str))

    # Convert to lowercase
    text = text.lower()

    # Keep only English letters and spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def generate_wordcloud(text):
    """
    Generates a word cloud from the given text.

    Args:
        text (str): The text to generate a word cloud from.

    Returns:
        wordcloud.WordCloud: The generated word cloud.
    """
    if not text.strip():
        return None

    stopwords = set(STOPWORDS)

    # Common words that are not very useful in this dashboard
    additional_stopwords = {
        "pickme",
        "pick",
        "app",
        "apps",
        "driver",
        "drivers",
        "ride",
        "rides",
        "please",
        "very",
        "time",
        "today",
        "me",
        "one",
        "use",
        "using",
    }

    # Update the stopwords set
    stopwords.update(additional_stopwords)

    wordcloud = WordCloud(
        width=900,
        height=450,
        background_color="white",
        stopwords=stopwords,
        max_words=100,
        colormap="viridis",
    )

    return wordcloud.generate(text)
