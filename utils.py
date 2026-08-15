import re
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS


SENTIMENT_COLORS = {
    "Positive": "#16a34a",
    "Negative": "#dc2626",
}


def load_data(path):
    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

    df = df.dropna(subset=["date"])

    return df


def get_analyzed_reviews(df):
    return df[df["sentiment"].isin(["Positive", "Negative"])].copy()


def sentiment_distribution_chart(df):
    if df.empty:
        return None

    counts = df["sentiment"].value_counts().reset_index()
    counts.columns = ["sentiment", "count"]

    fig = px.pie(
        counts,
        names="sentiment",
        values="count",
        hole=0.45,
        color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
    )

    fig.update_traces(textinfo="percent+label")

    fig.update_layout(
        margin=dict(t=20, b=20, l=20, r=20),
        height=320,
        showlegend=True,
    )

    return fig


def sentiment_trend_chart(df):
    if df.empty:
        return None

    trend = df.set_index("date").resample("W").agg(
        Positive=("sentiment", lambda x: (x == "Positive").sum()),
        Negative=("sentiment", lambda x: (x == "Negative").sum()),
    ).reset_index()

    trend = trend.melt(
        id_vars="date",
        var_name="sentiment",
        value_name="count",
    )

    fig = px.area(
        trend,
        x="date",
        y="count",
        color="sentiment",
        labels={
            "date": "Date",
            "count": "Reviews",
        },
        color_discrete_map=SENTIMENT_COLORS,
    )

    fig.update_layout(
        margin=dict(t=20, b=20, l=20, r=20),
        height=350,
        xaxis_title="Date",
        yaxis_title="Reviews",
    )

    return fig


def get_top_reviews(df, sentiment, n=5):
    filtered = df[df["sentiment"] == sentiment].copy()

    filtered["text_length"] = filtered["review_text"].astype(str).str.len()

    # Show more meaningful reviews, not only very short ones like "good"
    filtered = filtered[filtered["text_length"] >= 15]

    filtered = filtered.sort_values(
        ["confidence", "text_length"],
        ascending=[False, False],
    )

    return filtered[["review_text", "rating", "date", "confidence"]].head(n)


def prepare_text_for_wordcloud(df, sentiment=None):
    text_df = df.copy()

    if sentiment:
        text_df = text_df[text_df["sentiment"] == sentiment]

    text = " ".join(text_df["review_text"].dropna().astype(str))
    text = text.lower()

    # Keep only English letters and spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def generate_wordcloud(text):
    if not text.strip():
        return None

    stopwords = set(STOPWORDS)

    # Common words that are not very useful in this dashboard
    stopwords.update({
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
    })

    wordcloud = WordCloud(
        width=900,
        height=450,
        background_color="white",
        stopwords=stopwords,
        max_words=100,
        colormap="viridis",
    )

    return wordcloud.generate(text)