import streamlit as st
import pandas as pd
from pathlib import Path

from utils import (
    load_data,
    get_analyzed_reviews,
    sentiment_distribution_chart,
    sentiment_trend_chart,
    get_top_reviews,
    prepare_text_for_wordcloud,
    generate_wordcloud,
)


# Page configuration
st.set_page_config(
    page_title="PickMe Sentiment Dashboard",
    page_icon="🚗",
    layout="wide",
)


# Dataset path
DATA_PATH = Path(__file__).parent / "data" / "pickme_reviews_with_sentiment.csv"


# Load and cache dataset for better performance
@st.cache_data
def load_pickme_data() -> pd.DataFrame:
    return load_data(DATA_PATH)


df = load_pickme_data()
analyzed = get_analyzed_reviews(df)


# Dashboard title and description
st.title("PickMe Google Play Review Sentiment Analysis")

st.caption(
    "Sentiment analysis of PickMe Google Play reviews using a pretrained Hugging Face model. "
    "Sinhala and Tamil reviews are kept in the dataset but not analyzed because the model is English-focused."
)


# Check whether valid data is available
if df.empty:
    st.error("No data found. Please check the data file.")
    st.stop()

if analyzed.empty:
    st.error("No analyzed English reviews found.")
    st.stop()

# Sidebar Filters

st.sidebar.header("Filters")

min_date = analyzed["date"].min().date()
max_date = analyzed["date"].max().date()

date_range = st.sidebar.slider(
    "Review date range",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
)

selected_sentiments = st.sidebar.multiselect(
    "Sentiment",
    options=["Positive", "Negative"],
    default=["Positive", "Negative"],
)

min_confidence = st.sidebar.slider(
    "Minimum confidence",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05,
)


# Apply date and confidence filters
base_filtered = analyzed[
    (analyzed["date"].dt.date >= date_range[0])
    & (analyzed["date"].dt.date <= date_range[1])
    & (analyzed["confidence"] >= min_confidence)
].copy()


# Apply selected sentiment filter
filtered = base_filtered[
    base_filtered["sentiment"].isin(selected_sentiments)
].copy()

# Key Metrics

total_reviews = len(df)
english_reviews = len(df[df["language"] == "English"])
non_english_reviews = total_reviews - english_reviews

positive_count = len(filtered[filtered["sentiment"] == "Positive"])
negative_count = len(filtered[filtered["sentiment"] == "Negative"])

positive_rate = (
    positive_count / len(filtered)
    if len(filtered) > 0
    else 0
)


# Display dashboard KPIs
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Reviews", total_reviews)
col2.metric("English Analyzed", len(analyzed))
col3.metric("Positive", positive_count)
col4.metric("Negative", negative_count)
col5.metric("Positive Rate", f"{positive_rate:.1%}")


st.caption(
    f"Non-English reviews kept but not analyzed: {non_english_reviews}"
)


if filtered.empty:
    st.warning("No reviews match the selected filters.")
    st.stop()

# Sentiment Distribution

st.subheader("Overall Sentiment Distribution")

distribution_chart = sentiment_distribution_chart(filtered)

if distribution_chart:
    st.plotly_chart(distribution_chart, use_container_width=True)
else:
    st.info("Not enough data for sentiment distribution.")

# Sentiment Trend

st.subheader("Sentiment Trend Over Time")

trend_chart = sentiment_trend_chart(filtered)

if trend_chart:
    st.plotly_chart(trend_chart, use_container_width=True)
else:
    st.info("Not enough data for sentiment trend.")


# Top Reviews

st.subheader("Top Positive and Negative Reviews")

review_column_config = {
    "review_text": st.column_config.TextColumn(
        "Review",
        width="large",
    ),
    "rating": st.column_config.NumberColumn(
        "Rating",
    ),
    "date": st.column_config.DatetimeColumn(
        "Date",
        format="YYYY-MM-DD",
    ),
    "confidence": st.column_config.ProgressColumn(
        "Confidence",
        min_value=0.0,
        max_value=1.0,
        format="%.3f",
    ),
}


left_col, right_col = st.columns(2)


with left_col:
    st.markdown("#### Top Positive")

    top_positive = get_top_reviews(base_filtered, "Positive", 5)

    if top_positive.empty:
        st.info("No positive reviews found.")
    else:
        st.dataframe(
            top_positive,
            use_container_width=True,
            hide_index=True,
            column_config=review_column_config,
        )


with right_col:
    st.markdown("#### Top Negative")

    top_negative = get_top_reviews(base_filtered, "Negative", 5)

    if top_negative.empty:
        st.info("No negative reviews found.")
    else:
        st.dataframe(
            top_negative,
            use_container_width=True,
            hide_index=True,
            column_config=review_column_config,
        )

# Word Cloud

st.subheader("Word Cloud")

tab_all, tab_positive, tab_negative = st.tabs(
    ["All Analyzed", "Positive", "Negative"]
)


with tab_all:
    text = prepare_text_for_wordcloud(base_filtered)
    wordcloud = generate_wordcloud(text)

    if wordcloud:
        st.image(wordcloud.to_array(), use_container_width=True)
    else:
        st.info("Not enough text to create a word cloud.")


with tab_positive:
    text = prepare_text_for_wordcloud(base_filtered, "Positive")
    wordcloud = generate_wordcloud(text)

    if wordcloud:
        st.image(wordcloud.to_array(), use_container_width=True)
    else:
        st.info("Not enough positive text to create a word cloud.")


with tab_negative:
    text = prepare_text_for_wordcloud(base_filtered, "Negative")
    wordcloud = generate_wordcloud(text)

    if wordcloud:
        st.image(wordcloud.to_array(), use_container_width=True)
    else:
        st.info("Not enough negative text to create a word cloud.")


# Model and data source
st.markdown("---")

st.caption(
    "Model: distilbert-base-uncased-finetuned-sst-2-english | "
    "Data source: Google Play Store reviews for PickMe Sri Lanka"
)