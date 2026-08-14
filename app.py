import streamlit as st
from transformers import pipeline


@st.cache_resource
def load_sentiment_pipeline():
    return pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    )


def main() -> None:
    st.set_page_config(page_title="PickMe Review Sentiment Analysis", page_icon="⭐")
    st.title("PickMe App Review Sentiment Analysis")
    st.write("Analyze PickMe app reviews using a Hugging Face sentiment model.")

    review_text = st.text_area(
        "Enter a PickMe app review",
        placeholder="Example: The app is easy to use and the drivers arrive quickly.",
        height=160,
    )

    if st.button("Analyze sentiment"):
        if not review_text.strip():
            st.warning("Please enter a review before running analysis.")
            return

        try:
            analyzer = load_sentiment_pipeline()
            result = analyzer(review_text)[0]
        except OSError:
            st.error(
                "Unable to load the Hugging Face model. Please check network access "
                "or local model availability and try again."
            )
            return

        label = result["label"].capitalize()
        confidence = result["score"] * 100

        st.subheader("Result")
        st.write(f"Sentiment: **{label}**")
        st.write(f"Confidence: **{confidence:.2f}%**")


if __name__ == "__main__":
    main()
