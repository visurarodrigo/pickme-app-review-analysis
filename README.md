# pickme-app-review-analysis

Sentiment analysis of PickMe app reviews using Hugging Face and Streamlit.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## What it does

- Accepts a PickMe app review as input.
- Uses Hugging Face `pipeline("sentiment-analysis")` to classify sentiment.
- Shows predicted sentiment label and confidence score.
