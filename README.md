# PickMe Sri Lanka: Sentiment Analysis Dashboard 🚗

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pickme-app-review-analysis.streamlit.app/)

## 📌 Overview
This project analyzes customer sentiment towards **PickMe**, Sri Lanka's leading ride-hailing and food delivery app. By scraping authentic Google Play Store reviews and applying a pretrained Hugging Face Transformer model (`DistilBERT`), this dashboard provides actionable business insights into customer satisfaction, common pain points, and sentiment trends over time.

**Live Dashboard:** [pickme-app-review-analysis.streamlit.app](https://pickme-app-review-analysis.streamlit.app/)

## 🖼️ Dashboard Previews

**Overall Dashboard**

![Overall Dashboard](Dashboard%20previews/overall.jpg)

**Sentiment Trend Over Time**

![Trend over time](Dashboard%20previews/Trend%20over%20time.jpg)

**Top Positive and Negative Reviews**

![Top Positive and Negative](Dashboard%20previews/Top%20Positive%20and%20Negative.jpg)

**Word Clouds**

| All Reviews | Positive | Negative |
|---|---|---|
| ![Word Cloud All](Dashboard%20previews/Word%20Cloud%20-%20all.jpg) | ![Word Cloud Positive](Dashboard%20previews/Word%20Cloud%20-%20Positive.jpg) | ![Word Cloud Negative](Dashboard%20previews/Word%20Cloud%20-%20Negative.jpg) |

## 🛠️ Tech Stack
- **Data Collection:** `google-play-scraper`
- **NLP / Sentiment Analysis:** Hugging Face `transformers` (DistilBERT)
- **Data Processing:** `pandas`
- **Visualization & Dashboard:** `streamlit`, `plotly`, `wordcloud`
- **Deployment:** Streamlit Cloud

## 📊 Dataset & Methodology
- **Source:** Google Play Store (PickMe Passenger App - `com.pickme.passenger`)
- **Size:** ~1,400+ recent authentic reviews
- **Language Handling:** The dataset contains a mix of English, Sinhala, and Tamil reviews. Since the pretrained DistilBERT model used is optimized for English, a custom Unicode-based language detector was implemented. Sinhala and Tamil reviews were preserved in the dataset for transparency but excluded from the sentiment scoring to ensure high analytical accuracy.
- **Inference Pipeline:** Reviews were processed locally in batches to generate sentiment labels (Positive/Negative) and confidence scores, which were then saved to a lightweight CSV for fast, low-cost cloud deployment.

## 🚀 How to Run Locally
1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-name>
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the dashboard:
   ```bash
   streamlit run app.py
   ```
   *(Note: To reproduce the full scraping and NLP inference pipeline, run `scrape_data.py` followed by `sentiment_model.py`. You will need `transformers` and `torch` installed for the inference step).*

## 🔮 Future Improvements
- **Multilingual NLP:** Fine-tune XLM-RoBERTa or Sinhala-specific models to analyze the Sinhala and Tamil reviews currently marked as "Not Analyzed".
- **Aspect-Based Sentiment Analysis:** Extract specific pain points (e.g., "pricing", "driver behavior", "app bugs") rather than just overall positive/negative sentiment.

## 👨‍💻 Author
**Visura Rodrigo**  
Data Science & Business Analytics Undergraduate | KDU, Sri Lanka  
[LinkedIn](YOUR_LINKEDIN_URL) | [GitHub](YOUR_GITHUB_URL)

**⚖️ Disclaimer:** 
This is an independent, educational portfolio project created for data science learning purposes. The creator is not affiliated with, sponsored by, or endorsed by PickMe or Spacebyte Holdings (Pvt) Ltd. All data analyzed consists of publicly available user reviews from the Google Play Store. The insights presented reflect user sentiment and do not represent the views of the developer or the company.
