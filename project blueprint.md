
### The Project Blueprint

Here is the folder structure you will end up with by the end of this week:

```text
sl-pickme-sentiment/
│
├── .gitignore               # Hides virtual env, data, and cache files from GitHub
├── requirements.txt         # Lists all Python libraries needed to run the app
├── README.md                # The "storefront" of your project on GitHub
│
├── scrape_data.py           # [STEP 2] The script that downloads PickMe reviews
├── pickme_reviews.csv       # [STEP 2] The raw data file (Hidden from GitHub)
│
├── sentiment_model.py       # [STEP 3] Loads the Hugging Face model and runs predictions
│
├── utils.py                 # [STEP 4] Helper functions (Plotly charts, WordClouds)
└── app.py                   # [STEP 4] The main Streamlit UI dashboard
```

---

### Step-by-Step Execution Plan

Here is exactly what we are doing at each step, so you can track our progress:

#### **Step 1: Environment & Repo Setup (DONE ✅)**
*   **Files created:** `.gitignore`, `requirements.txt`, virtual environment (`venv/`).
*   **What happened:** We secured your workspace, linked it to GitHub, and installed the base tools.

#### **Step 2: Data Collection (CURRENT STEP 📍)**
*   **Files we will create:** `scrape_data.py`, `pickme_reviews.csv`.
*   **What happens:** You will run a script that talks to Google Play, downloads 800 English PickMe reviews, cleans the text (removes blank rows and duplicates), and saves it as a CSV locally. 
*   *Note: We will update `.gitignore` here so you don't accidentally upload a heavy CSV to GitHub.*

#### **Step 3: Hugging Face Model Inference (NEXT ⏭️)**
*   **Files we will create:** `sentiment_model.py`.
*   **What happens:** We will write a clean Python script that downloads a lightweight, pretrained sentiment model (like `distilbert-base-uncased-finetuned-sst-2-english`) from Hugging Face. We will run your CSV through this model to tag every review as Positive, Negative, or Neutral, and save the final dataset.

#### **Step 4: Streamlit Dashboard (UI & Visualization)**
*   **Files we will create:** `utils.py`, `app.py`.
*   **What happens:** 
    *   `utils.py` will hold the logic for generating your Plotly bar charts and WordClouds. Keeping this separate makes your code look very professional.
    *   `app.py` will be the actual webpage. It will load the data, call the utils, and display the dashboard for the user.

#### **Step 5: Deployment & Documentation**
*   **Files we will update/create:** `README.md`, push to GitHub.
*   **What happens:** We will connect your GitHub repo to Streamlit Cloud, ensure it deploys without errors, and write a beautiful README so recruiters understand your business logic and technical skills.

---