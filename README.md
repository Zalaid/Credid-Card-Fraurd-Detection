# Fraud Detection MLOps Pipeline

A machine learning project that trains and benchmarks 12+ models on real credit card fraud data, selects the best model, and serves real-time predictions via a REST API.

## Project Status

| Step | Description | Status |
|------|-------------|--------|
| 1 | Project setup & environment | Done |
| 2 | Dataset download & EDA | Done |
| 3 | Data preprocessing pipeline | Pending |
| 4 | MLflow experiment tracking setup | Pending |
| 5 | Train & benchmark 12 models | Pending |
| 6 | Model comparison & XGBoost tuning | Pending |
| 7 | FastAPI inference service | Pending |
| 8 | Tests | Pending |
| 9 | Docker containerization | Pending |
| 10 | CI/CD with GitHub Actions | Pending |
| 11 | Deploy on Render | Pending |
| 12 | README & demo polish | Pending |

## Project Structure

```
fraud-detection/
├── data/                    # Raw and processed data
├── notebooks/               # EDA and experimentation
├── src/
│   ├── data/                # Data loading and preprocessing
│   ├── models/              # Model training scripts
│   ├── evaluation/          # Metrics and comparison
│   └── api/                 # FastAPI application
├── mlruns/                  # MLflow experiment logs
├── models/                  # Saved trained models
├── tests/                   # Unit and integration tests
├── .github/workflows/       # CI/CD pipeline (populated in Step 10)
├── requirements.txt
└── README.md
```

## Tech Stack

- **ML:** scikit-learn, XGBoost, LightGBM, CatBoost
- **Imbalance handling:** SMOTE (imbalanced-learn)
- **Experiment tracking:** MLflow
- **API:** FastAPI + Uvicorn
- **Containerization:** Docker
- **CI/CD:** GitHub Actions
- **Deployment:** Render (free tier)

## Setup

```bash
# Clone the repo
git clone <repo-url>
cd fraud-detection

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

## EDA Notebook (Step 2)

`notebooks/01_eda.ipynb` covers the full exploratory analysis of the dataset:

| Section | What it shows |
|---------|--------------|
| Data overview | Shape, dtypes, memory usage |
| Missing values | Confirmed zero missing values — no imputation needed |
| Class imbalance | Pie chart + bar chart — 99.83% normal, 0.17% fraud (578:1 ratio) |
| Amount analysis | Fraud transactions tend to be smaller; histogram + box plot by class |
| Time analysis | Normal transactions dip at night; fraud is spread throughout the day |
| V1–V28 distributions | Side-by-side histograms — V4, V11, V12, V14, V17 show strongest separation |
| Correlation heatmap | Full 31×31 feature correlation matrix |
| Fraud correlations | Ranked bar chart of which features correlate most with fraud |
| Key findings | Summary table + why SMOTE and AUC-ROC are necessary |

To run the notebook yourself:
```bash
# Option 1 — open interactively
jupyter notebook notebooks/01_eda.ipynb

# Option 2 — execute headlessly from terminal (re-runs all cells and saves output)
jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output 01_eda.ipynb --output-dir notebooks/
```

## CI/CD Pipeline (.github/workflows/)

This folder is intentionally empty until **Step 10**. GitHub Actions automatically scans `.github/workflows/` for `.yml` files every time you push code to GitHub. Once `ci-cd.yml` is added there, every push to `main` will trigger this automated sequence:

```
git push → GitHub detects .github/workflows/ci-cd.yml →
  [1] Lint   — checks code style (flake8)
  [2] Test   — runs pytest, all tests must pass
  [3] Build  — builds the Docker image
  [4] Push   — uploads image to Docker Hub
  [5] Deploy — triggers a redeploy on Render (live URL updates automatically)
```

Nothing runs until the file exists there — that is why the folder is empty right now.

## Dataset

Credit Card Fraud Detection — 284,807 transactions, 492 fraudulent (0.17% fraud rate).  
Source: [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
