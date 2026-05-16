# Fraud Detection MLOps Pipeline

A machine learning project that trains and benchmarks 12+ models on real credit card fraud data, selects the best model, and serves real-time predictions via a REST API.

## Project Status

| Step | Description | Status |
|------|-------------|--------|
| 1 | Project setup & environment | Done |
| 2 | Dataset download & EDA | Done |
| 3 | Data preprocessing pipeline | Done |
| 4 | MLflow experiment tracking setup | Done |
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

## MLflow Experiment Tracking (Step 4)

`src/mlflow_setup.py` configures MLflow before any model is trained.

| What | Detail |
|------|--------|
| Backend | SQLite — stored at `mlruns/mlflow.db` (recommended over file-based tracking) |
| Experiment | `fraud-detection-benchmark` — all 12 model runs are grouped here |
| `init_mlflow()` | Sets tracking URI and creates the experiment. Called once at the top of the training script |
| `log_model_run()` | Helper used by every model — logs params, metrics, and the saved model artifact in one call |

**What gets logged per model run:**
- Hyperparameters (e.g. `n_estimators`, `max_depth`, `learning_rate`)
- Metrics: `auc_roc`, `f1`, `precision`, `recall`, `accuracy`
- The fitted model artifact (loadable directly from MLflow)
- Model signature (input/output schema)

**To open the MLflow dashboard:**
```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
# then open http://localhost:5000
```

## Data Preprocessing Pipeline (Step 3)

`src/data/preprocess.py` prepares the data identically for all 12 models.

| Stage | Detail |
|-------|--------|
| Load | Reads `data/raw/creditcard.csv` (284,807 rows) |
| Scale | `RobustScaler` on `Amount` and `Time` only — V1–V28 are already PCA-scaled. RobustScaler is used because it ignores outliers (large transaction amounts won't skew the scaling) |
| Split | 80/20 train/test split, stratified — preserves the 0.17% fraud ratio in both sets |
| SMOTE | Applied to training data only — synthetic fraud examples added until classes are balanced (394 fraud → 227,451 fraud). Test data is never touched |
| Save | Splits saved to `data/processed/`, scaler saved to `models/scaler.pkl` (see files below) |

**Files generated after running:**

```
data/processed/
├── X_train_raw.pkl      # Training features (before SMOTE) — 227,845 rows
├── y_train_raw.pkl      # Training labels  (before SMOTE) — 394 fraud cases
├── X_train_smote.pkl    # Training features (after SMOTE)  — 454,902 rows (balanced)
├── y_train_smote.pkl    # Training labels  (after SMOTE)   — 227,451 fraud cases
├── X_test.pkl           # Test features (never touched by SMOTE) — 56,962 rows
└── y_test.pkl           # Test labels                           — 98 fraud cases

models/
└── scaler.pkl           # Fitted RobustScaler — reused by the API at inference time
```

**Console output:**
```
Train : 227,845 rows  (fraud: 394)
Test  :  56,962 rows  (fraud: 98)
After SMOTE — normal: 227,451  fraud: 227,451
```

To run:
```bash
python src/data/preprocess.py
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
