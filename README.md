# Fraud Detection MLOps Pipeline

A machine learning project that trains and benchmarks 12+ models on real credit card fraud data, selects the best model (XGBoost), and serves real-time predictions via a FastAPI REST API — packaged in Docker with full CI/CD.

---

## Project Status

| Step | Description | Status |
|------|-------------|--------|
| 1 | Project setup & environment | Done |
| 2 | Dataset download & EDA | Done |
| 3 | Data preprocessing pipeline | Done |
| 4 | MLflow experiment tracking setup | Done |
| 5 | Train & benchmark 12 models | Done |
| 6 | Model comparison & XGBoost tuning | Pending |
| 7 | FastAPI inference service | Pending |
| 8 | Tests | Pending |
| 9 | Docker containerization | Pending |
| 10 | CI/CD with GitHub Actions | Pending |
| 11 | Deploy on Render | Pending |
| 12 | README & demo polish | Pending |

---

## Project Structure

```
fraud-detection/
├── data/
│   ├── raw/
│   │   └── creditcard.csv          # Original Kaggle dataset (284,807 rows)
│   └── processed/
│       ├── X_train_raw.pkl         # Training features before SMOTE
│       ├── y_train_raw.pkl         # Training labels  before SMOTE
│       ├── X_train_smote.pkl       # Training features after SMOTE (balanced)
│       ├── y_train_smote.pkl       # Training labels  after SMOTE (balanced)
│       ├── X_test.pkl              # Test features (never touched by SMOTE)
│       └── y_test.pkl              # Test labels
│
├── notebooks/
│   └── 01_eda.ipynb                # Exploratory data analysis
│
├── src/
│   ├── __init__.py
│   ├── mlflow_setup.py             # MLflow init + log_model_run() helper
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocess.py           # Scaling, train/test split, SMOTE
│   ├── models/
│   │   ├── __init__.py
│   │   └── train_all_models.py     # (Step 5 — pending)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── compare_models.py       # (Step 6 — pending)
│   └── api/
│       ├── __init__.py
│       ├── main.py                 # (Step 7 — pending)
│       └── schemas.py              # (Step 7 — pending)
│
├── models/                          # 10 .pkl files total
│   ├── scaler.pkl                  # Fitted RobustScaler — Step 3 (not a model)
│   ├── logistic_regression.pkl     # Step 5
│   ├── decision_tree.pkl           # Step 5
│   ├── random_forest.pkl           # Step 5
│   ├── extra_trees.pkl             # Step 5
│   ├── adaboost.pkl                # Step 5
│   ├── gradient_boosting.pkl       # Step 5
│   ├── xgboost.pkl                 # Step 5
│   ├── lightgbm.pkl                # Step 5
│   └── catboost.pkl                # Step 5
│
├── mlruns/
│   └── mlflow.db                   # SQLite database for MLflow experiment tracking
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py       # (Step 8 — pending)
│   ├── test_model.py               # (Step 8 — pending)
│   └── test_api.py                 # (Step 8 — pending)
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml               # (Step 10 — pending)
│
├── Dockerfile                      # (Step 9 — pending)
├── docker-compose.yml              # (Step 9 — pending)
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| ML models | scikit-learn, XGBoost, LightGBM, CatBoost |
| Imbalance handling | SMOTE (imbalanced-learn) |
| Experiment tracking | MLflow (SQLite backend) |
| API | FastAPI + Uvicorn |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Deployment | Render (free tier) |

---

## Setup

```bash
# Clone the repo
git clone <repo-url>
cd fraud-detection

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# Install all dependencies
pip install -r requirements.txt
```

---

## Step 1 — Project Setup & Environment

### `requirements.txt`
Lists every Python package the project depends on, grouped by purpose. All versions use `>=` (minimum version) instead of `==` (exact version) so pip automatically picks pre-built binary wheels for Python 3.13 — avoids compilation errors on Windows.

| Group | Packages | Purpose |
|-------|---------|---------|
| Core data science | pandas, numpy, scikit-learn, scipy | Data loading, manipulation, ML algorithms |
| Visualization | matplotlib, seaborn | Charts and plots in the EDA notebook |
| Gradient boosting | xgboost, lightgbm, catboost | The three advanced boosting libraries we benchmark |
| Imbalance | imbalanced-learn | Provides SMOTE to fix the 578:1 class imbalance |
| Experiment tracking | mlflow | Records every model's metrics and artifacts |
| Web API | fastapi, uvicorn, pydantic | FastAPI is the framework, uvicorn runs it, pydantic validates input |
| Testing | pytest, pytest-cov, httpx | pytest runs tests, httpx lets us call the API in tests |
| Notebooks | jupyter, nbconvert | Run and execute `.ipynb` notebooks |
| Utilities | joblib, python-dotenv | joblib saves/loads `.pkl` files, dotenv loads env variables |

### `.gitignore`
Tells git which files and folders to never commit. Key exclusions:

| Excluded | Why |
|----------|-----|
| `data/raw/*.csv` | Dataset is 144 MB — too large for git |
| `data/processed/` | Generated files — reproducible by running `preprocess.py` |
| `models/*.pkl` | Large binary files — tracked via MLflow instead |
| `mlruns/` | Auto-generated experiment logs |
| `venv/` | Virtual environment — each developer creates their own |
| `.env` | Would expose secrets (API keys, passwords) |

### `src/__init__.py`, `src/data/__init__.py`, etc.
Empty files that tell Python "this folder is a package." Without them, `from src.data.preprocess import run` would fail with an import error. Every subfolder under `src/` and `tests/` has one.

### Folder structure
Created upfront so every future script has a predictable place to read from and write to — no hardcoded paths scattered across files.

---

## Step 2 — EDA Notebook

### `notebooks/01_eda.ipynb`

| Section | What it shows |
|---------|--------------|
| Data overview | Shape (284,807 × 31), dtypes, memory usage |
| Missing values | Zero missing values — dataset is clean |
| Class imbalance | 99.83% normal, 0.17% fraud — 578:1 ratio (pie chart + bar chart) |
| Amount analysis | Fraud transactions tend to be smaller; histogram + box plot by class |
| Time analysis | Normal transactions dip at night; fraud is spread evenly throughout the day |
| V1–V28 distributions | Side-by-side histograms — V4, V11, V12, V14, V17 show strongest class separation |
| Correlation heatmap | Full 31×31 feature correlation matrix |
| Fraud correlations | Ranked bar chart of features most correlated with fraud |
| Key findings | Summary + why SMOTE and AUC-ROC are the right choices for this dataset |

```bash
# Run interactively
jupyter notebook notebooks/01_eda.ipynb

# Run headlessly from terminal (executes all cells and saves output)
jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output 01_eda.ipynb --output-dir notebooks/
```

---

## Step 3 — Data Preprocessing Pipeline

### `src/data/preprocess.py`

| Stage | Detail |
|-------|--------|
| Load | Reads `data/raw/creditcard.csv` |
| Scale | `RobustScaler` on `Amount` and `Time` only — V1–V28 are already PCA-scaled. RobustScaler ignores outliers so large amounts don't skew the scaling |
| Split | 80/20 stratified split — preserves 0.17% fraud ratio in both train and test sets |
| SMOTE | Applied to training data only — fraud examples grow from 394 → 227,451 (balanced). Test data is never touched |
| Save | All splits written to `data/processed/`; fitted scaler written to `models/scaler.pkl` |

**Generated files:**
```
data/processed/
├── X_train_raw.pkl      # 227,845 rows — original imbalanced training features
├── y_train_raw.pkl      # 394 fraud cases out of 227,845
├── X_train_smote.pkl    # 454,902 rows — SMOTE-balanced training features
├── y_train_smote.pkl    # 227,451 fraud cases (perfectly balanced)
├── X_test.pkl           # 56,962 rows — test features (untouched)
└── y_test.pkl           # 98 fraud cases out of 56,962

models/
└── scaler.pkl           # Fitted RobustScaler — loaded by the API at inference time
```

```bash
python src/data/preprocess.py
```

---

## Step 5 — Train & Benchmark 9 Models

### `src/models/train_all_models.py`
Trains all 9 models one by one, evaluates each on the untouched test set, logs everything to MLflow, and prints a final ranked comparison table.

**Why train on SMOTE data but test on raw data?**
SMOTE is only used to help the model learn — the test set must stay original (imbalanced) to reflect real-world conditions where 99.83% of transactions are normal.

**The 9 models and what each one does:**

| # | Model | How it works |
|---|-------|-------------|
| 1 | Logistic Regression | Draws a straight decision boundary — simple but fast baseline |
| 2 | Decision Tree | Asks yes/no questions on features (e.g. "V14 < -2?") to reach a prediction |
| 3 | Random Forest | Builds 100 decision trees independently and lets them vote — reduces overfitting |
| 4 | Extra Trees | Like Random Forest but splits are chosen randomly — even faster, often similar accuracy |
| 5 | AdaBoost | Trains trees sequentially, each one focusing harder on the examples the previous tree got wrong |
| 6 | Gradient Boosting | Also sequential, but minimises a loss function mathematically at each step — more accurate than AdaBoost |
| 7 | XGBoost | Optimised Gradient Boosting with regularisation to prevent overfitting — usually the winner |
| 8 | LightGBM | Microsoft's gradient boosting — grows trees leaf-first instead of level-first, trains faster |
| 9 | CatBoost | Yandex's gradient boosting — handles ordered data well, no need to tune as much |

**Why AUC-ROC is the primary metric (not accuracy):**
A model that predicts "normal" for every transaction gets 99.83% accuracy but catches zero fraud. AUC-ROC measures how well the model ranks fraud above normal across all decision thresholds — a perfect model scores 1.0, random guessing scores 0.5.

**Files in `models/` after training (10 total):**
```
models/
├── scaler.pkl              # From Step 3 — RobustScaler fitted on Amount & Time
├── logistic_regression.pkl # Trained model #1
├── decision_tree.pkl       # Trained model #2
├── random_forest.pkl       # Trained model #3
├── extra_trees.pkl         # Trained model #4
├── adaboost.pkl            # Trained model #5
├── gradient_boosting.pkl   # Trained model #6
├── xgboost.pkl             # Trained model #7
├── lightgbm.pkl            # Trained model #8
└── catboost.pkl            # Trained model #9
```
`scaler.pkl` is not a model — it was created in Step 3 and lives here because the API needs it at inference time alongside the final model.
Each `.pkl` is the fitted model saved with joblib. They are gitignored (too large) but regenerated by re-running this script.

**MLflow logs per model:** hyperparameters + 5 metrics (`auc_roc`, `f1`, `precision`, `recall`, `accuracy`) + the model artifact itself.

```bash
python src/models/train_all_models.py
```

### Model Benchmark Results

| Rank | Model | AUC-ROC | F1 | Precision | Recall | Accuracy |
|------|-------|---------|----|-----------|----|---------|
| — | — | *(populated after training)* | | | | |

*(Full table with real scores added after Step 5 runs — view live in MLflow UI)*

```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

---

## Step 4 — MLflow Experiment Tracking

### `src/mlflow_setup.py`
Configures MLflow and provides a reusable logging helper so every model training script logs results in exactly the same way.

| Function | What it does |
|----------|-------------|
| `init_mlflow()` | Points MLflow at the SQLite database, creates the `fraud-detection-benchmark` experiment if it doesn't exist. Called once at the top of the training script |
| `log_model_run()` | Called once per model — opens an MLflow run, logs all params and metrics, saves the fitted model as an artifact, then closes the run. Prints a one-line summary to the console |

**What `log_model_run()` records for each model:**

| Logged item | Example |
|-------------|---------|
| Tag | `model_name = "XGBoost"` |
| Hyperparameters | `n_estimators=300`, `max_depth=6`, `learning_rate=0.1` |
| Metrics | `auc_roc=0.9821`, `f1=0.8734`, `precision=0.912`, `recall=0.847`, `accuracy=0.9993` |
| Model artifact | Fitted model saved and versioned inside MLflow |
| Model signature | Input schema (30 features) + output schema (0 or 1) — used by the API |

### `mlruns/mlflow.db`
SQLite database file auto-created when `init_mlflow()` runs for the first time. Stores all experiment metadata, run IDs, parameters, metrics, and artifact paths. Excluded from git via `.gitignore` — regenerated automatically when training runs.

```bash
# Open MLflow dashboard after Step 5 trains the models
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
# then open http://localhost:5000
```

---

## CI/CD Pipeline (.github/workflows/)

This folder is intentionally empty until **Step 10**. GitHub Actions automatically scans `.github/workflows/` for `.yml` files on every push. Once `ci-cd.yml` is added, every `git push` to `main` triggers:

```
git push → GitHub Actions →
  [1] Lint   — flake8 code style check
  [2] Test   — pytest (all tests must pass)
  [3] Build  — docker build
  [4] Push   — push image to Docker Hub
  [5] Deploy — trigger redeploy on Render
```

---

## Dataset

**Credit Card Fraud Detection** — [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

| Property | Value |
|----------|-------|
| Total transactions | 284,807 |
| Fraud cases | 492 (0.17%) |
| Features | 30 (V1–V28 are PCA-transformed, plus Amount and Time) |
| Target | `Class` — 0 = normal, 1 = fraud |
