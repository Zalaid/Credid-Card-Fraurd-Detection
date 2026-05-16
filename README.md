# Fraud Detection MLOps Pipeline

A machine learning project that trains and benchmarks 9 models on real credit card fraud data, selects the best model, and serves real-time predictions via a FastAPI REST API — packaged in Docker with full CI/CD.

---

## Project Status

| Step | Description | Status |
|------|-------------|--------|
| 1 | Project setup & environment | Done |
| 2 | Dataset download & EDA | Done |
| 3 | Data preprocessing pipeline | Done |
| 4 | MLflow experiment tracking setup | Done |
| 5 | Train & benchmark 9 models | Done |
| 6 | Model comparison & XGBoost tuning | Done |
| 7 | FastAPI inference service | Done |
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
│   │   ├── creditcard.csv          # Original Kaggle dataset (284,807 rows)
│   │   └── class_distribution.png  # Pie chart from EDA
│   └── processed/
│       ├── X_train_raw.pkl         # Training features before SMOTE
│       ├── y_train_raw.pkl         # Training labels  before SMOTE
│       ├── X_train_smote.pkl       # Training features after SMOTE (balanced)
│       ├── y_train_smote.pkl       # Training labels  after SMOTE (balanced)
│       ├── X_test.pkl              # Test features (never touched by SMOTE)
│       ├── y_test.pkl              # Test labels
│       └── charts/
│           ├── metric_comparison.png  # Bar chart comparing all models
│           └── roc_curves.png         # ROC curves for all models
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
│   │   ├── train_all_models.py     # Trains all 9 models
│   │   ├── tune_xgboost.py         # GridSearchCV tuning — produces final model
│   │   └── tune_catboost.py        # GridSearchCV tuning — kept for comparison
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── compare_models.py       # Scores all models, generates charts
│   └── api/
│       ├── __init__.py
│       ├── main.py                 # Step 7 — FastAPI app, /predict /health /
│       └── schemas.py              # Step 7 — input/output data shapes
│
├── models/
│   ├── scaler.pkl                  # Step 3 — RobustScaler (not a model)
│   ├── logistic_regression.pkl     # Step 5
│   ├── decision_tree.pkl           # Step 5
│   ├── random_forest.pkl           # Step 5
│   ├── extra_trees.pkl             # Step 5
│   ├── adaboost.pkl                # Step 5
│   ├── gradient_boosting.pkl       # Step 5
│   ├── xgboost.pkl                 # Step 5
│   ├── lightgbm.pkl                # Step 5
│   ├── catboost.pkl                # Step 5
│   ├── best_model.pkl              # Step 6 — baseline CatBoost copy (highest AUC-ROC)
│   ├── xgboost_tuned.pkl           # Step 6 — FINAL MODEL served by the API
│   └── catboost_final.pkl          # Step 6 — tuned CatBoost (kept for comparison)
│
├── mlruns/
│   ├── mlflow.db                   # SQLite database — all experiment runs & metrics
│   └── 1/                          # Experiment folder (ID=1)
│       └── models/m-<id>/          # One folder per model run
│           ├── artifacts/model.pkl # Model copy saved by MLflow
│           ├── artifacts/MLmodel   # MLflow metadata (input/output schema)
│           └── artifacts/requirements.txt
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
Lists every Python package the project needs, grouped by purpose. All versions use `>=` so pip automatically picks pre-built wheels for Python 3.13 — avoids build errors on Windows.

| Group | Packages | Purpose |
|-------|---------|---------|
| Core data science | pandas, numpy, scikit-learn, scipy | Data loading, manipulation, ML algorithms |
| Visualization | matplotlib, seaborn | Charts and plots in the EDA notebook |
| Gradient boosting | xgboost, lightgbm, catboost | The advanced boosting libraries we benchmark |
| Imbalance | imbalanced-learn | Provides SMOTE to fix the 578:1 class imbalance |
| Experiment tracking | mlflow | Records every model's metrics and artifacts |
| Web API | fastapi, uvicorn, pydantic | FastAPI is the framework, uvicorn runs it, pydantic validates input |
| Testing | pytest, pytest-cov, httpx | pytest runs tests, httpx lets us call the API in tests |
| Notebooks | jupyter, nbconvert | Run and execute `.ipynb` notebooks |
| Utilities | joblib, python-dotenv | joblib saves/loads `.pkl` files, dotenv loads env variables |

### `.gitignore`
Tells git which files and folders to never commit.

| Excluded | Why |
|----------|-----|
| `data/raw/*.csv` | Dataset is 144 MB — too large for git |
| `data/processed/` | Auto-generated — reproducible by running `preprocess.py` |
| `models/*.pkl` | Large binary files — tracked via MLflow instead |
| `mlruns/` | Auto-generated experiment logs |
| `venv/` | Virtual environment — each developer creates their own |
| `.env` | Would expose secrets like API keys |

### `src/__init__.py`, `src/data/__init__.py`, etc.
Empty files that tell Python "this folder is a package." Without them, importing code across folders (e.g. `from src.data.preprocess import run`) would fail. Every subfolder under `src/` and `tests/` has one.

---

## Step 2 — EDA Notebook

### `notebooks/01_eda.ipynb`
A Jupyter notebook that explores the raw dataset before any modelling. Run it to understand the data.

| Section | What it shows |
|---------|--------------|
| Data overview | Shape (284,807 × 31), dtypes, memory usage |
| Missing values | Zero missing values — dataset is clean, no fixes needed |
| Class imbalance | 99.83% normal, 0.17% fraud — 578:1 ratio |
| Amount analysis | Fraud transactions tend to be smaller on average |
| Time analysis | Normal transactions dip at night; fraud is spread throughout the day |
| V1–V28 distributions | V4, V11, V12, V14, V17 show the clearest difference between fraud and normal |
| Correlation heatmap | Full 31×31 feature correlation matrix |
| Fraud correlations | Which features are most linked to fraud |
| Key findings | Why SMOTE and AUC-ROC are the right choices for this dataset |

**Folders and files created by this step:**

`data/raw/class_distribution.png` — the pie chart + bar chart showing class imbalance, saved automatically when the notebook runs.

```bash
# Open and run interactively
jupyter notebook notebooks/01_eda.ipynb

# Run all cells from the terminal without opening a browser
jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output 01_eda.ipynb --output-dir notebooks/
```

---

## Step 3 — Data Preprocessing Pipeline

### `src/data/preprocess.py`
Reads the raw CSV, cleans and transforms the data, and saves everything ready for training.

| Stage | What happens |
|-------|-------------|
| Load | Reads `data/raw/creditcard.csv` |
| Scale | Applies `RobustScaler` to `Amount` and `Time` only. V1–V28 are already scaled by the bank. RobustScaler is chosen because it ignores extreme outliers — a $10,000 transaction won't distort the scaling |
| Split | 80% goes to training, 20% to testing. The split is stratified — meaning both sets keep the same 0.17% fraud ratio as the original |
| SMOTE | Fraud cases in the training set grow from 394 → 227,451 by creating synthetic examples. The test set is never touched — it stays real-world imbalanced |
| Save | All splits saved as `.pkl` files; scaler saved separately |

**Folders and files created by this step:**

```
data/processed/               ← new folder created here
├── X_train_raw.pkl           training features (before SMOTE) — 227,845 rows
├── y_train_raw.pkl           training labels  (before SMOTE) — only 394 fraud
├── X_train_smote.pkl         training features (after SMOTE)  — 454,902 rows, perfectly balanced
├── y_train_smote.pkl         training labels  (after SMOTE)   — 227,451 fraud, 227,451 normal
├── X_test.pkl                test features — 56,962 rows, never modified
└── y_test.pkl                test labels   — 98 fraud out of 56,962

models/
└── scaler.pkl                the fitted RobustScaler — the API loads this at startup to scale
                              incoming transaction data before passing it to the model
```

```bash
python src/data/preprocess.py
```

---

## Step 4 — MLflow Experiment Tracking

### `src/mlflow_setup.py`
Sets up MLflow so every model training run is automatically recorded in a central dashboard.

| Function | What it does |
|----------|-------------|
| `init_mlflow()` | Points MLflow at the local SQLite database and creates the `fraud-detection-benchmark` experiment. Called once before training starts |
| `log_model_run()` | Called once per model — records the model name, all hyperparameters, all 5 metrics, and saves a copy of the fitted model. Prints a one-line summary to the console |

**What gets recorded for each model:**

| Item | Example |
|------|---------|
| Model name | `"XGBoost"` |
| Hyperparameters | `n_estimators=300`, `max_depth=8`, `learning_rate=0.1` |
| Metrics | `auc_roc=0.9802`, `f1=0.8155`, `recall=0.8571` |
| Model artifact | A copy of the fitted model, stored and versioned by MLflow |
| Model signature | What the input looks like (30 features) and what the output looks like (0 or 1) |

**Folders and files created by this step:**

```
mlruns/                       ← new folder created here (gitignored)
├── mlflow.db                 SQLite database — stores all run IDs, metrics, and parameters
└── 1/                        folder for experiment ID 1 ("fraud-detection-benchmark")
    └── models/m-<run-id>/    one sub-folder is created per model run
        ├── artifacts/
        │   ├── model.pkl         MLflow's own copy of the fitted model
        │   ├── MLmodel           metadata file — records input/output schema and model flavour
        │   ├── requirements.txt  exact packages needed to reload this model later
        │   └── conda.yaml        conda environment for full reproducibility
```

Why does MLflow save its own copy of each model? So you can go back to any past run in the dashboard and reload the exact model that produced those metrics — even if you retrain and overwrite the file in `models/` later.

```bash
# View all experiment runs in the browser
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
# then open http://localhost:5000
```

---

## Step 5 — Train & Benchmark 9 Models

### `src/models/train_all_models.py`
Trains all 9 models one by one on the SMOTE-balanced training data, evaluates each on the real test data, logs results to MLflow, and prints a ranked comparison table.

**Why train on SMOTE data but test on raw data?**
SMOTE only helps the model learn — the test set must stay original (imbalanced) to reflect real life, where 99.83% of transactions are normal.

**Why AUC-ROC and not accuracy?**
A model that predicts "normal" for every transaction gets 99.83% accuracy but catches zero fraud. AUC-ROC measures how well a model separates fraud from normal across all possible decision thresholds. A perfect model = 1.0. Random guessing = 0.5.

**The 9 models:**

| # | Model | How it works |
|---|-------|-------------|
| 1 | Logistic Regression | Draws a straight decision boundary — fast and simple baseline |
| 2 | Decision Tree | Asks yes/no questions on feature values to reach a prediction |
| 3 | Random Forest | Builds 100 trees independently and takes a majority vote |
| 4 | Extra Trees | Like Random Forest but tree splits are chosen randomly — faster |
| 5 | AdaBoost | Trains trees one after another, each one focusing on the mistakes of the previous |
| 6 | Gradient Boosting | Same idea as AdaBoost but mathematically more precise |
| 7 | XGBoost | Optimised Gradient Boosting with regularisation — usually the strongest performer |
| 8 | LightGBM | Microsoft's version — grows trees leaf-first, trains faster on large datasets |
| 9 | CatBoost | Yandex's version — handles data ordering well, less tuning needed |

**Folders and files created by this step:**

```
models/                          ← 9 new files added here
├── logistic_regression.pkl      trained Logistic Regression — 1.5 KB
├── decision_tree.pkl            trained Decision Tree      — 44 KB
├── random_forest.pkl            trained Random Forest      — 16 MB (100 trees)
├── extra_trees.pkl              trained Extra Trees        — 52 MB (100 trees)
├── adaboost.pkl                 trained AdaBoost           — 64 KB
├── gradient_boosting.pkl        trained Gradient Boosting  — 253 KB
├── xgboost.pkl                  trained XGBoost (baseline) — 643 KB
├── lightgbm.pkl                 trained LightGBM           — 692 KB
└── catboost.pkl                 trained CatBoost           — 229 KB

mlruns/1/models/                 ← 9 new run folders added by MLflow automatically
```

All `.pkl` files are gitignored (too large) but regenerated by re-running this script.

```bash
python src/models/train_all_models.py
```

### Benchmark Results (ranked by AUC-ROC)

| Rank | Model | AUC-ROC | F1 | Precision | Recall | Accuracy |
|------|-------|---------|----|-----------|----|---------|
| 1 | CatBoost | 0.9819 | 0.6121 | 0.4699 | 0.8776 | 0.9981 |
| 2 | Gradient Boosting | 0.9809 | 0.2730 | 0.1606 | 0.9082 | 0.9917 |
| 3 | XGBoost | 0.9798 | 0.7288 | 0.6232 | 0.8776 | 0.9989 |
| 4 | Extra Trees | 0.9782 | 0.8737 | 0.9022 | 0.8469 | 0.9996 |
| 5 | LightGBM | 0.9772 | 0.7143 | 0.6071 | 0.8673 | 0.9988 |
| 6 | AdaBoost | 0.9739 | 0.1014 | 0.0537 | 0.9082 | 0.9723 |
| 7 | Random Forest | 0.9733 | 0.8497 | 0.8632 | 0.8367 | 0.9995 |
| 8 | Logistic Regression | 0.9712 | 0.1110 | 0.0591 | 0.9184 | 0.9747 |
| 9 | Decision Tree | 0.8704 | 0.1298 | 0.0704 | 0.8265 | 0.9809 |

---

## Step 6 — Model Comparison & Final Model Selection

### `src/evaluation/compare_models.py`
Loads all 9 trained models, scores them on the test set, generates visual comparison charts, and saves the best one.

| What it does | Detail |
|---|---|
| Score all models | Loads each `.pkl` and runs predictions on `X_test` |
| Bar chart | 4-panel chart — AUC-ROC, F1, Precision, Recall side by side for all 9 models |
| ROC curves | All 9 models on one plot — a curve closer to the top-left corner = better fraud detection |
| Save best | Copies the highest AUC-ROC model to `models/best_model.pkl` — which is CatBoost |

**What is a ROC curve?** It shows how many frauds you catch vs how many normal transactions you accidentally flag, as you adjust the decision threshold. The area under it (AUC-ROC) is the single most useful number for comparing models on imbalanced data.

**Folders and files created by this step:**

```
data/processed/charts/            ← new folder created here
├── metric_comparison.png         4-panel bar chart comparing all 9 models across AUC-ROC, F1,
│                                 Precision and Recall — shows which model wins on each metric
└── roc_curves.png                all 9 ROC curves on one plot — each line is one model,
                                  the diagonal dashed line = random guessing (AUC=0.5)

models/
└── best_model.pkl                copy of the highest AUC-ROC model — CatBoost (0.9819)
```

```bash
python src/evaluation/compare_models.py
```

---

### `src/models/tune_xgboost.py` and `src/models/tune_catboost.py`
After the benchmark, the top two models (CatBoost and XGBoost) were both tuned using GridSearchCV — trying 27 combinations each to find the best settings.

**What is GridSearchCV?** You give it a list of settings to try. It trains and tests every combination using cross-validation (splits the data 3 ways, trains on 2 parts, tests on 1, repeats) and picks the winner.

**XGBoost settings tried:**

| Setting | Values | What it controls |
|---------|--------|-----------------|
| `n_estimators` | 100, 200, 300 | Number of trees — more = more accurate but slower |
| `max_depth` | 4, 6, 8 | How deep each tree — deeper learns more patterns but can overfit |
| `learning_rate` | 0.05, 0.1, 0.2 | How much each tree corrects the previous — smaller = more stable |

**Best XGBoost combination:** `n_estimators=300`, `max_depth=8`, `learning_rate=0.1`

**CatBoost settings tried:**

| Setting | Values | What it controls |
|---------|--------|-----------------|
| `iterations` | 200, 300, 500 | Number of trees |
| `depth` | 4, 6, 8 | Tree depth |
| `learning_rate` | 0.05, 0.1, 0.2 | Learning step size |

**Best CatBoost combination:** `iterations=500`, `depth=8`, `learning_rate=0.1`

---

### Full Model Comparison & Final Selection

Here is every candidate compared side by side:

| Model | AUC-ROC | F1 | Precision | Recall |
|-------|---------|-----|-----------|--------|
| Baseline CatBoost | 0.9819 | 0.6121 | 0.4699 | 0.8776 |
| Baseline XGBoost | 0.9798 | 0.7288 | 0.6232 | 0.8776 |
| **Tuned XGBoost** ✅ | **0.9802** | **0.8155** | **0.6232** | **0.8571** |
| Tuned CatBoost | 0.9759 | 0.7119 | 0.6087 | 0.8571 |

**What each metric means in plain English:**
- **AUC-ROC** — Overall ability to tell fraud apart from normal. Higher = better at all thresholds
- **Recall** — Out of every 100 actual frauds, how many did we catch? Missing fraud is costly
- **Precision** — Out of every 100 transactions we flagged as fraud, how many were actually fraud? Low precision = too many false alarms
- **F1** — The balance between Recall and Precision. A high F1 means you're catching fraud without raising too many false alarms

**Why Tuned XGBoost was chosen as the final model:**

1. **Best F1 score (0.8155)** — highest of all four candidates. F1 is the most important metric for a fraud detector because it forces a balance: you want to catch fraud (Recall) but you also don't want to block legitimate transactions all day (Precision). CatBoost's baseline F1 of 0.6121 means it flags too many innocent transactions.

2. **Best Precision (0.6232)** — for every 100 transactions flagged as fraud, 62 are actually fraud. The baseline CatBoost only gets this right 47% of the time — meaning more than half its fraud alerts would be wrong.

3. **AUC-ROC still excellent (0.9802)** — only 0.0017 below the baseline CatBoost. The difference is negligible in practice.

4. **Why tuned CatBoost underperformed its baseline** — CatBoost is designed to already be near-optimal with default settings. Applying GridSearchCV on SMOTE-balanced data caused it to overfit the synthetic training examples (CV score hit 1.0 — a red flag). The default CatBoost generalises better.

5. **XGBoost is faster at inference** — serving predictions in real time, XGBoost loads and runs faster, which matters for keeping API response time under 100ms.

**Files created by this step:**

```
models/
├── xgboost_tuned.pkl    ← FINAL MODEL — served by the API
├── catboost_final.pkl   tuned CatBoost — kept for reference
└── best_model.pkl       baseline CatBoost copy — highest raw AUC-ROC

mlruns/1/models/         two new run folders added by MLflow (one per tuned model)
```

```bash
python src/models/tune_xgboost.py    # produces xgboost_tuned.pkl  ← final
python src/models/tune_catboost.py   # produces catboost_final.pkl  (comparison)
```

---

## Step 7 — FastAPI Inference Service

### `src/api/schemas.py`
Defines the exact shape of what the API accepts and returns using Pydantic. If the input is missing a field or has the wrong type, FastAPI automatically rejects it with a clear error before it even reaches the model.

| Schema | Fields | Purpose |
|--------|--------|---------|
| `TransactionInput` | Time, V1–V28, Amount (30 fields total) | What the caller must send |
| `PredictionOutput` | is_fraud, fraud_probability, inference_ms | What the API sends back |

---

### `src/api/main.py`
The FastAPI application. Three endpoints:

| Endpoint | Method | What it does |
|----------|--------|-------------|
| `/` | GET | Returns API name, version, model name, and list of endpoints |
| `/health` | GET | Returns `{"status": "ok"}` — used by Docker to check if the container is alive |
| `/predict` | POST | Accepts transaction features, returns fraud prediction |

**How the model is loaded:**
The model and scaler are loaded **once when the server starts**, not on every request. Loading takes ~200ms. Once loaded, each prediction only takes 2–5ms. This is what keeps response time under 100ms.

**What happens inside `/predict`:**
1. Receives the 30 feature values as JSON
2. Arranges them in the exact column order the model was trained on
3. Scales `Amount` and `Time` using the saved `scaler.pkl` (same transformation as training)
4. Runs `model.predict_proba()` to get the fraud probability
5. Returns result + how many milliseconds it took

---

### Understanding V1–V28 — How Inference Actually Works

**You never type V1–V28 yourself.** Here is what they are and how inference works in the real world vs testing:

**In real life (bank deployment):**
```
Customer swipes card
      ↓
Bank's internal system collects: card number, merchant, location,
device fingerprint, time, amount, purchase history...
      ↓
Bank runs PCA on all that data internally to protect privacy
      ↓
Produces: V1=-2.31, V2=1.95, V3=-1.61 ... V28=0.09, Amount=0.00, Time=406
      ↓
Sends these 30 numbers to our API → gets back {"is_fraud": true}
```

V1–V28 are not random — they are mathematically derived from real transaction data. They carry real fraud signal, just in a form that doesn't expose private card information.

**For testing (using rows from the dataset):**
Take any row from `creditcard.csv`, remove the `Class` column, and send the rest to the API. Two real examples tested and confirmed working:

```bash
# Fraud transaction (Class=1 in dataset) — API correctly returns is_fraud: true
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 406, "Amount": 0.0,
    "V1": -2.3122, "V2": 1.9519, "V3": -1.6097, "V4": 3.9979,
    "V5": -0.5222, "V6": -1.4265, "V7": -2.5374, "V8": 1.3914,
    "V9": -2.7700, "V10": -2.7722, "V11": 3.2020, "V12": -2.8992,
    "V13": -0.5950, "V14": -4.2895, "V15": 0.3898, "V16": -1.1407,
    "V17": -2.8300, "V18": -0.0168, "V19": 0.4165, "V20": 0.3269,
    "V21": 0.1474, "V22": -0.1703, "V23": 0.0359, "V24": -0.4118,
    "V25": 0.0714, "V26": 0.0719, "V27": 0.2127, "V28": 0.0952
  }'
# → {"is_fraud": true, "fraud_probability": 1.0, "inference_ms": 5.01}

# Normal transaction (Class=0 in dataset) — API correctly returns is_fraud: false
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 0, "Amount": 149.62,
    "V1": -1.3598, "V2": -0.0728, "V3": 2.5363, "V4": 1.3782,
    "V5": -0.3383, "V6": 0.4624, "V7": 0.2396, "V8": 0.0987,
    "V9": 0.3638, "V10": 0.0908, "V11": -0.5516, "V12": -0.6178,
    "V13": -0.9914, "V14": -0.3112, "V15": 1.4681, "V16": -0.4704,
    "V17": 0.2080, "V18": 0.0258, "V19": 0.4040, "V20": 0.2514,
    "V21": -0.0183, "V22": 0.2778, "V23": -0.1105, "V24": 0.0669,
    "V25": 0.1285, "V26": -0.1892, "V27": 0.1336, "V28": -0.0211
  }'
# → {"is_fraud": false, "fraud_probability": 0.0, "inference_ms": 2.2}
```

**Interactive docs (Swagger UI):**
When the API is running, open `http://localhost:8000/docs` in your browser. It shows every endpoint, lets you fill in the fields, and test with one click — no curl needed.

**To start the API:**
```bash
uvicorn src.api.main:app --reload
# then open http://localhost:8000
```

---

## CI/CD Pipeline (.github/workflows/)

This folder is empty until **Step 10**. GitHub Actions scans `.github/workflows/` for `.yml` files on every push. Once `ci-cd.yml` is added, every `git push` to `main` automatically triggers:

```
git push → GitHub Actions →
  [1] Lint   — checks code style with flake8
  [2] Test   — runs pytest, all tests must pass
  [3] Build  — builds the Docker image
  [4] Push   — pushes image to Docker Hub
  [5] Deploy — triggers a redeploy on Render (live URL updates automatically)
```

---

## Dataset

**Credit Card Fraud Detection** — [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

| Property | Value |
|----------|-------|
| Total transactions | 284,807 |
| Fraud cases | 492 (0.17%) |
| Features | 30 (V1–V28 are PCA-transformed by the bank + Amount and Time) |
| Target | `Class` — 0 = normal, 1 = fraud |
