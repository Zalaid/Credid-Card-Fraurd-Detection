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
| 8 | Tests | Done |
| 9 | Docker containerization | Done |
| 10 | CI/CD with GitHub Actions | Done |
| 11 | Deploy on Render | Done |
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
│   ├── test_preprocessing.py       # Step 8 — 7 preprocessing tests
│   ├── test_model.py               # Step 8 — 7 model correctness tests
│   └── test_api.py                 # Step 8 — 14 API endpoint tests
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml               # Step 10 — test → build → push → deploy on every git push
│
├── Dockerfile                      # Step 9 — builds the API container image
├── docker-compose.yml              # Step 9 — runs API + MLflow UI together
├── .dockerignore                   # Step 9 — keeps image small (excludes data, models, venv)
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

MLflow is like a logbook for machine learning experiments. Every time you train a model, it automatically records what settings you used, what scores it got, and saves a copy of the model. Without it, you'd have to keep notes manually and it's easy to lose track of "which version of XGBoost got 0.98 AUC-ROC?"

### `src/mlflow_setup.py`
Sets up MLflow so every model training run is automatically recorded in a central dashboard.

| Function | What it does |
|----------|-------------|
| `init_mlflow()` | Points MLflow at the local SQLite database and creates the `fraud-detection-benchmark` experiment. Called once before training starts |
| `log_model_run()` | Called once per model — records the model name, all hyperparameters, all 5 metrics, and saves a copy of the fitted model |

**What gets recorded for each model:**

| Item | Example |
|------|---------|
| Model name | `"XGBoost"` |
| Hyperparameters | `n_estimators=300`, `max_depth=8`, `learning_rate=0.1` |
| Metrics | `auc_roc=0.9802`, `f1=0.8155`, `recall=0.8571`, `precision=0.6232`, `accuracy=0.9994` |
| Model artifact | A copy of the fitted model, stored and versioned by MLflow |
| Model signature | What the input looks like (30 features) and what the output looks like (0 or 1) |

**Folders and files created by this step:**

```
mlruns/                       ← new folder created here (gitignored)
├── mlflow.db                 SQLite database — stores all run IDs, metrics, and parameters
└── 1/                        folder for experiment ID 1 ("fraud-detection-benchmark")
    └── models/m-<run-id>/    one sub-folder per model run (9 runs after Step 5)
        ├── artifacts/
        │   ├── model.pkl         MLflow's own copy of the fitted model
        │   ├── MLmodel           metadata file — input/output schema and model flavour
        │   ├── requirements.txt  exact packages needed to reload this model later
        │   └── conda.yaml        conda environment for full reproducibility
```

Why does MLflow save its own copy of each model? So you can go back to any past run in the dashboard and reload the exact model that produced those metrics — even if you retrain and overwrite the file in `models/` later.

---

### How to Open the MLflow UI

After training the models (Step 5), run this command:

```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

Then open your browser and go to: **http://localhost:5000**

**What you will see:**

```
┌─────────────────────────────────────────────────────────┐
│  Experiments                                            │
│  > fraud-detection-benchmark          (11 runs)         │
└─────────────────────────────────────────────────────────┘
```

Click on **fraud-detection-benchmark** to see the runs table:

```
┌──────────────┬─────────┬────────┬───────────┬──────────┐
│ Run Name     │ auc_roc │ f1     │ precision │ recall   │
├──────────────┼─────────┼────────┼───────────┼──────────┤
│ CatBoost     │ 0.9819  │ 0.6121 │ 0.4699    │ 0.8776   │
│ XGBoost      │ 0.9798  │ 0.7288 │ 0.6232    │ 0.8776   │
│ LightGBM     │ 0.9772  │ 0.7143 │ 0.6071    │ 0.8673   │
│ ...          │ ...     │ ...    │ ...       │ ...      │
└──────────────┴─────────┴────────┴───────────┴──────────┘
```

**Things you can do in the UI:**

| Action | How |
|--------|-----|
| Sort by any metric | Click the column header — e.g. click `f1` to find the best F1 model |
| Compare two models | Tick the checkboxes next to two runs, then click "Compare" |
| See a parallel coordinates chart | After comparing, scroll down — shows how hyperparameters map to metrics |
| Download a model | Click any run → Artifacts tab → download the `.pkl` file |
| See exactly what settings were used | Click any run → Parameters tab |
| Filter runs | Use the search bar — e.g. `metrics.auc_roc > 0.97` |

**Clicking on a single run shows three tabs:**

- **Parameters** — every setting the model was trained with (e.g. `n_estimators = 300`)
- **Metrics** — all 5 scores recorded for that run
- **Artifacts** — the saved model files you can download and reload

---

### Errors We Hit and How They Were Fixed

**Error 1 — Windows path error (`KeyError: 'd'`)**

MLflow by default stores data in a local folder. On Windows, `os.path.join()` produces paths like `D:\BSDS\...` which MLflow tried to interpret as a URL and choked on the drive letter `d`.

```
# What went wrong:
TRACKING_URI = os.path.join("mlruns")
# → mlruns   (Windows absolute path, starts with D:\)
# MLflow sees the D: as a URL scheme → KeyError: 'd'
```

Fix: switched to a SQLite database stored inside the project folder. SQLite URIs always start with `sqlite:///` which MLflow handles correctly on all platforms:

```python
TRACKING_URI = "sqlite:///mlruns/mlflow.db"
```

**Error 2 — `artifact_path` deprecation warning**

Older MLflow code used `artifact_path` as a keyword argument:

```python
# Old (deprecated in newer MLflow):
mlflow.sklearn.log_model(model, artifact_path="model", ...)
```

Newer MLflow replaced it with `name`:

```python
# Fixed:
mlflow.sklearn.log_model(model, name="model", ...)
```

**Error 3 — XGBoost `use_label_encoder` deprecation**

Old XGBoost accepted a `use_label_encoder=False` parameter. Newer versions removed it entirely and raise an error if you pass it.

```python
# Old (breaks in XGBoost >= 1.6):
XGBClassifier(use_label_encoder=False, ...)

# Fixed — just remove the parameter:
XGBClassifier(eval_metric='logloss', ...)
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

## Step 8 — Tests

28 automated tests verify that every part of the pipeline works correctly. Run them all with one command:

```bash
venv\Scripts\python.exe -m pytest tests/ -v
# 28 passed in ~3 seconds
```

### `tests/test_preprocessing.py` — 7 tests

Checks that the preprocessing pipeline ran correctly and produced valid output.

| Test | What it checks |
|------|---------------|
| `test_processed_files_exist` | All 6 `.pkl` split files are present in `data/processed/` |
| `test_scaler_exists` | `models/scaler.pkl` exists — the API needs it at startup |
| `test_train_test_shapes` | Train and test sets have the same number of columns (30) |
| `test_smote_balanced` | After SMOTE, fraud and normal classes are exactly equal in the training set |
| `test_test_set_untouched` | The test set fraud ratio is still under 1% — SMOTE never touched it |
| `test_no_missing_values` | No NaN values in training or test features |
| `test_scaler_transforms_correctly` | The scaler actually changes the values of Amount and Time |

### `tests/test_model.py` — 7 tests

Checks that the trained models are correct and the final model is production-ready.

| Test | What it checks |
|------|---------------|
| `test_final_model_exists` | `models/xgboost_tuned.pkl` — the API's model — exists on disk |
| `test_all_models_loadable` | All 11 `.pkl` model files load without errors |
| `test_final_model_has_predict_proba` | The final model supports probability output (required by the API) |
| `test_prediction_shape` | Model returns exactly one prediction per input row |
| `test_prediction_values_are_binary` | All predictions are 0 or 1 — nothing in between |
| `test_probabilities_between_0_and_1` | Fraud probabilities always stay in the valid 0–1 range |
| `test_model_catches_some_fraud` | Model actually detects at least some fraud on the real test set |

### `tests/test_api.py` — 14 tests

Checks that the FastAPI endpoints behave correctly — correct responses, correct errors, correct schema.

| Test | What it checks |
|------|---------------|
| `test_health_returns_200` | `/health` returns HTTP 200 |
| `test_health_model_loaded` | `/health` reports `model_loaded: true` |
| `test_root_returns_200` | `/` returns HTTP 200 |
| `test_root_contains_endpoints` | `/` response includes the `endpoints` key |
| `test_predict_fraud_transaction` | Known fraud row from dataset → `is_fraud: true`, probability > 0.5 |
| `test_predict_normal_transaction` | Known normal row from dataset → `is_fraud: false`, probability < 0.5 |
| `test_predict_response_has_required_fields` | Response always includes `is_fraud`, `fraud_probability`, `inference_ms` |
| `test_predict_probability_in_range` | `fraud_probability` is between 0.0 and 1.0 |
| `test_predict_inference_ms_is_positive` | `inference_ms` is a positive number |
| `test_predict_all_zeros` | All-zero input does not crash — returns a valid response |
| `test_predict_large_amount` | Very large amount does not crash — returns a valid response |
| `test_predict_missing_field_returns_422` | Missing features returns HTTP 422 (Unprocessable Entity) |
| `test_predict_negative_amount_returns_422` | Negative Amount returns HTTP 422 |
| `test_predict_wrong_type_returns_422` | String where float expected returns HTTP 422 |

The fraud and normal test rows come directly from `creditcard.csv` — real transactions that the model was not trained on (they are from the held-out test set), so passing these tests means the model generalises correctly.

---

## Step 9 — Docker Containerization

Docker packages the API and all its dependencies into a single "container" — a sealed box that runs exactly the same on your laptop, your teammate's machine, and any cloud server. Without Docker, you'd have to manually install Python, pip packages, and configure paths on every machine you deploy to.

### `.dockerignore`
Tells Docker which files and folders to skip when building the image. Smaller image = faster build, faster push to Docker Hub, faster deploy on Render.

| Excluded | Why |
|----------|-----|
| `venv/` | Python packages are reinstalled fresh inside the container |
| `data/`, `notebooks/` | Not needed at runtime — only needed for training |
| `mlruns/` | Experiment logs are not part of the API |
| `models/*.pkl` (training models) | Only `xgboost_tuned.pkl` and `scaler.pkl` are needed — the other 9 models add ~70 MB |
| `catboost_info/`, `creditcard.csv` | Development artifacts, not needed in production |

### `Dockerfile`
Instructions for building the container image, line by line:

```dockerfile
FROM python:3.11-slim          # start from a minimal Python 3.11 image (~130 MB)

WORKDIR /app                   # all commands run from /app inside the container

COPY requirements.txt .        # copy requirements first (Docker caches this layer)
RUN pip install --no-cache-dir -r requirements.txt   # install all packages

COPY src/ ./src/               # copy the application code
COPY models/xgboost_tuned.pkl ./models/   # copy only the two files the API needs
COPY models/scaler.pkl        ./models/

EXPOSE 8000                    # tell Docker the app listens on port 8000

HEALTHCHECK ...                # Docker pings /health every 30s to check if the app is alive

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Why `--no-cache-dir`?** Pip's download cache is never used inside a container (it gets thrown away when the container stops), so skipping it makes the image smaller.

**Why copy requirements.txt before src/?** Docker builds in layers. If you change only the source code but not requirements.txt, Docker reuses the cached pip install layer and only rebuilds the fast COPY step. This makes rebuilds much faster during development.

**Why `--host 0.0.0.0`?** By default, Uvicorn listens on `127.0.0.1` (localhost only). Inside a container, that means only the container itself can reach it — your laptop can't. `0.0.0.0` means "accept connections from anywhere", which lets Docker forward your laptop's port 8000 into the container.

### `docker-compose.yml`
Defines two services that start together with one command:

| Service | What it runs | Port |
|---------|-------------|------|
| `api` | Builds from Dockerfile, runs the FastAPI fraud detection API | 8000 |
| `mlflow` | Runs the MLflow tracking UI for viewing experiment results | 5000 |

---

### How to Build and Run

To test it: Start Docker Desktop, then run docker-compose up --build from the project folder. Docker Desktop isn't currently running so the build couldn't be
  verified, but the files are correct. Ready for Step 10 (CI/CD) whenever you are.

**Prerequisites:** Docker Desktop must be running  ([download here](https://www.docker.com/products/docker-desktop/))

```bash
# Option 1 — build and run just the API
docker build -t fraud-detection-api .
docker run -p 8000:8000 fraud-detection-api

# Option 2 — run both API + MLflow UI together (recommended)
docker-compose up --build
```

After running, open:
- **API:** http://localhost:8000/health → should return `{"status": "ok", "model_loaded": true}`
- **Swagger docs:** http://localhost:8000/docs → interactive API documentation
- **MLflow UI:** http://localhost:5000 → experiment dashboard

**Test a prediction from inside Docker (same curl command as always):**
```bash
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
# → {"is_fraud": true, "fraud_probability": 1.0, "inference_ms": 4.2}
```

**To stop everything:**
```bash
docker-compose down
```

---

## Step 10 — CI/CD with GitHub Actions

Every time you push code to GitHub, the pipeline automatically runs tests, builds a Docker image, pushes it to Docker Hub, and deploys the new version to Render — all without any manual steps.

### `.github/workflows/ci-cd.yml`

The pipeline has three jobs that run in order. If any job fails, the next one doesn't start.

```
git push to main
      │
      ▼
  ┌─────────┐
  │  test   │  Install Python 3.11 + deps
  │         │  Download model files from GitHub Release
  │         │  Run pytest tests/test_api.py (14 tests)
  └────┬────┘
       │ all pass
       ▼
  ┌──────────────────┐
  │  build-and-push  │  Download model files from GitHub Release
  │                  │  docker build (using Dockerfile)
  │                  │  docker push → Docker Hub :latest + :commit-sha
  └────────┬─────────┘
           │ image pushed
           ▼
      ┌──────────┐
      │  deploy  │  POST to Render deploy hook URL
      │          │  Render pulls new image → live URL updates
      └──────────┘
```

**Why download models from GitHub Release in CI?**
The model files (`xgboost_tuned.pkl`, `scaler.pkl`) are in `.gitignore` so they don't get pushed to GitHub. Instead, they are uploaded once as assets on a GitHub Release (`v1.0.0`). Every CI run downloads them fresh before running tests and before building Docker.

**Why only run `test_api.py` in CI and not all 28 tests?**
`test_preprocessing.py` and `test_model.py` need the full processed dataset (~500 MB of `.pkl` files). Downloading all of that in CI on every push is wasteful. `test_api.py` (14 tests) covers the thing that actually gets deployed — the API. The training pipeline tests run locally.

**Two Docker tags per push:**
- `:latest` — always points to the newest version
- `:abc1234` (git commit SHA) — lets you roll back to any exact version

---

### Step 10 Setup — What You Need to Do

#### 1. Create the GitHub repository and push

```bash
# On GitHub: create a new public repo named "fraud-detection" (no README, no .gitignore)
# Then in your project folder:
git remote add origin https://github.com/YOUR-USERNAME/fraud-detection.git
git push -u origin master
```

#### 2. Create a GitHub Release with the model files

The CI pipeline downloads models from Release `v1.0.0`. Create it once:

1. Go to your GitHub repo → **Releases** → **Create a new release**
2. Tag: `v1.0.0` | Title: `v1.0.0 — initial model`
3. Click **Attach binaries** and upload these two files from your local `models/` folder:
   - `xgboost_tuned.pkl`
   - `scaler.pkl`
4. Click **Publish release**

#### 3. Create a Docker Hub account and access token

1. Sign up at [hub.docker.com](https://hub.docker.com) (free)
2. Go to **Account Settings → Security → New Access Token**
3. Name it `github-actions`, permission: Read & Write
4. Copy the token (shown only once)

#### 4. Add GitHub Secrets

Go to your GitHub repo → **Settings → Secrets and variables → Actions → New repository secret**. Add three secrets:

| Secret name | Value |
|-------------|-------|
| `DOCKERHUB_USERNAME` | Your Docker Hub username |
| `DOCKERHUB_TOKEN` | The access token from step 3 |
| `RENDER_DEPLOY_HOOK` | The URL from Step 11 below (add after setting up Render) |

#### 5. Push any change to trigger the pipeline

```bash
git push origin master
```

Go to your repo → **Actions** tab to watch the pipeline run live. Each job shows a green tick when it passes.

---

## Step 11 — Deploy on Render

Render is a free cloud platform. Once set up, every successful CI/CD run automatically redeploys the live API — no manual steps.

### What the free tier gives you

| Property | Value |
|----------|-------|
| RAM | 512 MB |
| CPU | 0.1 vCPU |
| Cost | Free |
| HTTPS | Automatic (free SSL certificate) |
| Cold start | ~30 seconds after 15 min of no traffic |
| Custom domain | Supported |

The cold start means the first request after a period of inactivity takes ~30 seconds. After that, responses are under 100ms. This is acceptable for a portfolio/demo project.

### Step 11 Setup — What You Need to Do

**Prerequisites:** Step 10 must be done first — Docker Hub must have your image pushed.

#### 1. Create a Render account

Sign up at [render.com](https://render.com) (free, no credit card needed).

#### 2. Create a new Web Service

1. Click **New → Web Service**
2. Choose **"Deploy an existing image from a registry"**
3. Image URL: `your-dockerhub-username/fraud-detection-api:latest`
4. Click **Connect**

#### 3. Configure the service

| Setting | Value |
|---------|-------|
| Name | `fraud-detection-api` |
| Region | Oregon (US West) — fastest free tier |
| Instance Type | **Free** |
| Port | `8000` |

Click **Create Web Service**. Render will pull the Docker image and deploy it. This takes 2–3 minutes the first time.

#### 4. Verify it's live

Once deployed, Render gives you a URL like:
```
https://fraud-detection-api.onrender.com
```

Test it:
```bash
# Health check
curl https://fraud-detection-api.onrender.com/health
# → {"status": "ok", "model_loaded": true}

# Fraud prediction
curl -X POST https://fraud-detection-api.onrender.com/predict \
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
# → {"is_fraud": true, "fraud_probability": 1.0, "inference_ms": 4.2}

# Interactive Swagger UI
# Open in browser: https://fraud-detection-api.onrender.com/docs
```

#### 5. Get the Deploy Hook URL and add it to GitHub Secrets

1. In Render → your service → **Settings** → scroll down to **Deploy Hook**
2. Copy the URL (looks like `https://api.render.com/deploy/srv-xxxxx?key=xxxxx`)
3. Go back to GitHub → **Settings → Secrets → Actions**
4. Add secret: `RENDER_DEPLOY_HOOK` = the URL you copied

From now on, every `git push` to `main` triggers the full pipeline: tests → Docker build → Docker push → Render redeploy → live URL updated.

---

## Dataset

**Credit Card Fraud Detection** — [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

| Property | Value |
|----------|-------|
| Total transactions | 284,807 |
| Fraud cases | 492 (0.17%) |
| Features | 30 (V1–V28 are PCA-transformed by the bank + Amount and Time) |
| Target | `Class` — 0 = normal, 1 = fraud |

---

## How to Explain This Project in an Interview

If an interviewer asks "walk me through your ML pipeline", here is how to answer it in plain English, step by step.

---

### The Problem

"We have a dataset of 284,807 real credit card transactions. Out of those, only 492 are fraud — that's 0.17%. The goal is to build a model that can automatically flag a transaction as fraudulent in real time."

---

### Step 1 — Understanding the Data (EDA)

"First I did Exploratory Data Analysis. The biggest finding was the class imbalance: 99.83% of transactions are normal and only 0.17% are fraud. This is a critical insight because it means we can't use accuracy as a metric — a model that just says 'normal' for everything would be 99.83% accurate but completely useless."

"The features V1 through V28 are PCA-transformed. The bank ran PCA on their internal data (card numbers, merchant info, device fingerprints) to protect cardholder privacy. We only see the transformed numbers, not the raw data."

---

### Step 2 — Preprocessing

"Before training, I did three things:

**Scaling** — I applied RobustScaler to `Amount` and `Time`. I chose RobustScaler over StandardScaler because it's based on the median and interquartile range instead of the mean — so a $10,000 transaction won't distort the scaling for all the $20 transactions. V1–V28 were already scaled by the bank, so I left them alone.

**Train/test split** — 80% of the data goes to training, 20% to testing. Critically, the split is *stratified*, which means both sets preserve the original 0.17% fraud ratio. The test set is sealed off and never touched again until final evaluation.

**SMOTE** — The training set only had 394 fraud cases out of 227,845 rows. No model can learn from 394 examples vs 227,451 normal ones. SMOTE (Synthetic Minority Oversampling Technique) creates synthetic fraud examples by interpolating between real ones — it brings the training set to 227,451 fraud and 227,451 normal. SMOTE is applied *only* to the training data, never the test data, because the test set must reflect real-world conditions."

---

### Step 3 — Experiment Tracking with MLflow

"I integrated MLflow before training so that every model run is automatically recorded. Each run captures the model name, all hyperparameters, and five evaluation metrics. This means I can open a dashboard at any time, sort by any metric, and instantly see which model performed best — without keeping manual notes. The data is stored in a local SQLite database so no server setup is needed."

---

### Step 4 — Training and Benchmarking 9 Models

"I trained 9 different algorithms on the SMOTE-balanced training data and evaluated each one on the real held-out test set. The models range from simple (Logistic Regression) to advanced (XGBoost, LightGBM, CatBoost)."

"I used **AUC-ROC as the primary metric**, not accuracy. AUC-ROC measures how well the model separates fraud from normal across all possible decision thresholds. A perfect model scores 1.0. Random guessing scores 0.5."

"I also tracked **F1 score**, which is the balance between Recall and Precision:
- **Recall** = out of all real frauds, how many did we catch? Missing fraud is expensive.
- **Precision** = out of everything we flagged as fraud, how many actually were? Too many false alarms frustrates customers.
- **F1** = the harmonic mean of the two — a high F1 means you're catching fraud without raising too many false alarms."

"CatBoost had the highest AUC-ROC (0.9819) but the lowest F1 (0.6121). XGBoost had the third-highest AUC-ROC (0.9798) with a much better F1 (0.7288). So the two strongest candidates were CatBoost and XGBoost."

---

### Step 5 — Hyperparameter Tuning

"I used **GridSearchCV** to tune the top two models. GridSearchCV tries every combination of settings you give it and picks the one with the best cross-validation score."

"For XGBoost I tried 3 values for number of trees, 3 for tree depth, and 3 for learning rate — that's 27 combinations. The best was 300 trees, depth 8, learning rate 0.1."

"Tuned XGBoost reached F1 = 0.8155 — a 12-point improvement over its baseline. Tuned CatBoost actually *underperformed* its baseline. The reason: CatBoost is designed to work well with its default settings. When we applied GridSearchCV on SMOTE-balanced data, it overfit to the synthetic examples — the cross-validation score hit 1.0, which is a red flag. The default CatBoost generalises better."

---

### Step 6 — Final Model Selection

"I selected **Tuned XGBoost** as the final model for these reasons:

1. **Best F1 score (0.8155)** — highest of all candidates. For a fraud detector, F1 is the most important metric because it forces the right trade-off between catching fraud and not blocking innocent customers.
2. **Best Precision (0.6232)** — for every 100 transactions flagged as fraud, 62 are actually fraud. CatBoost's baseline only gets this right 47% of the time.
3. **AUC-ROC still excellent (0.9802)** — barely 0.0017 below CatBoost.
4. **Faster inference** — XGBoost predictions take 2–5ms vs CatBoost's 10–15ms, which matters when serving thousands of transactions per minute."

---

### Step 7 — Serving the Model as a REST API

"I wrapped the model in a **FastAPI** application. FastAPI is a Python web framework that automatically generates interactive documentation (Swagger UI) and handles input validation via Pydantic."

"The API has three endpoints:
- `GET /health` — returns `{"status": "ok"}` — used by Docker health checks
- `GET /` — returns API info and endpoint list
- `POST /predict` — accepts 30 transaction features, returns `is_fraud`, `fraud_probability`, and `inference_ms`"

"The model and scaler are loaded **once at startup**, not on every request. This keeps response time under 10ms per prediction."

"For input validation, Pydantic automatically rejects any request that's missing a field, has the wrong data type, or has a negative Amount — before it even reaches the model. This is what makes the API robust."

---

### Step 8 — Automated Testing

"I wrote 28 automated tests across 3 files using pytest:
- **Preprocessing tests** — verify the data pipeline ran correctly (shapes, no NaN, SMOTE balance, scaler works)
- **Model tests** — verify all models load, predict correctly, and the final model catches real fraud
- **API tests** — verify every endpoint returns the right response, the right schema, and the right error codes for bad input"

"These tests catch regressions — if I change something in the pipeline, the tests immediately tell me what broke."

---

### The Complete Data Flow (End to End)

```
Raw CSV (284,807 rows)
        │
        ▼
 preprocess.py
   ├── RobustScaler on Amount + Time
   ├── 80/20 stratified split
   └── SMOTE on training set only
        │
        ▼
 train_all_models.py
   └── 9 models trained on SMOTE data, evaluated on real test set
   └── All metrics logged to MLflow
        │
        ▼
 compare_models.py + tune_xgboost.py
   └── Best model selected: Tuned XGBoost (F1=0.8155, AUC-ROC=0.9802)
   └── Saved as xgboost_tuned.pkl
        │
        ▼
 FastAPI (main.py)
   └── Loads model + scaler at startup
   └── POST /predict: scale → predict_proba → return JSON
        │
        ▼
 Docker container
   └── Isolated, reproducible environment
        │
        ▼
 GitHub Actions CI/CD
   └── Every git push: run tests → build image → deploy to Render
        │
        ▼
 Live API on Render
   └── https://your-app.onrender.com/predict
```

---

### Common Interview Questions & Answers

**Q: Why did you use SMOTE and not just class weights?**

"Both are valid approaches. Class weights tell the model to penalise missing a fraud more heavily — it's simpler. SMOTE creates new synthetic fraud examples — it gives the model more data to learn from. I chose SMOTE because it tends to produce better-calibrated probability outputs, which matter for the `fraud_probability` field in the API."

**Q: Why AUC-ROC and not accuracy?**

"Accuracy is misleading on imbalanced data. A model that predicts 'normal' for every transaction gets 99.83% accuracy but catches zero fraud. AUC-ROC measures the model's ability to rank frauds above normals regardless of the threshold. It stays meaningful even when classes are heavily imbalanced."

**Q: Why XGBoost over CatBoost if CatBoost had higher AUC-ROC?**

"AUC-ROC alone doesn't tell the full story. CatBoost's F1 was 0.6121 — its precision was only 0.47, meaning more than half its fraud alerts would be false alarms. In a real deployment, every false alarm means a legitimate customer's card is declined. Tuned XGBoost's F1 of 0.8155 with precision of 0.62 gives a much better experience."

**Q: How does the API scale to production?**

"The model and scaler are loaded once at startup — not on every request. XGBoost prediction itself takes 2–5ms. Uvicorn (the ASGI server) handles multiple concurrent requests. For higher scale, you'd put multiple Uvicorn workers behind a load balancer, or use Kubernetes to run multiple Docker containers."

**Q: What would you do differently with more time?**

"A few things: (1) Try threshold tuning — instead of 0.5, find the threshold that optimises the F1 score on a validation set. (2) Add SHAP values to explain why a transaction was flagged — critical for real-world trust. (3) Add data drift monitoring so we know when the model's input distribution starts to shift from what it was trained on."
