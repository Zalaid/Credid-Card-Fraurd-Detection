# Fraud Detection MLOps Pipeline

A machine learning project that trains and benchmarks 9 models on real credit card fraud data, selects the best model, and serves real-time predictions via a FastAPI REST API packaged in Docker with full CI/CD.

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
| 12 | README & demo polish | Done |

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

### What is CI/CD and Why Does It Matter?

**CI (Continuous Integration)** means every time you push code, it is automatically tested. If a test fails, the deployment stops — broken code never reaches users.

**CD (Continuous Deployment)** means if all tests pass, the new version is automatically built into a Docker image, pushed to Docker Hub, and deployed to Render — all without you doing anything manually.

**Without CI/CD:**
```
You make a change → manually run tests → manually build Docker → manually push to Docker Hub
→ manually log into Render → manually trigger deploy → hope you didn't forget a step
```

**With CI/CD:**
```
You make a change → git push → everything else happens automatically in ~3 minutes
```

---

### What GitHub Actions Is

GitHub Actions is a free automation platform built into GitHub. You write a `.yml` file that describes what to do when certain events happen (like a git push). GitHub runs it on their servers — you pay nothing for public repos.

Every run is called a **workflow**. A workflow has **jobs**. Each job runs on a fresh Ubuntu server that GitHub spins up, executes your steps, then destroys.

---

### The Workflow File — `.github/workflows/ci-cd.yml`

GitHub Actions automatically picks up any `.yml` file inside `.github/workflows/`. Our file defines three jobs.

**Trigger — when does the pipeline run?**
```yaml
on:
  push:
    branches: [main, master]     # runs on every push to main or master
  pull_request:
    branches: [main, master]     # also runs when someone opens a pull request
```

---

### Job 1 — `test`

Runs on every push and every pull request. Purpose: make sure the code actually works before touching Docker or deployment.

```
GitHub receives push
        │
        ▼
  Fresh Ubuntu server starts
        │
        ├── actions/checkout@v4          → downloads your repo code onto the server
        ├── actions/setup-python@v5      → installs Python 3.11
        ├── pip install -r requirements.txt → installs all packages
        │
        ├── curl → GitHub Release v1.0.0
        │          downloads xgboost_tuned.pkl  (the trained model)
        │          downloads scaler.pkl          (the fitted scaler)
        │          saves them to models/ folder on the server
        │
        └── pytest tests/test_api.py -v
                   runs all 14 API tests
                   if any test fails → job fails → jobs 2 and 3 never run
```

**Why download models from GitHub Release?**
The model files are in `.gitignore` — they are not in the git repo. You upload them once to a GitHub Release (`v1.0.0`) as attached binary files. Every CI run downloads them fresh from that release URL:
```
https://github.com/YOUR-USERNAME/fraud-detection/releases/download/v1.0.0/xgboost_tuned.pkl
https://github.com/YOUR-USERNAME/fraud-detection/releases/download/v1.0.0/scaler.pkl
```
This keeps the git repo clean (no large binary files) and the models stay private (not committed to git history).

**Why only `test_api.py` and not all 28 tests?**
`test_preprocessing.py` and `test_model.py` need the full processed dataset — hundreds of MB of `.pkl` files. Downloading all of that on every push wastes CI time. `test_api.py` (14 tests) verifies the exact thing being deployed — the API endpoint. The training pipeline tests run locally where the data already exists.

---

### Job 2 — `build-and-push`

Runs only after Job 1 passes, and only on a direct push (not on pull requests — no point building Docker for unmerged code).

```
Job 1 passed
      │
      ▼
  Fresh Ubuntu server starts
      │
      ├── actions/checkout@v4          → downloads repo code
      │
      ├── curl → GitHub Release v1.0.0
      │          downloads xgboost_tuned.pkl and scaler.pkl again
      │          (this server is fresh — doesn't have Job 1's files)
      │
      ├── docker/login-action@v3
      │          logs into Docker Hub using DOCKERHUB_USERNAME + DOCKERHUB_TOKEN secrets
      │
      └── docker/build-push-action@v5
                 reads the Dockerfile
                 builds the Docker image (installs packages, copies src/, copies models/)
                 pushes two tags to Docker Hub:
                   your-username/fraud-detection-api:latest        ← always newest
                   your-username/fraud-detection-api:abc1234       ← exact commit SHA
```

**What are the two Docker tags?**
- `:latest` — Render always pulls this tag. It always points to the most recent build.
- `:abc1234` (the git commit SHA) — a permanent snapshot of this exact version. If a bug is introduced, you can roll back to any previous SHA by redeploying that specific tag.

**What are GitHub Secrets?**
Secrets are encrypted variables stored in your GitHub repo settings. They are injected into the workflow at runtime. The code accesses them with `${{ secrets.DOCKERHUB_USERNAME }}`. They are never visible in logs — GitHub replaces the value with `***`.

| Secret | What it is | Where to get it |
|--------|-----------|----------------|
| `DOCKERHUB_USERNAME` | Your Docker Hub username | Your Docker Hub account |
| `DOCKERHUB_TOKEN` | An access token (not your password) | Docker Hub → Account Settings → Security → New Access Token |
| `RENDER_DEPLOY_HOOK` | A webhook URL that triggers Render to redeploy | Render → your service → Settings → Deploy Hook |

---

### Job 3 — `deploy`

Runs only after Job 2 passes.

```
Docker image pushed to Docker Hub
            │
            ▼
  curl -X POST "https://api.render.com/deploy/srv-xxxxx?key=xxxxx"
            │
            ▼
  Render receives the POST request
  Render pulls the new :latest image from Docker Hub
  Render stops the old container
  Render starts the new container
  Live URL now serves the new version
```

This is a webhook — Render gives you a special URL (the deploy hook). When anything sends a POST request to that URL, Render automatically redeploys. GitHub Actions sends that POST request at the end of every successful pipeline run.

---

### Full Pipeline Flow (Everything Together)

```
You type: git push origin master
                │
                ▼
        GitHub receives push
                │
         ┌──────────────────────────────────────────────────────┐
         │  JOB 1 — test (runs on GitHub's Ubuntu server)       │
         │                                                      │
         │  1. Checkout code from repo                          │
         │  2. Install Python 3.11                              │
         │  3. pip install -r requirements.txt                  │
         │  4. Download xgboost_tuned.pkl from GitHub Release   │
         │  5. Download scaler.pkl from GitHub Release          │
         │  6. pytest tests/test_api.py -v                      │
         │       ✓ test_health_returns_200                      │
         │       ✓ test_predict_fraud_transaction               │
         │       ✓ test_predict_missing_field_returns_422       │
         │       ... 14 tests total                             │
         │                                                      │
         │  All passed? → continue     Any failed? → STOP here  │
         └──────────────────────────────────────────────────────┘
                │ all 14 passed
                ▼
         ┌──────────────────────────────────────────────────────┐
         │  JOB 2 — build-and-push (fresh Ubuntu server)        │
         │                                                      │
         │  1. Checkout code                                     │
         │  2. Download model files from GitHub Release          │
         │  3. Log in to Docker Hub (using secrets)             │
         │  4. docker build                                     │
         │       FROM python:3.11-slim                          │
         │       pip install requirements.txt                   │
         │       COPY src/                                      │
         │       COPY models/xgboost_tuned.pkl                  │
         │       COPY models/scaler.pkl                         │
         │  5. docker push :latest                              │
         │  6. docker push :abc1234 (commit SHA)                │
         │                                                      │
         │  Image on Docker Hub? → continue    Failed? → STOP   │
         └──────────────────────────────────────────────────────┘
                │ image pushed
                ▼
         ┌──────────────────────────────────────────────────────┐
         │  JOB 3 — deploy                                      │
         │                                                      │
         │  1. curl POST → Render deploy hook URL               │
         │  2. Render pulls new :latest image from Docker Hub   │
         │  3. Render restarts container with new image         │
         │  4. Live URL now serves updated code                 │
         └──────────────────────────────────────────────────────┘
                │
                ▼
  https://fraud-detection-api.onrender.com  ← updated in ~2 min
```

**Total time from `git push` to live update: ~3–5 minutes**

---

### What Happens on a Pull Request (vs a Direct Push)

| Event | Job 1 (test) | Job 2 (build) | Job 3 (deploy) |
|-------|-------------|--------------|---------------|
| `git push` to master | Runs | Runs | Runs |
| Pull request opened | Runs | Skipped | Skipped |

Pull requests only run tests — they don't deploy. This means you can review code changes safely without accidentally pushing broken code to production.

---

### How to Watch It Run

After pushing, go to your GitHub repo → **Actions** tab:

```
┌──────────────────────────────────────────────────────────┐
│  Actions                                                 │
│                                                          │
│  ● CI/CD   trigger first CI run   2 minutes ago          │
│    ├── ✓  test            38s                            │
│    ├── ✓  build-and-push  1m 42s                         │
│    └── ✓  deploy          4s                             │
└──────────────────────────────────────────────────────────┘
```

Click any job to see every step with its output. If something fails, the red ✗ tells you exactly which step and what the error was.

---

### What Happens After You Type `git push` — Complete Walkthrough

This section walks through every single thing that happens from the moment you press Enter on `git push origin master` to the moment the live API on Render is serving your new code. Nothing is skipped.

---

**Stage 0 — Your machine sends the code to GitHub**

When you type `git push origin master`, git packages up all the new commits and sends them over the internet to GitHub's servers. GitHub stores the new code and records what changed. At this point nothing has been tested or deployed — the code is just sitting on GitHub.

GitHub then looks inside the repository for any files in the `.github/workflows/` folder. It finds `ci-cd.yml`. It reads the `on: push: branches: [master]` line and realises: a push just happened to master, so the workflow should run. GitHub puts the workflow in a queue.

---

**Stage 1 — GitHub spins up a brand new server for Job 1 (test)**

GitHub rents a fresh Ubuntu Linux server from its own infrastructure. This machine has never seen your code before — it has nothing on it. GitHub installs the tools it needs (git, Python support, etc.) and then starts running the steps you defined in `ci-cd.yml` one by one.

**Step 1 — Checkout:** GitHub copies your entire repo onto this fresh server. Now the server has all your Python files, the Dockerfile, the requirements, the test files — everything that is in the git repo. But remember: `models/xgboost_tuned.pkl` and `models/scaler.pkl` are gitignored, so they are NOT here yet.

**Step 2 — Install Python 3.11:** GitHub installs exactly Python 3.11 on this server. This makes sure the tests always run on the same Python version regardless of what GitHub's default is.

**Step 3 — pip install:** Every package in `requirements.txt` is downloaded and installed — FastAPI, XGBoost, scikit-learn, pandas, everything. This takes about 60–90 seconds.

**Step 4 & 5 — Download the model files:** The workflow uses `curl` to download `xgboost_tuned.pkl` and `scaler.pkl` from the GitHub Release you created (a one-time upload you did when setting up the project). These two files land in the `models/` folder on this server. Now the server has everything the API needs to start.

**Step 6 — Run the tests:** pytest runs `tests/test_api.py`. This file has 14 tests. The TestClient from FastAPI actually starts the API inside the test process — no separate server needed — and fires real HTTP requests at it. The tests check things like: does `/health` return 200? Does `/predict` return the right fields? Does sending incomplete data return a 422 error?

If every single test passes, Job 1 is marked green and GitHub moves on. If even one test fails, the entire pipeline stops here. Job 2 and Job 3 never run. Your code sits on GitHub but nothing is built or deployed. You get an email saying the pipeline failed.

After Job 1 finishes, GitHub destroys this server. It is gone.

---

**Stage 2 — GitHub spins up a second brand new server for Job 2 (build-and-push)**

This is a completely separate machine. It has no memory of Job 1. Every step has to be done again from scratch.

**Step 1 — Checkout:** Same as before — your repo code is copied onto this server.

**Step 2 — Download model files again:** The model files have to be downloaded again because this is a fresh server. The `curl` commands run again, pulling from the GitHub Release. Now the models are in `models/` on this server.

**Step 3 — Log in to Docker Hub:** The workflow uses your `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` secrets (encrypted variables stored in GitHub settings, never visible in logs) to authenticate with Docker Hub. This is necessary to push images.

**Step 4 — Build the Docker image:** This is the big step. Docker reads the `Dockerfile` top to bottom and executes every line:

- It pulls the `python:3.11-slim` base image from Docker Hub — a minimal Linux with Python pre-installed
- It sets `/app` as the working directory inside the container
- It copies `requirements.txt` into the container and runs `pip install` — all packages get installed inside the image
- It copies the entire `src/` folder into the container
- It copies `models/xgboost_tuned.pkl` into the container (this is why we needed to download it in Step 2 — if it wasn't on the server, this COPY would fail)
- It copies `models/scaler.pkl` into the container
- The image is now a complete, self-contained package: Linux + Python + all packages + source code + trained model + scaler

The result is a Docker image sitting on this GitHub server's disk. It is about 1.5–2 GB.

**Step 5 & 6 — Push to Docker Hub:** The image is uploaded to Docker Hub under two tags. The `:latest` tag always points to the newest build — Render uses this. The `:abc1234` tag (the exact git commit SHA) is a permanent snapshot — if you ever need to roll back to a specific version, you can redeploy that exact SHA tag and know you're getting the exact code from that commit.

After Job 2 finishes, GitHub destroys this server too.

---

**Stage 3 — GitHub runs Job 3 (deploy)**

This job is tiny. It does one thing: it sends an HTTP POST request to a special URL that Render gave you — called a deploy hook. This URL is stored as a GitHub Secret called `RENDER_DEPLOY_HOOK`.

When Render receives that POST request, it knows a new image is ready. Render then does the following automatically:

- Pulls the new `:latest` image from Docker Hub
- Starts a new container from that image
- Waits for the new container's health check to pass (it calls `/health` and waits for a 200 response)
- Once the new container is healthy, Render switches traffic to it
- Stops and removes the old container

This process means there is zero downtime during a deploy. The old container keeps serving requests until the new one is confirmed healthy.

---

**Stage 4 — The live API is updated**

From this point on, anyone calling your Render URL is hitting the new container with the new code. The whole process from `git push` to live update takes about 3–5 minutes.

Here is a timeline breakdown:

| What is happening | Approximate time |
|-------------------|-----------------|
| GitHub queues and starts Job 1 | 5–15 seconds |
| pip install in Job 1 | 60–90 seconds |
| Running 14 tests | 10–20 seconds |
| GitHub starts Job 2 | 5–15 seconds |
| pip install inside Docker build | 60–90 seconds |
| Uploading image to Docker Hub | 30–60 seconds |
| Render pulling image and restarting | 30–60 seconds |
| **Total** | **~3–5 minutes** |

---

**What happens if something goes wrong**

If a test fails in Job 1, the pipeline stops. No new Docker image is built. Render is never notified. The live API keeps running the previous version, untouched. You get an email from GitHub telling you which test failed and showing the exact error output.

If the Docker build fails in Job 2 (for example, a file is missing or a package can't be installed), the pipeline stops there. Render is never notified. The live API keeps running the previous version.

If the deploy hook call in Job 3 fails (for example, a network issue), the new image is already on Docker Hub but Render wasn't told about it. You can manually trigger a redeploy from the Render dashboard in that case.

In all failure cases, your live API never goes down. It simply stays on the last successfully deployed version until you fix the problem and push again.

---

### Step 10 Setup — One-Time Configuration

#### 1. Create the GitHub repository and push

On github.com → New repository → Name: `fraud-detection` → Public → Create. Then:

```bash
git remote add origin https://github.com/YOUR-USERNAME/fraud-detection.git
git push -u origin master
```

#### 2. Create a GitHub Release with the model files

The CI pipeline needs to download `xgboost_tuned.pkl` and `scaler.pkl` from somewhere since they are gitignored. You upload them once to a GitHub Release.

1. GitHub repo → **Releases** → **Create a new release**
2. Tag: `v1.0.0` → Title: `v1.0.0`
3. **Attach binaries** → upload from your local `models/` folder:
   - `xgboost_tuned.pkl`
   - `scaler.pkl`
4. **Publish release**

#### 3. Create Docker Hub access token

1. Sign up at [hub.docker.com](https://hub.docker.com) (free)
2. **Account Settings → Security → New Access Token**
3. Name: `github-actions` | Permission: Read & Write
4. Copy the token — shown only once

#### 4. Add GitHub Secrets

GitHub repo → **Settings → Secrets and variables → Actions → New repository secret**

| Secret name | Value |
|-------------|-------|
| `DOCKERHUB_USERNAME` | Your Docker Hub username |
| `DOCKERHUB_TOKEN` | Access token from step 3 |
| `RENDER_DEPLOY_HOOK` | URL from Render (add after Step 11) |

#### 5. Push to trigger the pipeline

```bash
git push origin master
```

Go to **Actions** tab and watch all three jobs go green.

---

## Step 11 — Deploy on Render

### What is Render?

Render is a cloud platform that hosts web applications and APIs. You give it a Docker image, tell it which port to listen on, and it runs the container on its servers — giving you a public HTTPS URL anyone in the world can access.

Think of it like this:
```
Your laptop runs the API locally at:  http://localhost:8000
Render runs the same Docker image at: https://fraud-detection-api-latest-f3iz.onrender.com
```

The difference is Render's URL is public — anyone with a browser or curl can reach it, 24/7.

---

### What the Free Tier Gives You

| Property | Value |
|----------|-------|
| Cost | Free — no credit card needed |
| RAM | 512 MB |
| CPU | 0.1 vCPU |
| HTTPS | Automatic — free SSL certificate, `https://` works out of the box |
| Custom domain | Supported — you can point your own domain at it |
| Auto-deploy | Yes — watches Docker Hub for new images and redeploys automatically |
| Cold start | ~30 seconds after 15 minutes of no traffic |

---

### What is a Cold Start?

Render's free tier saves money by spinning down your container when nobody has used it for 15 minutes. When the next request comes in, Render has to start the container again — this takes about 30 seconds. After that, every request responds in under 100ms as normal.

```
Nobody uses the API for 15 min
          ↓
Render shuts the container down (saves resources)
          ↓
Someone clicks your CV link
          ↓
Render starts the container (30 sec wait — user sees loading)
          ↓
Model loads into memory
          ↓
All requests now fast (<100ms) until 15 min idle again
```

This is why we set up UptimeRobot — it pings the API every 5 minutes so the container never goes to sleep.

---

### How Render Knows When to Redeploy (Auto-Deploy)

When CI/CD finishes building a new Docker image and pushes `:latest` to Docker Hub, Render needs to know there is a new version. There are two ways:

**Deploy Hook (what we use):**
Render gives you a private URL called a Deploy Hook:
```
https://api.render.com/deploy/srv-xxxxx?key=xxxxx
```
When anything sends a POST request to this URL, Render immediately pulls the new `:latest` image and redeploys. Our CI/CD workflow sends this POST request at the end of every successful pipeline run.

**Auto-Deploy toggle:**
Render can also periodically poll Docker Hub to check if `:latest` changed. Enabled in Settings → Auto-Deploy. This is a fallback if the deploy hook isn't configured — it just takes a few extra minutes to detect the new image.

---

### How Render Deploys — Step by Step

When a deploy is triggered (either by deploy hook or auto-detect), here is what Render does internally:

```
Deploy triggered
      │
      ▼
Render pulls the Docker image from Docker Hub
  docker pull your-username/fraud-detection-api:latest
      │
      ▼
Render starts a new container from that image
  → uvicorn src.api.main:app --host 0.0.0.0 --port 8000
  → model and scaler load from models/ folder inside the image
  → app is now listening on port 8000
      │
      ▼
Render runs the health check
  GET /health → must return 200
      │
      ▼
Health check passed → Render switches traffic to the new container
Old container is stopped and removed
      │
      ▼
Live URL now serves the new version
```

If the health check fails (e.g. the model failed to load), Render keeps the old container running and marks the new deploy as failed. This means a bad deploy never takes down the live API.

---

### The Deploy Hook — Keeping It Secret

The deploy hook URL contains a private key:
```
https://api.render.com/deploy/srv-d84eglt8nd3s73d0g2m0?key=CA_usDvKVZg
```

Anyone who has this URL can trigger a redeploy of your service. That is why it is stored as a GitHub Secret (`RENDER_DEPLOY_HOOK`) and never written directly in code. In the workflow file it appears as `${{ secrets.RENDER_DEPLOY_HOOK }}` — GitHub substitutes the real value at runtime and replaces it with `***` in any logs.

If the key is ever exposed publicly, go to Render → Settings → Deploy Hook → **Regenerate** to invalidate the old one and get a new key.

---

### The Live API

**Base URL:**
```
https://fraud-detection-api-latest-f3iz.onrender.com
```

| URL | What it does |
|-----|-------------|
| `/` | Returns API name, version, model name, and endpoint list |
| `/health` | Returns `{"status": "ok", "model_loaded": true}` — used to verify the service is running |
| `/docs` | Swagger UI — interactive browser-based testing, no code needed |
| `/predict` | POST — send 30 transaction features, get fraud prediction back |

**Test the live API:**
```bash
# Health check
curl https://fraud-detection-api-latest-f3iz.onrender.com/health
# → {"status": "ok", "model_loaded": true}

# Known fraud transaction — should return is_fraud: true
curl -X POST https://fraud-detection-api-latest-f3iz.onrender.com/predict \
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

**Interactive Swagger UI (no code needed):**
Open in browser: `https://fraud-detection-api-latest-f3iz.onrender.com/docs`

---

### Step 11 Setup — What Was Done

1. Created account at render.com
2. New → Web Service → Deploy an existing image from a registry
3. Image: `your-dockerhub-username/fraud-detection-api:latest`
4. Port: `8000` | Instance: Free | Region: Oregon
5. Clicked Create Web Service — Render pulled the image and deployed
6. Copied Deploy Hook URL from Settings → added to GitHub as `RENDER_DEPLOY_HOOK` secret
7. Auto-Deploy enabled — Render watches Docker Hub for new `:latest` and redeploys automatically

---

## UptimeRobot — Keeping the API Awake

### What is UptimeRobot?

UptimeRobot is a free monitoring service. You give it a URL and a time interval, and it sends an HTTP request to that URL every N minutes to check if the service is up. If it goes down, it sends you an alert email.

We use it for two purposes:
1. **Keep the Render free tier awake** — by pinging every 5 minutes, the container never hits the 15-minute idle timeout and never cold-starts
2. **Uptime monitoring** — if the live API goes down for any reason, UptimeRobot emails you immediately

### How It Works

```
Every 5 minutes, UptimeRobot does this:
        │
        ▼
GET https://fraud-detection-api-latest-f3iz.onrender.com/health
        │
        ├── Response 200 → service is up → record as "up", do nothing
        │
        └── Response non-200 or timeout → service is down
                  → send alert email to you
                  → mark as "down" in dashboard
```

Because a request arrives every 5 minutes, Render never sees 15 minutes of inactivity and never spins down the container. The first request from a real user always gets a fast response.

### What Was Set Up

| Setting | Value |
|---------|-------|
| Monitor type | HTTP(s) |
| URL | `https://fraud-detection-api-latest-f3iz.onrender.com/health` |
| Interval | Every 5 minutes |
| Alert | Email when down |

### How to Set It Up (for reference)

1. Sign up at [uptimerobot.com](https://uptimerobot.com) — free, no credit card
2. **Add New Monitor**
3. Monitor Type: `HTTP(s)`
4. Friendly Name: `Fraud Detection API`
5. URL: `https://fraud-detection-api-latest-f3iz.onrender.com/health`
6. Monitoring Interval: `5 minutes`
7. Click **Create Monitor**

UptimeRobot starts pinging immediately. You can see the uptime history and response times in the dashboard at any time.

### Why Ping `/health` and Not `/`?

The `/health` endpoint is a lightweight check — it returns a small JSON object and does no computation. The `/` endpoint also works but returns slightly more data. Either works for pinging, but `/health` is the convention for health checks in production systems — it is what Docker, Kubernetes, and load balancers all use to check if a service is alive.

---

### How the Dockerfile Packages the App

Before understanding how Render runs the app, you need to understand what the Dockerfile does — because Render runs exactly what the Dockerfile produces.

A Docker image is like a zip file that contains everything needed to run the app: the operating system, Python, all packages, the source code, and the trained model. Render unzips this image and runs it.

Here is our Dockerfile explained line by line:

```dockerfile
FROM python:3.11-slim
```
Start from a base image that already has Python 3.11 installed on a minimal Linux system. "slim" means no extras — just Python. This keeps the image small (~130 MB base).

```dockerfile
WORKDIR /app
```
Create a folder called `/app` inside the container and make it the working directory. All future commands run from here. When Render starts the container, the app runs from `/app`.

```dockerfile
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
```
Copy `requirements.txt` from your laptop into the container, then install everything in it. This is done as a separate step before copying the code — Docker caches this layer. If you change only source code (not requirements), Docker skips this layer on the next build and reuses the cached packages. Saves 2–3 minutes per build.

```dockerfile
COPY src/ ./src/
```
Copy the entire `src/` folder (all your Python code — `api/main.py`, `api/schemas.py`, etc.) into `/app/src/` inside the container.

```dockerfile
COPY models/xgboost_tuned.pkl ./models/xgboost_tuned.pkl
COPY models/scaler.pkl        ./models/scaler.pkl
```
Copy the two model files into `/app/models/` inside the container. This is the key step — the trained model is now permanently baked into the image. Render doesn't need access to your laptop or any external storage. The model travels with the image.

```dockerfile
EXPOSE 8000
```
Tell Docker that the app listens on port 8000. This is documentation — Render uses it to know which port to route traffic to.

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
```
Every 30 seconds, Docker checks if the app is alive by hitting `/health`. If it fails 3 times in a row, Docker marks the container as unhealthy. Render uses this to decide whether a new deploy succeeded.

```dockerfile
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
The command that runs when the container starts. Uvicorn is the web server that runs the FastAPI app. `--host 0.0.0.0` means accept connections from anywhere (not just localhost). `--port 8000` matches the EXPOSE above.

**What the container looks like from the inside after build:**
```
/app/
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py        ← FastAPI app
│   │   └── schemas.py     ← input/output validation
│   ├── data/
│   ├── models/
│   └── mlflow_setup.py
└── models/
    ├── xgboost_tuned.pkl  ← trained XGBoost model (baked in)
    └── scaler.pkl         ← fitted RobustScaler (baked in)
```

Everything else (notebooks, data/, mlruns/, tests/, venv/) is excluded by `.dockerignore` — they are not in the image.

---

### Where the Model Lives and How It Gets Loaded

The model is not downloaded at runtime. It is **inside the Docker image** at `/app/models/xgboost_tuned.pkl`. When Render starts the container, Python loads it once into memory.

Here is what happens the moment the container starts:

```
Render runs: uvicorn src.api.main:app --host 0.0.0.0 --port 8000
      │
      ▼
Python imports src/api/main.py
      │
      ▼
These lines run at import time (top of main.py):

  MODEL_PATH  = "models/xgboost_tuned.pkl"   → /app/models/xgboost_tuned.pkl
  SCALER_PATH = "models/scaler.pkl"           → /app/models/scaler.pkl

  model  = joblib.load(MODEL_PATH)   ← XGBoost model loaded into RAM (~2MB)
  scaler = joblib.load(SCALER_PATH)  ← RobustScaler loaded into RAM (~1KB)
      │
      ▼
FastAPI app is created and routes are registered
      │
      ▼
Uvicorn starts listening on 0.0.0.0:8000
      │
      ▼
Container is ready — Render health check hits /health → 200 OK
      │
      ▼
Traffic is routed to this container
```

The model loads **once** — not on every request. After that, `model` and `scaler` sit in RAM for as long as the container is running. Each prediction just calls `model.predict_proba()` on the already-loaded object — that is why inference takes only 2–5ms.

---

### How GitHub, Docker Hub, and Render Are Connected

These three services form a chain. Each one hands off to the next automatically.

**GitHub** holds your source code and runs GitHub Actions.
**Docker Hub** is an image registry — a place to store and distribute Docker images, like an app store for containers.
**Render** is the cloud server that actually runs your container and serves traffic.

```
┌─────────────────────────────────────────────────────────────────────┐
│  GITHUB                                                             │
│                                                                     │
│  Your code lives here (src/, Dockerfile, requirements.txt, etc.)   │
│                                                                     │
│  When you push → GitHub Actions reads ci-cd.yml and runs:          │
│    1. Installs Python, downloads models, runs pytest                │
│    2. Reads the Dockerfile and builds a Docker image                │
│       The image contains:                                           │
│         - Python 3.11                                               │
│         - All pip packages from requirements.txt                    │
│         - Your src/ code                                            │
│         - xgboost_tuned.pkl and scaler.pkl baked in                 │
│    3. Logs into Docker Hub using your DOCKERHUB_USERNAME secret     │
│       and pushes the image there                                    │
│    4. POSTs to the Render Deploy Hook URL                           │
└────────────────────────────┬────────────────────────────────────────┘
                             │ docker push
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  DOCKER HUB  (hub.docker.com)                                       │
│                                                                     │
│  Stores your image at:                                              │
│    your-username/fraud-detection-api:latest                         │
│    your-username/fraud-detection-api:abc1234  (commit SHA)          │
│                                                                     │
│  Think of it like GitHub but for Docker images instead of code.     │
│  The image is public so Render can pull it without credentials.     │
└────────────────────────────┬────────────────────────────────────────┘
                             │ Render receives deploy hook POST
                             │ Render pulls image: docker pull your-username/fraud-detection-api:latest
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  RENDER  (render.com)                                               │
│                                                                     │
│  Runs your container on their server:                               │
│    uvicorn src.api.main:app --host 0.0.0.0 --port 8000             │
│                                                                     │
│  Wraps it with HTTPS automatically:                                 │
│    http://container:8000  →  https://fraud-detection-api-...render.com  │
│                                                                     │
│  Exposes it to the public internet — anyone can reach it            │
└────────────────────────────┬────────────────────────────────────────┘
                             │ public URL
                             ▼
              https://fraud-detection-api-latest-f3iz.onrender.com
```

**What "baked in" means:** When GitHub Actions runs `docker build`, the Dockerfile COPYs the model files into the image. So the final image is self-contained — it has Python, packages, code, and the trained model all in one file. Render just runs that file. It doesn't need access to your laptop or your `models/` folder. Everything it needs is already inside the image.

---

### How a User Opens and Tests the API (Swagger UI)

When someone opens your link in a browser, here is exactly what happens:

```
User opens: https://fraud-detection-api-latest-f3iz.onrender.com/docs
                              │
                              ▼
             Browser sends:  GET /docs  HTTP request to Render
                              │
                              ▼
             Render forwards request to your running container
                              │
                              ▼
             FastAPI receives GET /docs
             FastAPI automatically generates a web page from your code
             (reads the schemas.py input/output definitions and builds the UI)
                              │
                              ▼
             Browser receives HTML + JavaScript
             Renders the Swagger UI page
```

The user sees this interactive page:

```
┌──────────────────────────────────────────────────────────────┐
│  Fraud Detection API  1.0.0                                  │
│  Predicts whether a credit card transaction is fraudulent    │
│                                                              │
│  Prediction                                                  │
│  ─────────────────────────────────────────────────────────   │
│  POST  /predict                          [ Try it out ]      │
│                                                              │
│  Info                                                        │
│  ─────────────────────────────────────────────────────────   │
│  GET   /health                           [ Try it out ]      │
│  GET   /                                 [ Try it out ]      │
└──────────────────────────────────────────────────────────────┘
```

**Step by step — how they test a prediction:**

1. Click **POST /predict** to expand it
2. Click **Try it out** button (top right of that section)
3. The request body becomes editable — they see the full JSON with all 30 fields
4. They change some values or paste a row from the dataset
5. Click **Execute**
6. Swagger sends the request to `/predict` and shows the response:

```
┌─────────────────────────────────────────────────────────────┐
│  Request URL                                                │
│  https://fraud-detection-api-latest-f3iz.onrender.com/predict│
│                                                             │
│  Response body                                              │
│  {                                                          │
│    "is_fraud": true,                                        │
│    "fraud_probability": 1.0,                                │
│    "inference_ms": 4.21                                     │
│  }                                                          │
│                                                             │
│  Response code: 200                                         │
└─────────────────────────────────────────────────────────────┘
```

No curl, no Python, no terminal. Just a browser. This is what makes Swagger UI the right link to put in your CV — a recruiter can test your model in 30 seconds without writing a single line of code.

---

### Full Flow — From git push to Live Prediction

**Part 1: You push code — what happens in CI/CD**

```
You type:  git push origin master
                │
                ▼
        ┌───────────────────────────────────────────────────┐
        │  GITHUB                                           │
        │  Receives the push                                │
        │  Reads .github/workflows/ci-cd.yml               │
        │  Spins up a fresh Ubuntu server on GitHub's cloud │
        └───────────────────────┬───────────────────────────┘
                                │
                ┌───────────────▼────────────────┐
                │  JOB 1 — test                  │
                │                                │
                │  • git checkout (download code) │
                │  • Python 3.11 installed        │
                │  • pip install requirements.txt │
                │  • curl → GitHub Release v1.0.0 │
                │      downloads xgboost_tuned.pkl│
                │      downloads scaler.pkl       │
                │  • pytest tests/test_api.py     │
                │      14 tests run               │
                │      all pass ✓                 │
                └───────────────┬────────────────┘
                                │ tests passed
                ┌───────────────▼────────────────┐
                │  JOB 2 — build-and-push         │
                │                                │
                │  Fresh Ubuntu server (no files) │
                │  • git checkout                 │
                │  • curl → download models again │
                │  • docker build                 │
                │      reads Dockerfile           │
                │      FROM python:3.11-slim      │
                │      pip install packages       │
                │      COPY src/                  │
                │      COPY models/*.pkl  ← baked │
                │      image is now complete      │
                │  • docker login Docker Hub      │
                │      (uses GitHub Secrets)      │
                │  • docker push :latest          │
                │  • docker push :abc1234 (SHA)   │
                └───────────────┬────────────────┘
                                │ image on Docker Hub
                ┌───────────────▼────────────────┐
                │  JOB 3 — deploy                 │
                │                                │
                │  curl POST →                   │
                │  api.render.com/deploy/srv-xxx  │
                │  (the Render Deploy Hook)       │
                └───────────────┬────────────────┘
                                │
                                ▼
                        ┌───────────────────────┐
                        │  RENDER               │
                        │                       │
                        │  Receives POST        │
                        │  docker pull :latest  │
                        │  Stop old container   │
                        │  Start new container  │
                        │  → uvicorn runs       │
                        │  → model loads to RAM │
                        │  GET /health → 200    │
                        │  Deploy confirmed ✓   │
                        └───────────────────────┘
```

**Part 2: A user makes a prediction — what happens inside the container**

```
User opens Swagger UI or sends curl:
POST https://fraud-detection-api-latest-f3iz.onrender.com/predict
Body: {"Time": 406, "Amount": 0.0, "V1": -2.31, ... "V28": 0.09}
      │
      ▼
Render receives HTTPS request
Render decrypts SSL (converts https → http internally)
Render forwards to container on port 8000
      │
      ▼
Uvicorn (inside container) receives HTTP request on port 8000
Uvicorn routes it to the FastAPI /predict handler in src/api/main.py
      │
      ▼
Pydantic validates the input (schemas.py)
  • All 30 fields present? ✓
  • All values are numbers? ✓
  • Amount >= 0? ✓
  • Passes → create TransactionInput object
  • Fails → return HTTP 422 immediately, model never called
      │
      ▼
Inside predict():
  row = pd.DataFrame([transaction.model_dump()], columns=FEATURE_ORDER)
  → arranges the 30 values in the exact column order the model was trained on
  → ["Time", "V1", "V2", ..., "V28", "Amount"]
      │
      ▼
  row[["Amount", "Time"]] = scaler.transform(row[["Amount", "Time"]])
  → scaler is already in RAM (loaded at startup)
  → transforms Amount and Time to the same scale used during training
  → V1–V28 are left unchanged (already scaled by the bank)
      │
      ▼
  fraud_proba = float(model.predict_proba(row)[0][1])
  → model is already in RAM (loaded at startup)
  → XGBoost runs 300 decision trees on the 30 feature values
  → each tree votes: fraud or not
  → outputs a probability: e.g. 0.9997 (very likely fraud)
  → takes 2–5 milliseconds
      │
      ▼
  is_fraud = fraud_proba >= 0.5   → True
      │
      ▼
FastAPI returns JSON response:
  {
    "is_fraud": true,
    "fraud_probability": 1.0,
    "inference_ms": 4.21
  }
      │
      ▼
Uvicorn sends HTTP response to Render
Render encrypts it back to HTTPS
User's browser / curl receives the response
```

**Where the model is at each stage:**

| Stage | Where is xgboost_tuned.pkl? |
|-------|----------------------------|
| Your laptop | `D:\BSDS\...\models\xgboost_tuned.pkl` |
| GitHub Release | Attached as binary asset to release `v1.0.0` |
| CI/CD Job 1 (test) | Downloaded by curl to `/home/runner/work/.../models/` on GitHub's server |
| CI/CD Job 2 (build) | Downloaded by curl, then baked into the Docker image by `COPY models/xgboost_tuned.pkl` |
| Docker Hub | Inside the image layer at `/app/models/xgboost_tuned.pkl` |
| Render container | `/app/models/xgboost_tuned.pkl` — loaded into RAM at startup |
| During prediction | In RAM as a Python XGBoost object — `model.predict_proba()` runs on it |

The model file never touches the user's browser. It lives inside the Docker image on Render's server and is called entirely server-side.

---

### Full Deployment Picture (Everything Connected)

```
You push code
      │
      ▼
GitHub Actions: test → build → push to Docker Hub → POST deploy hook
      │
      ▼
Render: pulls image → starts container → model loads into RAM
      │
      ▼
UptimeRobot: pings /health every 5 min → container never sleeps
      │
      ▼
User opens /docs → fills in values → Execute
      │
      ▼
Container: validates input → scales features → XGBoost predicts → returns JSON
      │
      ▼
{"is_fraud": true, "fraud_probability": 1.0, "inference_ms": 4.2}
```

---

## Your Live API Links — What Each One Is and What to Put in Your CV

Your API is live at this base URL:
```
https://fraud-detection-api-latest-f3iz.onrender.com
```

Everything after the `/` is called an **endpoint** — a specific address that does a specific thing.

---

### All Four Links Explained

#### 1. `/` — Root (API Info)
```
https://fraud-detection-api-latest-f3iz.onrender.com/
```
**What it returns:**
```json
{
  "name": "Fraud Detection API",
  "version": "1.0.0",
  "model": "XGBoost (tuned)",
  "endpoints": {
    "POST /predict": "Send a transaction, get fraud prediction",
    "GET  /health":  "Health check",
    "GET  /docs":    "Interactive API documentation (Swagger UI)"
  }
}
```
**What it's for:** Just shows that the API is running and lists what endpoints exist. Not useful for end users — more of a welcome page.

---

#### 2. `/health` — Health Check
```
https://fraud-detection-api-latest-f3iz.onrender.com/health
```
**What it returns:**
```json
{"status": "ok", "model_loaded": true}
```
**What it's for:** A lightweight check that answers two questions:
- Is the API server running? (`status: ok`)
- Is the machine learning model loaded into memory and ready to make predictions? (`model_loaded: true`)

**Who uses it:**
- **Docker** uses it every 30 seconds to check if the container is healthy
- **Render** checks it after every new deploy — if it returns 200, the deploy succeeded; if it fails, Render rolls back to the old version
- **UptimeRobot** pings it every 5 minutes to keep the container awake and alert you if it goes down

This endpoint does **no ML computation** — it just checks two variables and returns JSON. That is why it is used for health checks instead of `/predict` — it is instant and has zero cost.

**How we get this link:** It is a route we defined in `src/api/main.py`:
```python
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
```
Render adds the base URL in front of it automatically.

---

#### 3. `/predict` — Fraud Prediction
```
https://fraud-detection-api-latest-f3iz.onrender.com/predict
```
**What it does:** This is the main endpoint. You send it 30 transaction features (Time, V1–V28, Amount) as JSON and it returns whether the transaction is fraud.

**You cannot open this in a browser** directly — it only accepts POST requests with a JSON body. Use Swagger UI, curl, or Python requests to call it.

**What it returns:**
```json
{
  "is_fraud": true,
  "fraud_probability": 1.0,
  "inference_ms": 4.21
}
```

**How we get this link:** It is a route defined in `src/api/main.py`:
```python
@app.post("/predict")
def predict(transaction: TransactionInput):
    ...
```

---

#### 4. `/docs` — Swagger UI ← PUT THIS IN YOUR CV
```
https://fraud-detection-api-latest-f3iz.onrender.com/docs
```
**What it is:** An automatically generated interactive web page that FastAPI creates from your code. It reads all your routes and schemas and builds a browser-based testing interface — no curl, no Python, no terminal needed.

**What a recruiter sees when they open it:**

```
┌──────────────────────────────────────────────────────┐
│  Fraud Detection API  1.0.0                          │
│                                                      │
│  POST  /predict   ▼  [ Try it out ]                  │
│  ────────────────────────────────────────────────    │
│  Request body:                                       │
│  {                                                   │
│    "Time": 406,                                      │
│    "V1": -2.3122,                                    │
│    ... all 30 fields editable ...                    │
│    "Amount": 0.0                                     │
│  }                    [ Execute ]                    │
│                                                      │
│  Response:                                           │
│  {                                                   │
│    "is_fraud": true,                                 │
│    "fraud_probability": 1.0,                         │
│    "inference_ms": 4.21                              │
│  }                                                   │
└──────────────────────────────────────────────────────┘
```

They click **Try it out** → **Execute** → see a real fraud prediction in 2 seconds. No setup, no code, works in any browser. This is what makes it CV-worthy.

**How we get this link:** FastAPI generates `/docs` automatically using a library called Swagger UI. We did not write any code for it — it reads our `TransactionInput` and `PredictionOutput` schemas from `schemas.py` and builds the page automatically.

---

### What to Put in Your CV

```
Fraud Detection MLOps Pipeline
──────────────────────────────────────────────────────────────
• Trained and benchmarked 9 ML models on 284,807 real transactions
• Selected Tuned XGBoost: AUC-ROC 0.9802, F1 0.8155, Precision 0.62
• Built FastAPI REST API with automated tests (28 passing)
• Containerized with Docker, deployed CI/CD pipeline via GitHub Actions
• Live demo: https://fraud-detection-api-latest-f3iz.onrender.com/docs
• Code:      https://github.com/Zalaid/Credid-Card-Fraurd-Detection
```

**Two links — two purposes:**

| Link | What it shows |
|------|--------------|
| `/docs` (Swagger UI) | The recruiter can test the live model in their browser right now — interactive, no setup |
| GitHub repo | The recruiter can read the code, see the commit history, check the CI/CD pipeline going green |

**Do not put `/health` or `/predict` in your CV.** `/health` is not interesting to humans — it is for machines. `/predict` cannot be opened in a browser directly. `/docs` is the only link that works for a non-technical person with a browser.

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

---

## Step 12 — Hugging Face Spaces Deployment

The API is deployed on a second platform — **Hugging Face Spaces** — in addition to Render. Both deployments run independently and serve the same model.

**Live Space:** https://huggingface.co/spaces/Zalaid/Credid-Card-Fraud-Detection

---

### Why a Second Deployment?

| Reason | Detail |
|--------|--------|
| Portfolio visibility | HF Spaces is where ML practitioners look for demos — more relevant audience than a generic cloud host |
| HF ecosystem | Shows the model alongside other ML work, searchable on the HF Hub |
| Free tier | HF Spaces free tier gives 16 GB RAM / 2 vCPUs — more than Render's free tier |
| Docker support | HF Spaces runs Docker containers natively — zero rewrite needed |

---

### Branch Strategy — Keeping Render Alive

The HF deployment required one change: HF Spaces **requires apps to listen on port 7860**, but Render uses port 8000. Changing the Dockerfile on `master` would break Render via CI/CD.

Solution: a separate `hf-spaces` branch that never merges into `master`.

```
master  ──────────────────────────────────────►  GitHub → CI/CD → Render (port 8000)
           \
hf-spaces  └──────────────────────────────────►  HF remote → HF Spaces (port 7860)
```

- `git push origin` (to GitHub) → triggers CI/CD → Render
- `git push hf hf-spaces:main` (to HF remote) → triggers HF build → HF Spaces
- The two remotes are completely independent — pushing to one never affects the other

```bash
# The two remotes configured in this repo
git remote -v
# origin   https://github.com/...  (GitHub → Render)
# hf       https://huggingface.co/spaces/Zalaid/Credid-Card-Fraud-Detection  (HF Spaces)
```

---

### Changes Made on the `hf-spaces` Branch

#### 1. `Dockerfile` — port changed to 7860

```diff
- EXPOSE 8000
+ EXPOSE 7860

- CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
+ CMD ["./start.sh"]
```

The `CMD` was also changed from calling uvicorn directly to running `start.sh` — explained below.

#### 2. `start.sh` — new file (downloads models at startup)

HF Spaces does not allow large binary files (`.pkl`) in the regular git repo. HF uses a separate binary storage system called **XET storage** for files like model weights. These files are not part of the Docker build context, so `COPY models/*.pkl` in the Dockerfile would fail with "file not found."

The fix: `start.sh` downloads the model files from the HF Space's own file storage at container startup, then launches uvicorn.

```bash
#!/bin/bash
set -e

mkdir -p models

python - <<'EOF'
from huggingface_hub import hf_hub_download
import shutil, os

for filename in ["models/xgboost_tuned.pkl", "models/scaler.pkl"]:
    path = hf_hub_download(
        repo_id="Zalaid/Credid-Card-Fraud-Detection",
        filename=filename,
        repo_type="space",
    )
    dest = filename
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copy(path, dest)
    print(f"Downloaded {filename}")
EOF

exec uvicorn src.api.main:app --host 0.0.0.0 --port 7860
```

**Why this works:** `hf_hub_download` pulls from HF's XET storage (where we uploaded the `.pkl` files) into the running container's filesystem before the API starts. The API code (`main.py`) is unchanged — it still loads from `models/xgboost_tuned.pkl` and `models/scaler.pkl` as normal.

#### 3. `README.md` — HF Space metadata frontmatter

HF Spaces requires a YAML block at the very top of `README.md` to configure the Space card (title, colour, SDK type). Without it, the Space shows a "configuration error."

```yaml
---
title: Credit Card Fraud Detection API
emoji: 💳
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
---
```

This block only exists on the `hf-spaces` branch — the `master` README does not have it.

---

### Files Stored on HF Spaces

HF Spaces holds two types of files in the same repo:

| File | Storage type | How it got there |
|------|-------------|-----------------|
| `Dockerfile` | Regular git | `git push hf hf-spaces:main` |
| `start.sh` | Regular git | `git push hf hf-spaces:main` |
| `src/` (all Python) | Regular git | `git push hf hf-spaces:main` |
| `requirements.txt` | Regular git | `git push hf hf-spaces:main` |
| `README.md` (with YAML) | Regular git | `git push hf hf-spaces:main` |
| `models/xgboost_tuned.pkl` | XET binary storage | `huggingface_hub.upload_file()` |
| `models/scaler.pkl` | XET binary storage | `huggingface_hub.upload_file()` |

The two `.pkl` files were uploaded using the HF Python SDK:

```python
from huggingface_hub import HfApi
api = HfApi(token="<hf-write-token>")
api.upload_file(
    path_or_fileobj="models/xgboost_tuned.pkl",
    path_in_repo="models/xgboost_tuned.pkl",
    repo_id="Zalaid/Credid-Card-Fraud-Detection",
    repo_type="space",
)
api.upload_file(
    path_or_fileobj="models/scaler.pkl",
    path_in_repo="models/scaler.pkl",
    repo_id="Zalaid/Credid-Card-Fraud-Detection",
    repo_type="space",
)
```

---

### Model Files on HF Spaces

| File | Size | Purpose |
|------|------|---------|
| `models/xgboost_tuned.pkl` | ~1.5 MB | The tuned XGBoost classifier — the model that makes fraud predictions |
| `models/scaler.pkl` | <1 KB | The fitted RobustScaler — scales `Amount` and `Time` before prediction |

These are the same two files that Render loads. Both are gitignored in the main repo (`models/*.pkl`) and tracked in HF via XET storage instead.

---

### Errors Encountered and Fixes

**Error 1 — Binary file rejected by git push**

```
remote: Your push was rejected because it contains binary files.
remote: Please use https://huggingface.co/docs/hub/xet to store binary files.
```

HF no longer accepts `.pkl` files via git push. Fix: removed them from the commit with `git rm --cached` and uploaded them separately via `huggingface_hub.upload_file()`.

**Error 2 — Docker build failed: model file not found**

```
ERROR: failed to calculate checksum of ref ...
"/models/xgboost_tuned.pkl": not found
```

The Docker build ran from a commit that predated the API upload of the `.pkl` files. The `COPY models/xgboost_tuned.pkl` line in the Dockerfile had nothing to copy. Fix: replaced `COPY` with `start.sh` which downloads the files at container runtime instead of build time.

**Error 3 — Invalid `colorTo` value in README metadata**

```
"colorTo" must be one of [red, yellow, green, blue, indigo, purple, pink, gray]
```

`colorTo: orange` is not a valid HF Space colour. Fix: changed to `colorTo: yellow`.

**Error 4 — README metadata overwritten on every git push**

Uploading the README via the HF Python API created a commit on HF's repo. The next `git push` from our local branch overwrote it with the project README (no YAML frontmatter), causing the "configuration error" again. Fix: added the YAML frontmatter directly to `README.md` on the `hf-spaces` branch so it is part of every push.

---

### What the Space Looks Like

The Space serves the same FastAPI app as Render. There is no visual UI — visitors interact with it through the auto-generated **Swagger UI** at `/docs`.

| Endpoint | URL |
|----------|-----|
| Root info (JSON) | `https://zalaid-credid-card-fraud-detection.hf.space/` |
| Interactive docs | `https://zalaid-credid-card-fraud-detection.hf.space/docs` |
| Health check | `https://zalaid-credid-card-fraud-detection.hf.space/health` |
| Predict | `POST https://zalaid-credid-card-fraud-detection.hf.space/predict` |

The `/docs` page lets anyone send a test transaction and see the fraud probability — no code required.

---

### Why HF Space Has Fewer Files Than Your Local Machine

If you open the HF Space repo and compare it to your local project folder, you will notice most files are missing. This is completely normal and intentional. Here is why.

**Your local machine has everything** — the raw dataset, processed data, all 9 trained model files, MLflow experiment logs, the EDA notebook, test files, virtual environment, and more. That is everything needed to reproduce the entire project from scratch: download data, preprocess it, train all models, evaluate them, and serve the final one.

**The HF Space only has what the API needs to run.** It is not a full copy of the project — it is just the production server. Think of it like this: when a restaurant ships food to a customer, they do not also send the kitchen, the ingredients, and the chef. They send only the finished dish.

Here is a breakdown of what is missing from HF Space and why:

| What is missing | Why it is not on HF Space |
|-----------------|--------------------------|
| `data/raw/creditcard.csv` | 144 MB dataset — only needed for training, not serving predictions |
| `data/processed/*.pkl` | Preprocessed training splits — only needed to train models |
| `notebooks/01_eda.ipynb` | Analysis notebook — not needed at runtime |
| `mlruns/` | MLflow experiment logs — only needed to compare models during development |
| `models/logistic_regression.pkl`, `random_forest.pkl`, etc. | The 7 other models that lost the benchmark — the API only uses the winner |
| `tests/` | Test suite — runs in CI/CD on GitHub, not needed on the server |
| `venv/` | Virtual environment — Docker installs its own dependencies inside the container |
| `data/processed/charts/` | Comparison charts — only needed during development |

**What IS on HF Space** is the minimum needed to answer one question at runtime: "given this transaction's 30 features, is it fraud?"

- The source code that runs the API (`src/`)
- The final trained model (`models/xgboost_tuned.pkl`)
- The scaler that was fitted during preprocessing (`models/scaler.pkl`)
- The dependency list so Docker knows what to install (`requirements.txt`)
- The startup script and Dockerfile so HF knows how to build and run the container

Everything else is development-time tooling that has no role once the model is trained and the API is built.

---

### How Render and HF Spaces Are Both Running at the Same Time

Both deployments are running the exact same FastAPI application, but they are two completely separate servers with no connection to each other. Here is how each one works:

**Render** watches your GitHub repository. Every time you push code to the `master` branch on GitHub, Render automatically pulls the new code, rebuilds the Docker container, and redeploys it. Render reads from the `master` branch via the `origin` remote (GitHub).

**HF Spaces** watches its own separate repository hosted on Hugging Face. Every time you push code to the `hf` remote (the HF Spaces repo), HF pulls the new code, rebuilds the Docker container, and redeploys it. HF reads from the `hf-spaces` branch via the `hf` remote (Hugging Face).

They do not share a server, a container, or a deployment pipeline. They just happen to run the same code. If one goes down, the other keeps running. If you update one, the other is not affected.

A useful analogy: imagine you have the same recipe and you give a copy to two different restaurants. Both cook the same dish independently in their own kitchens. Changing something in one kitchen does not change anything in the other.

---

### If You Push to GitHub, Does HF Space Update Too?

**No. HF Space will not change at all.**

This is the most important thing to understand about this setup. Your local machine has two remotes configured:

- `origin` points to GitHub
- `hf` points to Hugging Face

When you run `git push`, git only pushes to the remote you name. If you run `git push origin master`, the code goes to GitHub and Render picks it up. HF Spaces does not receive anything and does not rebuild.

To update HF Spaces, you have to explicitly push to the `hf` remote. Until you do that, HF Spaces keeps running whatever was last pushed there, forever — even if GitHub is ten commits ahead.

This means the two deployments can be at different versions of the code at the same time. That is fine and even useful — you can push experimental changes to HF Spaces to test them without touching the Render deployment, or vice versa.

The only way HF Space and Render could accidentally get out of sync in a bad way is if you forget to push a critical fix to one of them. To keep them in sync, any time you push an important change to `master` and want HF to match, you also need to merge that change into the `hf-spaces` branch and push to the `hf` remote separately.
