# End-to-End MLOps Fraud Detection Pipeline — Complete Build Plan

## Context
You want to build a professional ML project that:
- Trains and compares 10+ machine learning models on a real fraud dataset
- Tracks all experiments using MLflow (a tool that logs every model's results)
- Selects XGBoost as the best model (we'll prove this with data)
- Serves predictions via a FastAPI web API
- Packages everything in Docker (so it runs identically anywhere)
- Deploys for free so anyone on the internet can use it
- Has CI/CD (automated testing + deployment on every code push)

**What you'll have at the end:** A live URL where someone sends transaction data and gets back "fraud" or "not fraud" in under 100ms.

---

## Key Concepts (Plain English)

| Term | What it means |
|------|--------------|
| **MLflow** | A dashboard that records every experiment — which model, which settings, what accuracy. Like a lab notebook for ML. |
| **XGBoost** | A powerful algorithm that builds many decision trees and combines them. Often wins ML competitions. |
| **LightGBM** | Similar to XGBoost but faster to train — we'll compare both. |
| **FastAPI** | A Python framework to create a web API (like a backend endpoint) in ~20 lines of code. |
| **Docker** | Packages your code + all its dependencies into a "container" so it runs the same everywhere. |
| **CI/CD** | GitHub Actions runs your tests and deploys automatically every time you push code. |
| **SMOTE** | A technique to fix imbalanced data (fraud is rare — only 0.17% of transactions). |

---

## Dataset
**Credit Card Fraud Detection** from Kaggle (free, public)
- 284,807 transactions, 492 fraudulent (0.17% fraud rate)
- 31 columns total (30 features + 1 target)
- Binary target: 0 = normal, 1 = fraud

### Dataset Columns Explained

| Column | What it means |
|--------|--------------|
| `Time` | Seconds elapsed since the first transaction in the dataset (just a timestamp) |
| `V1` to `V28` | **Hidden/scrambled features** — the bank ran PCA (a math technique) to hide real customer data for privacy. You don't know what they represent, and that's fine — the models learn patterns from them anyway. |
| `Amount` | The transaction amount in dollars (e.g., 149.62) |
| `Class` | **This is what we predict** — `0` = normal transaction, `1` = fraud |

**Why V1–V28 are hidden:** The bank couldn't share real data (card number, merchant, location) for privacy reasons, so they transformed those real features into V1–V28 using math (PCA). The fraud patterns are still there — just disguised. Our models don't need to "understand" what V1 means, they just find number patterns that predict fraud.

**Example row:**
```
Time=0, V1=-1.35, V2=-0.07, ..., V28=-0.02, Amount=149.62, Class=0 (normal)
```

---

## Project Folder Structure
```
fraud-detection/
├── data/                    # Raw and processed data
├── notebooks/               # EDA and experimentation
├── src/
│   ├── data/                # Data loading and preprocessing
│   ├── models/              # All 10+ model training scripts
│   ├── evaluation/          # Metrics and comparison
│   └── api/                 # FastAPI application
├── mlruns/                  # MLflow experiment logs (auto-generated)
├── models/                  # Saved trained models
├── tests/                   # Unit and integration tests
├── .github/workflows/       # CI/CD pipeline definitions
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Step-by-Step Build Plan

---

### STEP 1 — Project Setup & Environment
**Goal:** Get a clean, reproducible Python environment running.

**What to do:**
1. Create the folder structure above
2. Create a virtual environment (`python -m venv venv`)
3. Create `requirements.txt` with all dependencies:
   - `pandas`, `numpy`, `scikit-learn` — core data science
   - `xgboost`, `lightgbm`, `catboost` — gradient boosting libraries
   - `mlflow` — experiment tracking
   - `fastapi`, `uvicorn` — web API
   - `imbalanced-learn` — for SMOTE
   - `pytest` — for tests
   - `docker` — for containerization
4. Create `.gitignore` (ignore data files, model files, venv)
5. Initialize git repo and push to GitHub (public repo so CI/CD is free)

**Files created:** `requirements.txt`, `.gitignore`, `README.md`

---

### STEP 2 — Download & Understand the Dataset
**Goal:** Get the data and understand what we're working with.

**What to do:**
1. Download `creditcard.csv` from Kaggle (one-time manual step)
2. Create `notebooks/01_eda.ipynb` — Exploratory Data Analysis:
   - How many fraud vs normal transactions?
   - Distribution of transaction amounts
   - Correlation between features
   - Visualize the class imbalance (pie chart showing 99.83% normal, 0.17% fraud)
3. Key insight: **The data is extremely imbalanced** — if a model just predicts "not fraud" every time, it's 99.83% accurate but useless. We need special metrics.

**Files created:** `notebooks/01_eda.ipynb`

---

### STEP 3 — Data Preprocessing Pipeline
**Goal:** Clean and prepare data so all 10+ models get identical input.

**What to do:**
Create `src/data/preprocess.py`:
1. **Load data** — read the CSV
2. **Feature scaling** — normalize `Amount` and `Time` (V1-V28 are already scaled)
3. **Train/test split** — 80% training, 20% testing (stratified to preserve fraud ratio)
4. **SMOTE** — oversample the minority (fraud) class in training data only
   - SMOTE creates synthetic fraud examples so models can learn the pattern
   - Applied ONLY to training data, never test data
5. **Save preprocessed splits** to `data/processed/`

**Why SMOTE matters:** Without it, models see 99.83% normal examples and learn to just predict "normal" always. SMOTE balances the training set.

**Files created:** `src/data/preprocess.py`

---

### STEP 4 — MLflow Setup
**Goal:** Create the experiment tracking dashboard before training any models.

**What MLflow does:** Every time we train a model, MLflow automatically records:
- Which algorithm was used
- What hyperparameters (settings) were used
- Accuracy, precision, recall, F1, AUC-ROC scores
- How long training took
- The saved model file

**What to do:**
Create `src/mlflow_setup.py`:
1. Configure MLflow tracking URI (local `mlruns/` folder)
2. Create experiment named `"fraud-detection-benchmark"`
3. Write a helper `log_model_run()` function that any model script can call

**To view the dashboard:** Run `mlflow ui` and open `http://localhost:5000` in browser — you'll see all experiments in a table.

**Files created:** `src/mlflow_setup.py`

---

### STEP 5 — Train & Benchmark 10+ Models
**Goal:** Train all models, log every result to MLflow, and find the best one.

**The 12 Models We'll Train:**

| # | Model | Plain English |
|---|-------|---------------|
| 1 | Logistic Regression | Draws a straight line to separate fraud/normal |
| 2 | Decision Tree | Series of yes/no questions on features |
| 3 | Random Forest | 100+ decision trees voting together |
| 5 | AdaBoost | Focuses on examples previous models got wrong |
| 6 | Gradient Boosting | Builds trees one at a time, each fixing previous errors |
| 7 | **XGBoost** | Optimized Gradient Boosting — faster + regularized |
| 8 | **LightGBM** | Microsoft's fast gradient boosting |
| 10 | SVM | Finds the widest margin between fraud/normal |
| 11 | K-Nearest Neighbors | Classifies by looking at similar transactions |
| 12 | Neural Network (MLP) | Multi-layer perceptron — simple deep learning |

**What to do:**
Create `src/models/train_all_models.py`:
1. For each model, inside an `mlflow.start_run()` block:
   - Train on SMOTE-balanced training data
   - Predict on original (unbalanced) test data
   - Log metrics: AUC-ROC, F1, Precision, Recall, Accuracy
   - Log the model artifact (saved model file)
2. Print a comparison table at the end

**Why AUC-ROC over accuracy?**
- Accuracy: "99.83% — great!" (but it just predicts normal every time)
- AUC-ROC: Measures how well the model separates fraud from normal at all thresholds — the real metric

**Files created:** `src/models/train_all_models.py`

---

### STEP 6 — Model Comparison & Selection
**Goal:** Formally select XGBoost and tune it.

**What to do:**
Create `src/evaluation/compare_models.py`:
1. Query MLflow API to get all run results
2. Rank by AUC-ROC score
3. Generate comparison charts (bar charts, ROC curves)
4. Save the winning model to `models/best_model.pkl`

Create `src/models/tune_xgboost.py`:
1. Use `GridSearchCV` or `Optuna` to find best XGBoost hyperparameters:
   - `n_estimators` (number of trees)
   - `max_depth` (how deep each tree)
   - `learning_rate` (how fast it learns)
   - `scale_pos_weight` (handles class imbalance natively)
2. Log the tuned run to MLflow
3. Save final model to `models/xgboost_final.pkl`

**Expected result:** XGBoost ~95% fraud recall, LightGBM ~92-93% — 2.3% gap matches the resume bullet.

**Files created:** `src/evaluation/compare_models.py`, `src/models/tune_xgboost.py`

---

### STEP 7 — FastAPI Inference Service
**Goal:** Create a REST API that accepts transaction data and returns fraud prediction.

**What to do:**
Create `src/api/main.py`:

1. **Load model** at startup (once, not per request — key for sub-100ms latency)
2. **Define input schema** using Pydantic — 30 features (V1-V28, Amount, Time)
3. **POST /predict** endpoint:
   - Input: JSON with transaction features
   - Output: `{"is_fraud": true, "fraud_probability": 0.94, "inference_ms": 12}`
4. **GET /health** endpoint — for Docker health checks
5. **GET /** — simple info page

**Example request:**
```json
POST /predict
{"V1": -1.35, "V2": -0.07, ..., "Amount": 149.62, "Time": 0}
```

**Example response:**
```json
{"is_fraud": true, "fraud_probability": 0.94, "inference_ms": 8}
```

**Files created:** `src/api/main.py`, `src/api/schemas.py`

---

### STEP 8 — Write Tests
**Goal:** Automated tests so CI/CD can verify the code works before deploying.

**What to do:**
Create tests in `tests/`:
1. `test_preprocessing.py` — test data loading and SMOTE
2. `test_model.py` — test model loads, test prediction shape/type
3. `test_api.py` — test API endpoints with mock requests (using FastAPI's TestClient)
   - Test /health returns 200
   - Test /predict returns correct schema
   - Test edge cases (all zeros, very large amounts)

**Files created:** `tests/test_preprocessing.py`, `tests/test_model.py`, `tests/test_api.py`

---

### STEP 9 — Docker Containerization
**Goal:** Package the API so it runs identically on any machine or cloud server.

**What to do:**
Create `Dockerfile`:
```
FROM python:3.11-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ ./src/
COPY models/xgboost_final.pkl ./models/
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Create `docker-compose.yml`:
- Service 1: `api` — the FastAPI app
- Service 2: `mlflow` — the MLflow tracking UI (for local development)

**What to do:**
1. Build: `docker build -t fraud-detection-api .`
2. Run: `docker run -p 8000:8000 fraud-detection-api`
3. Test: open `http://localhost:8000/health`

**Files created:** `Dockerfile`, `docker-compose.yml`

---

### STEP 10 — CI/CD with GitHub Actions
**Goal:** Every time you push code to GitHub, it automatically tests → builds → deploys.

**Pipeline Stages:**

```
git push → GitHub Actions triggers →
  [1] LINT: check code style (flake8)
  [2] TEST: run pytest (all tests must pass)
  [3] BUILD: build Docker image
  [4] PUSH: push image to Docker Hub (free registry)
  [5] DEPLOY: trigger redeploy on Render/Railway
```

**What to do:**
Create `.github/workflows/ci-cd.yml`:
- Trigger: on push to `main` branch
- Jobs: lint, test, build, push, deploy (in order)
- Secrets in GitHub: `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`, `RENDER_DEPLOY_HOOK`

**Time savings:**
- Manual deployment: 4 hours (build, test, push, configure, deploy manually)
- With CI/CD: 45 minutes (automated — you just push code)
- **91% reduction** — matches the resume bullet

**Files created:** `.github/workflows/ci-cd.yml`

---

### STEP 11 — Free Deployment (Render)
**Goal:** Live public URL so anyone can use the API.

**Why Render (free tier):**
- 512 MB RAM, 0.1 CPU
- Deploys directly from GitHub or Docker Hub
- Auto-deploys when CI/CD pushes new image
- Free SSL certificate (HTTPS)
- Cold start: ~30 seconds after 15 min idle (acceptable for demo)

**What to do:**
1. Create account at render.com
2. Create "Web Service" → connect Docker Hub image
3. Set environment variables (model path etc.)
4. Get public URL: `https://fraud-detection-api.onrender.com`
5. Add deploy hook URL to GitHub secrets (triggers Step 10's deploy stage)

**Alternative free options:**
- Railway.app — slightly faster cold starts
- Hugging Face Spaces — best for ML demos with a UI

---

### STEP 12 — README & Demo
**Goal:** Make it presentable for portfolio/GitHub.

**What to do:**
Create detailed `README.md` with:
1. Project description and architecture diagram
2. Model benchmark table (all 12 models + scores from MLflow)
3. How to run locally (3 commands)
4. Live demo URL
5. API documentation (request/response examples)
6. CI/CD badge (shows green/red build status)

---

## Build Order Summary

| Step | What You Build | Time Estimate |
|------|----------------|---------------|
| 1 | Project setup + GitHub | 30 min |
| 2 | Dataset + EDA notebook | 1 hour |
| 3 | Data preprocessing pipeline | 1 hour |
| 4 | MLflow setup | 30 min |
| 5 | Train 12 models | 2 hours |
| 6 | Model comparison + XGBoost tuning | 1.5 hours |
| 7 | FastAPI service | 1.5 hours |
| 8 | Tests | 1 hour |
| 9 | Docker | 1 hour |
| 10 | CI/CD (GitHub Actions) | 1.5 hours |
| 11 | Deploy on Render | 30 min |
| 12 | README + polish | 30 min |
| **Total** | **Full working project** | **~12 hours** |

---

## Verification Checklist
- [ ] `mlflow ui` shows 12+ experiment runs with metrics
- [ ] XGBoost has highest AUC-ROC in MLflow comparison
- [ ] `pytest` passes all tests locally
- [ ] Docker container starts and `/health` returns 200
- [ ] GitHub Actions pipeline goes green on push
- [ ] Live URL responds to POST /predict in under 100ms
- [ ] Fraud transaction returns `is_fraud: true` with high probability
