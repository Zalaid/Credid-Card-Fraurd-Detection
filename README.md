# Fraud Detection MLOps Pipeline

An end-to-end ML pipeline that trains and benchmarks 12+ models on real credit card fraud data, selects the best model (XGBoost), and serves real-time predictions via a FastAPI REST API — packaged in Docker with full CI/CD.

## Architecture

```
Raw Data → Preprocessing (SMOTE) → Train 12 Models → MLflow Tracking
                                                          ↓
                                               XGBoost (Best Model)
                                                          ↓
                                            FastAPI → Docker → Render (Live)
                                                          ↑
                                              GitHub Actions CI/CD
```

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 transactions, 492 fraudulent (0.17% fraud rate).

## Model Benchmark

| Model | AUC-ROC | F1 | Recall |
|-------|---------|----|--------|
| XGBoost | — | — | — |
| LightGBM | — | — | — |
| ... | ... | ... | ... |

*(Will be populated after Step 5)*

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Preprocess data
python src/data/preprocess.py

# 3. Train all models
python src/models/train_all_models.py

# 4. Start API
uvicorn src.api.main:app --reload
```

## API Usage

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"V1": -1.35, "V2": -0.07, "Amount": 149.62, "Time": 0, ...}'
```

Response:
```json
{"is_fraud": true, "fraud_probability": 0.94, "inference_ms": 8}
```

## Run with Docker

```bash
docker build -t fraud-detection-api .
docker run -p 8000:8000 fraud-detection-api
```

## Tech Stack

- **ML:** scikit-learn, XGBoost, LightGBM, CatBoost
- **Tracking:** MLflow
- **API:** FastAPI + Uvicorn
- **Imbalance:** SMOTE (imbalanced-learn)
- **Infra:** Docker, GitHub Actions, Render
