import os
import time
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src.api.schemas import TransactionInput, PredictionOutput

MODELS_DIR  = os.path.join('models')
MODEL_PATH  = os.path.join(MODELS_DIR, 'xgboost_tuned.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')

FEATURE_ORDER = (
    ['Time'] +
    [f'V{i}' for i in range(1, 29)] +
    ['Amount']
)

app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Predicts whether a credit card transaction is fraudulent. "
        "Powered by a tuned XGBoost model trained on 284,807 real transactions."
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Load model and scaler once at startup — not on every request
# Loading takes ~200ms; inference itself takes <10ms
# ---------------------------------------------------------------------------
@app.on_event("startup")
def load_artifacts():
    global model, scaler
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run tune_xgboost.py first.")
    if not os.path.exists(SCALER_PATH):
        raise RuntimeError(f"Scaler not found at {SCALER_PATH}. Run preprocess.py first.")
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", tags=["Info"])
def root():
    return {
        "name"       : "Fraud Detection API",
        "version"    : "1.0.0",
        "model"      : "XGBoost (tuned)",
        "endpoints"  : {
            "POST /predict" : "Send a transaction, get fraud prediction",
            "GET  /health"  : "Health check",
            "GET  /docs"    : "Interactive API documentation (Swagger UI)",
        }
    }


@app.get("/health", tags=["Info"])
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict(transaction: TransactionInput):
    """
    Send a transaction's 30 features and get back:
    - is_fraud: true or false
    - fraud_probability: confidence score between 0 and 1
    - inference_ms: how long the prediction took

    Note on V1–V28: These are PCA-transformed features produced by the bank's
    internal systems to protect cardholder privacy. You don't enter them manually —
    the bank's transaction pipeline generates them automatically. For testing,
    copy any row from creditcard.csv and remove the Class column.
    """
    try:
        # build DataFrame in the exact column order the model was trained on
        row = pd.DataFrame([transaction.model_dump()], columns=FEATURE_ORDER)

        # scale Amount and Time (same transformation applied during training)
        row[['Amount', 'Time']] = scaler.transform(row[['Amount', 'Time']])

        t0              = time.perf_counter()
        fraud_proba     = float(model.predict_proba(row)[0][1])
        inference_ms    = round((time.perf_counter() - t0) * 1000, 2)

        is_fraud = fraud_proba >= 0.5

        return PredictionOutput(
            is_fraud=is_fraud,
            fraud_probability=round(fraud_proba, 4),
            inference_ms=inference_ms,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
