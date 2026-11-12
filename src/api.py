from datetime import datetime, timezone
from pathlib import Path
import csv
import json

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features import FEATURES

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "fraud_model.joblib"
LOG_PATH = ROOT / "logs" / "predictions.csv"

app = FastAPI(title="FraudGuard API", version="1.0.0",
              description="Educational credit-card fraud scoring API.")

class Transaction(BaseModel):
    Time: float = Field(ge=0)
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(ge=0)

def load_bundle():
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503,
                            detail="Model not found. Run `python -m src.train --data data/creditcard.csv` first.")
    return joblib.load(MODEL_PATH)

def log_prediction(record):
    LOG_PATH.parent.mkdir(exist_ok=True)
    new_file = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(record.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(record)

@app.get("/")
def root():
    return {"service": "FraudGuard", "status": "running", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok", "model_available": MODEL_PATH.exists()}

@app.post("/predict")
def predict(transaction: Transaction):
    bundle = load_bundle()
    row = transaction.model_dump()
    X = pd.DataFrame([[row[c] for c in FEATURES]], columns=FEATURES)
    model = bundle["model"]
    probability = float(model.predict_proba(X)[0, 1])
    threshold = float(bundle.get("threshold", 0.5))
    predicted_fraud = probability >= threshold

    # Log only anonymized numeric features and outputs for this educational demo.
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "fraud_probability": probability,
        "predicted_class": int(predicted_fraud),
        **row
    }
    log_prediction(record)
    return {
        "prediction": "potential_fraud" if predicted_fraud else "likely_legitimate",
        "fraud_probability": round(probability, 6),
        "threshold": threshold,
        "disclaimer": "Decision-support demo only; do not use as the sole basis for real financial decisions."
    }
