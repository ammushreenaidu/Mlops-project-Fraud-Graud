import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import train_test_split

from src.features import FEATURES, TARGET, validate_columns

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"

def train(data_path):
    df = pd.read_csv(data_path)
    validate_columns(df)
    X, y = df[FEATURES], df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=250,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    threshold = 0.5
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "model": "RandomForestClassifier",
        "threshold": threshold,
        "test_rows": int(len(y_test)),
        "fraud_count_test": int(y_test.sum()),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "average_precision_pr_auc": float(average_precision_score(y_test, probabilities)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix_tn_fp_fn_tp": confusion_matrix(y_test, predictions).ravel().tolist(),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
        "features": FEATURES
    }

    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "threshold": threshold},
                ARTIFACTS / "fraud_model.joblib")
    with open(ARTIFACTS / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps({k: v for k, v in metrics.items()
                      if k != "classification_report"}, indent=2))
    print(f"\nSaved model: {ARTIFACTS / 'fraud_model.joblib'}")
    print(f"Saved metrics: {ARTIFACTS / 'metrics.json'}")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/creditcard.csv")
    args = parser.parse_args()
    path = Path(args.data)
    if not path.is_absolute():
        path = ROOT / path
    train(path)
