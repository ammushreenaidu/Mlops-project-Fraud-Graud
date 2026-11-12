from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "logs" / "predictions.csv"
METRICS_PATH = ROOT / "artifacts" / "metrics.json"

def main():
    if not LOG_PATH.exists():
        print("No prediction logs yet. Call POST /predict first.")
        return
    df = pd.read_csv(LOG_PATH)
    report = {
        "prediction_count": int(len(df)),
        "predicted_fraud_count": int((df["predicted_class"] == 1).sum()),
        "predicted_fraud_rate": float(df["predicted_class"].mean()),
        "mean_fraud_probability": float(df["fraud_probability"].mean()),
        "feature_mean": {c: float(df[c].mean()) for c in ["Time", "Amount", "V1", "V2", "V3"] if c in df}
    }
    if METRICS_PATH.exists():
        with open(METRICS_PATH, encoding="utf-8") as f:
            report["training_metrics_available"] = True
            report["training_test_pr_auc"] = json.load(f).get("average_precision_pr_auc")
    print(json.dumps(report, indent=2))
    print("\nBasic monitoring only: compare feature distributions against a validated reference dataset.")
    print("Prediction logs do not reveal actual fraud rates unless reliable labels are later attached.")

if __name__ == "__main__":
    main()
