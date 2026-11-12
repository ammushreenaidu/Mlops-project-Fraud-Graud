# FraudGuard: Real-Time Credit Card Fraud Detection with MLOps

A runnable educational MLOps project: training, evaluation, FastAPI serving, prediction logging, monitoring summary, and retraining workflow.

## Dataset
Download `creditcard.csv` from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and place it in `data/creditcard.csv`.
The CSV must contain `Time`, `V1`...`V28`, `Amount`, and `Class`. The dataset is highly imbalanced; use precision, recall, F1, and PR-AUC alongside the confusion matrix.

## Setup (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.train --data data/creditcard.csv
uvicorn src.api:app --reload
```
Open http://127.0.0.1:8000/docs to test the API.

## Linux/macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train --data data/creditcard.csv
uvicorn src.api:app --reload
```

## Example API request
The API expects all 30 input features (`Time`, `V1` through `V28`, `Amount`). Use a real row's features for a meaningful example. Do not include `Class` in the request.

```json
{
  "Time": 0,
  "V1": -1.35, "V2": -0.07, "V3": 2.54, "V4": 1.38,
  "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
  "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
  "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
  "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
  "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
  "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02,
  "Amount": 149.62
}
```
This is only an illustrative payload, not a validated transaction or expected prediction.

## MLOps workflow
1. Data validation and stratified train/test split.
2. Train a class-weighted Random Forest.
3. Save the model and metrics.
4. Serve predictions via FastAPI.
5. Log prediction timestamps, probabilities, and inputs (do not log sensitive real customer data).
6. Run `python -m src.monitor` to inspect prediction volume and feature drift indicators.
7. Run `python -m src.train --data data/creditcard.csv` to retrain, then compare metrics before promoting a model.

## Notes
- This is a teaching demo, not a bank-ready fraud prevention system.
- Model scores and threshold require validation on representative, current data.
- This dataset is old and anonymized; it does not represent all modern payment environments.
- Monitoring predictions alone cannot establish model quality without trustworthy labels.
