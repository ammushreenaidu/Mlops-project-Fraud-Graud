# FraudGuard: Real-Time Credit Card Fraud Detection Using MLOps

## Abstract
FraudGuard is an educational machine learning operations project that demonstrates how a credit-card fraud classifier can be trained, evaluated, deployed as an API, monitored, and retrained. A Random Forest classifier is trained on anonymized transaction features. Because fraud is a rare event, model quality is assessed using precision, recall, F1-score, ROC-AUC, average precision (PR-AUC), and a confusion matrix. FastAPI provides an inference endpoint, while prediction logs support basic operational monitoring. The project emphasizes reproducibility, schema validation, model artifact management, and the distinction between a classroom prototype and a production financial system.

## 1. Introduction
Digital payment systems need methods to identify suspicious activity quickly. Machine learning can learn patterns in previously labeled transactions, but the model must also be operated reliably after training. MLOps brings together data preparation, model development, deployment, monitoring, and controlled retraining.

## 2. Problem Statement
Develop a repeatable pipeline that classifies each transaction as potentially fraudulent or likely legitimate and provides an API for scoring. The pipeline should preserve a held-out test set, report suitable metrics for imbalanced classification, and log operational predictions for monitoring.

## 3. Objectives
- Validate the dataset schema and target labels.
- Train a Random Forest model with class weighting.
- Evaluate the model using precision, recall, F1, ROC-AUC, PR-AUC, and a confusion matrix.
- Save a reusable model artifact and metrics file.
- Deploy predictions through FastAPI.
- Log predictions and produce a monitoring summary.
- Define a retraining and model-promotion process.

## 4. Dataset Description
The project uses the Kaggle Credit Card Fraud Detection dataset. It contains 284,807 transactions, with 492 fraud cases, and is extremely imbalanced. `Time`, `Amount`, and `V1` through `V28` are input features; `Class` is the target. The V features are anonymized PCA components. The dataset is a benchmark and does not represent every modern payment environment.

Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## 5. Methodology
1. Load the CSV and verify required columns, missing values, and binary target values.
2. Separate the 30 input features from the target.
3. Create a stratified 80/20 train-test split.
4. Fit a Random Forest classifier with class weighting on the training data.
5. Generate probabilities on the held-out test set.
6. Apply a documented initial threshold of 0.5 and compute evaluation metrics.
7. Save the fitted model, threshold, and metrics.
8. Start the FastAPI service and submit feature values to `/predict`.
9. Log timestamp, predicted class, probability, and input features.
10. Inspect log summaries and retrain only when data quality and model performance justify it.

## 6. Model and Evaluation
The baseline is `RandomForestClassifier` with 250 trees and `class_weight="balanced_subsample"`. The threshold is configurable in the saved model bundle. No performance values should be claimed until the project is run on the downloaded dataset. The actual results are written to `artifacts/metrics.json`.

Metrics:
- Precision: fraction of predicted fraud cases that are truly fraud.
- Recall: fraction of all fraud cases detected.
- F1-score: harmonic mean of precision and recall.
- ROC-AUC: ranking performance across thresholds.
- PR-AUC / average precision: particularly informative when the positive class is rare.
- Confusion matrix: true negatives, false positives, false negatives, true positives.

## 7. Deployment
FastAPI exposes:
- `GET /`: service information.
- `GET /health`: basic health and model-file check.
- `POST /predict`: returns a fraud probability and predicted category.

Interactive API documentation is available at `/docs`. Docker support is provided for packaging the API after a trained artifact exists.

## 8. Monitoring
The service appends predictions to `logs/predictions.csv`. `python -m src.monitor` reports prediction count, predicted-fraud rate, mean predicted probability, and selected feature means. This is a starter monitoring layer, not a complete drift detector. A stronger deployment would compare incoming features to a reference window, monitor latency/errors, protect logs, and evaluate predictions once trustworthy labels become available.

## 9. Retraining Strategy
Retraining can be triggered by:
- A meaningful shift in input-feature distributions.
- Deterioration in precision, recall, or PR-AUC once labels are available.
- A scheduled review interval or an approved new labeled dataset.

Before replacing a model, train a candidate, evaluate it on a fixed validation process, compare it with the current model, record its version and metrics, and promote it only if it meets predefined acceptance criteria. Do not retrain automatically from unverified predictions.

## 10. Limitations and Ethics
The dataset is anonymized and historical. Its features do not include many real-world signals such as merchant category or device information. False positives can inconvenience legitimate customers, while false negatives can miss fraud. Real financial deployment requires privacy and security controls, bias and impact assessment, human oversight, incident response, and validation with current representative data.

## 11. Conclusion
FraudGuard demonstrates the main components of an MLOps workflow: training, evaluation, artifact storage, API deployment, logging, monitoring, and retraining planning. It can be extended with MLflow experiment tracking, CI tests, a registry, drift detection, and a cloud deployment. The model is an educational prototype and must not be treated as a production banking control.

## 12. Run Instructions
See the root `README.md` for installation, dataset placement, training, API launch, testing, monitoring, and Docker commands.
