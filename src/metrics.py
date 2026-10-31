"""
metrics.py
This file computes the 6 evaluation metrics for binary classification: Accuracy, AUC, Precision, Recall, F1, MCC.
Positive class = 1 (GBM, the aggressive tumour grade).
"""

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
)


def evaluate_model(model, X, y):
    """Return a dict of Accuracy, AUC, Precision, Recall, F1, MCC."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]   # probability of GBM, needed for AUC

    return {
        "Accuracy":  accuracy_score(y, y_pred),
        "AUC":       roc_auc_score(y, y_prob),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall":    recall_score(y, y_pred, zero_division=0),
        "F1":        f1_score(y, y_pred, zero_division=0),
        "MCC":       matthews_corrcoef(y, y_pred),
    }