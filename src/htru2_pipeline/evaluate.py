"""Evaluation on held-out data: metrics that suit an imbalanced target."""
from __future__ import annotations

import json
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from .config import Config


def evaluate(model, X, y, cfg: Config) -> dict:
    """Return accuracy, ROC-AUC, PR-AUC, per-class precision/recall/F1 and counts."""
    pos = cfg.positive_label
    classes = list(model.classes_)
    neg = next(c for c in classes if c != pos)

    y_pred = model.predict(X)
    proba = model.predict_proba(X)[:, classes.index(pos)]
    report = classification_report(y, y_pred, output_dict=True, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[neg, pos]).ravel()

    def _cls(label):
        r = report[str(label)]
        return {
            "precision": round(r["precision"], 4),
            "recall": round(r["recall"], 4),
            "f1": round(r["f1-score"], 4),
        }

    return {
        "n_rows": int(len(y)),
        "accuracy": round(accuracy_score(y, y_pred), 4),
        "roc_auc": round(roc_auc_score(y, proba), 4),
        "pr_auc": round(average_precision_score(y, proba, pos_label=pos), 4),
        "positive": _cls(pos),
        "negative": _cls(neg),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        },
    }


def classification_report_text(model, X, y) -> str:
    return classification_report(y, model.predict(X))


def save_metrics(metrics: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))
    return path
