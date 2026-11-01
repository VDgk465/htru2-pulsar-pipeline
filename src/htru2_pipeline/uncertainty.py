"""How much should we trust the test-set numbers? Bootstrap intervals and paired tests.

The test set is a finite sample (~330 pulsars), so every metric carries sampling
noise. We resample the test rows with replacement many times and recompute the
metrics each time:

* the spread of the resampled values gives a confidence interval per metric;
* for "is variant B really different from the baseline?", both models are scored
  on the SAME resamples and we look at the interval of the difference (a paired
  bootstrap). If it contains 0, the difference is within noise.

What this does NOT capture: variation from retraining (different seeds or training
samples). It measures test-set sampling noise only. With many metrics and variants,
a few intervals will exclude 0 by chance, so read them as evidence, not proof.
"""
from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import Config
from .train import candidate_path

logger = logging.getLogger(__name__)

METRICS = ["pr_auc", "roc_auc", "precision", "recall", "f1", "accuracy"]


def predict_arrays(model, X, cfg: Config):
    """Boolean 'predicted positive' and the positive-class probability."""
    pos = cfg.positive_label
    classes = list(model.classes_)
    y_pred = np.asarray(model.predict(X)) == pos
    y_score = model.predict_proba(X)[:, classes.index(pos)]
    return y_pred, y_score


def _metrics(t: np.ndarray, p: np.ndarray, s: np.ndarray) -> dict:
    """t, p: boolean true/predicted positive; s: positive-class score."""
    tp = int(np.sum(t & p))
    fp = int(np.sum(~t & p))
    fn = int(np.sum(t & ~p))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "pr_auc": average_precision_score(t, s),
        "roc_auc": roc_auc_score(t, s),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": float(np.mean(t == p)),
    }


def point_estimates(y_true, y_pred, y_score, cfg: Config) -> dict:
    t = np.asarray(y_true) == cfg.positive_label
    return _metrics(t, y_pred, y_score)


def bootstrap_indices(n: int, cfg: Config) -> np.ndarray:
    """Row indices for every resample. Same seed + n => identical resamples for all models."""
    rng = np.random.default_rng(cfg.random_state)
    return rng.integers(0, n, size=(cfg.evaluation.bootstrap_samples, n))


def bootstrap_samples(y_true, y_pred, y_score, cfg: Config, idx=None) -> pd.DataFrame:
    """One row of metrics per resample of the test set."""
    t = np.asarray(y_true) == cfg.positive_label
    idx = bootstrap_indices(len(t), cfg) if idx is None else idx
    rows = []
    for ix in idx:
        tb = t[ix]
        if tb.all() or not tb.any():  # AUC is undefined without both classes
            rows.append({m: np.nan for m in METRICS})
        else:
            rows.append(_metrics(tb, y_pred[ix], y_score[ix]))
    return pd.DataFrame(rows, columns=METRICS)


def _bounds(cfg: Config):
    alpha = (1 - cfg.evaluation.confidence) / 2
    return alpha, 1 - alpha


def ci_rows(variant: str, samples: pd.DataFrame, estimates: dict, cfg: Config) -> list[dict]:
    lo_q, hi_q = _bounds(cfg)
    return [
        {
            "variant": variant, "metric": m,
            "estimate": round(float(estimates[m]), 4),
            "ci_low": round(float(samples[m].quantile(lo_q)), 4),
            "ci_high": round(float(samples[m].quantile(hi_q)), 4),
        }
        for m in METRICS
    ]


def paired_rows(variant, baseline, samples, base_samples, estimates, base_estimates,
                cfg: Config) -> list[dict]:
    """Difference (variant - baseline) per metric, on identical resamples."""
    lo_q, hi_q = _bounds(cfg)
    rows = []
    for m in METRICS:
        diff = samples[m] - base_samples[m]
        low, high = float(diff.quantile(lo_q)), float(diff.quantile(hi_q))
        rows.append({
            "variant": variant, "baseline": baseline, "metric": m,
            "difference": round(float(estimates[m] - base_estimates[m]), 4),
            "ci_low": round(low, 4), "ci_high": round(high, 4),
            "excludes_zero": bool(low > 0 or high < 0),
        })
    return rows


def run_uncertainty(names, X_test, y_test, cfg: Config, baseline: str = "full_data",
                    save: bool = True):
    """Load each saved candidate, bootstrap the test set, return (ci_df, paired_df)."""
    if baseline not in names:
        raise ValueError(f"baseline '{baseline}' must be one of the variants {list(names)}")
    idx = bootstrap_indices(len(y_test), cfg)

    samples, estimates = {}, {}
    for name in names:
        path = candidate_path(cfg, name)
        if not path.exists():
            raise FileNotFoundError(
                f"No trained model for '{name}' at {path}; run the training stage first"
            )
        y_pred, y_score = predict_arrays(joblib.load(path), X_test, cfg)
        samples[name] = bootstrap_samples(y_test, y_pred, y_score, cfg, idx)
        estimates[name] = point_estimates(y_test, y_pred, y_score, cfg)
        logger.info("Bootstrapped %s (%d resamples)", name, len(idx))

    ci = pd.DataFrame([r for n in names for r in ci_rows(n, samples[n], estimates[n], cfg)])
    paired = pd.DataFrame(
        [
            r for n in names if n != baseline
            for r in paired_rows(n, baseline, samples[n], samples[baseline],
                                 estimates[n], estimates[baseline], cfg)
        ],
        columns=["variant", "baseline", "metric", "difference", "ci_low", "ci_high",
                 "excludes_zero"],
    )
    if save:
        out = cfg.reports_dir / "evaluation"
        out.mkdir(parents=True, exist_ok=True)
        ci.to_csv(out / "confidence_intervals.csv", index=False)
        paired.to_csv(out / "paired_comparison.csv", index=False)
    return ci, paired
