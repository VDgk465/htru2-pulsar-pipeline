"""k-means representative sampling: shrink the training set, keep the diversity.

For each class, cluster the (standardised) training rows with k-means and keep
only the real row closest to each centroid. The negative class gets
`ratio * k` clusters so the original class balance is maintained.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler

from .config import Config
from .evaluate import evaluate, save_metrics
from .report import flat_row
from .train import train_full_data_model

logger = logging.getLogger(__name__)


def ratio_from_counts(n_pos: int, n_neg: int, mode: str = "round") -> int:
    """Negative-to-positive ratio as a whole number (at least 1)."""
    if n_pos <= 0:
        raise ValueError("No positive samples to compute a class ratio from")
    ratio = n_neg / n_pos
    if mode == "round":
        value = round(ratio)
    elif mode == "floor":
        value = math.floor(ratio)
    else:
        raise ValueError(f"ratio_mode must be 'round' or 'floor', got {mode!r}")
    return max(1, int(value))


def select_representatives(X: pd.DataFrame, k: int, cfg: Config) -> pd.DataFrame:
    """Rows of X (original, unscaled values) closest to each of k k-means centroids."""
    if not 1 <= k <= len(X):
        raise ValueError(f"k must be between 1 and {len(X)}, got {k}")
    X_scaled = StandardScaler().fit_transform(X)
    km = KMeans(
        n_clusters=k, random_state=cfg.random_state, n_init=cfg.sampling.kmeans_n_init
    ).fit(X_scaled)
    closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, X_scaled)
    return X.iloc[sorted({int(i) for i in np.asarray(closest)})]


def build_representative_set(X_train: pd.DataFrame, y_train: pd.Series, k: int, cfg: Config):
    """k positive representatives plus ratio*k negative ones, with aligned labels."""
    pos_mask = y_train == cfg.positive_label
    X_pos, X_neg = X_train[pos_mask], X_train[~pos_mask]
    ratio = ratio_from_counts(len(X_pos), len(X_neg), cfg.sampling.ratio_mode)

    pos_reps = select_representatives(X_pos, k, cfg)
    neg_reps = select_representatives(X_neg, k * ratio, cfg)
    if min(len(pos_reps), len(neg_reps)) < cfg.model.cv_folds:
        raise ValueError(
            f"k={k} gives {len(pos_reps)} positive and {len(neg_reps)} negative "
            f"representatives; each class needs at least cv_folds={cfg.model.cv_folds}"
        )
    X_rep = pd.concat([pos_reps, neg_reps])
    y_rep = y_train.loc[X_rep.index]
    logger.info(
        "k=%d: %d positives + %d negatives (ratio %d) = %d training rows",
        k, len(pos_reps), len(neg_reps), ratio, len(X_rep),
    )
    return X_rep, y_rep


def run_kmeans_experiments(
    X_train, y_train, X_test, y_test, cfg: Config, ks=None, save: bool = True
) -> pd.DataFrame:
    """Train and evaluate a tuned RF on the representative set for every k."""
    rows = []
    for k in ks or cfg.sampling.ks:
        X_rep, y_rep = build_representative_set(X_train, y_train, k, cfg)
        name = f"kmeans_k{k}"
        result = train_full_data_model(X_rep, y_rep, cfg, name=name, save=save)
        metrics = evaluate(result.model, X_test, y_test, cfg)
        if save:
            save_metrics(metrics, cfg.reports_dir / "evaluation" / f"{name}_metrics.json")
        rows.append(flat_row(
            name, len(X_rep), result.best_params, result.best_cv_score, metrics, k=k,
        ))
    table = pd.DataFrame(rows)
    if save:
        path = cfg.reports_dir / "kmeans_experiments.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)
    return table
