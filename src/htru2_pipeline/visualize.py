"""Figures. Built with matplotlib's object-oriented Figure (no pyplot), so they work
the same in notebooks, scripts, CI and Docker without a GUI backend."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from .config import Config


def save_figure(fig: Figure, path: str | Path, dpi: int = 120) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    return path


def plot_feature_importance(importances: pd.DataFrame) -> Figure:
    data = importances.sort_values("importance")
    fig = Figure(figsize=(8, 4.5))
    ax = fig.subplots()
    ax.barh(data["feature"], data["importance"])
    ax.set(xlabel="Importance", title="Random Forest feature importance")
    fig.tight_layout()
    return fig


def plot_variant_comparison(summary: pd.DataFrame) -> Figure:
    """Grouped bars of positive-class precision / recall / F1 for every variant."""
    metrics = ["precision", "recall", "f1"]
    x = np.arange(len(summary))
    width = 0.25
    fig = Figure(figsize=(9, 4.5))
    ax = fig.subplots()
    for i, m in enumerate(metrics):
        ax.bar(x + (i - 1) * width, summary[m], width, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["variant"], rotation=15)
    ax.set(ylim=(0, 1.05), ylabel="Score (pulsar class)",
           title="Full-data model vs k-means representative sets")
    ax.legend()
    fig.tight_layout()
    return fig


def tsne_embedding(X_train: pd.DataFrame, X_test: pd.DataFrame, cfg: Config) -> np.ndarray:
    """2D t-SNE of the test set (default parameters).

    The scaler is fit on the TRAINING data and applied to the test data, so test
    statistics never influence the preprocessing.
    """
    scaler = StandardScaler().fit(X_train)
    return TSNE(n_components=2, random_state=cfg.random_state).fit_transform(
        scaler.transform(X_test)
    )


def plot_tsne(embedding: np.ndarray, y: pd.Series, cfg: Config) -> Figure:
    y = np.asarray(y)
    fig = Figure(figsize=(8, 6))
    ax = fig.subplots()
    for label in sorted(cfg.valid_labels):
        mask = y == label
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1], s=14,
            alpha=0.5 if label != cfg.positive_label else 0.8,
            label=cfg.label_names.get(label, f"Class {label}"),
        )
    ax.set(title="t-SNE of the test set", xlabel="t-SNE 1", ylabel="t-SNE 2")
    ax.legend()
    fig.tight_layout()
    return fig
