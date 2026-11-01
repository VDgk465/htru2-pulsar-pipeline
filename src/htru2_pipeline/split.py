"""Stratified train/test split and a check that the class balance was preserved."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import Config


def split_data(df: pd.DataFrame, cfg: Config):
    """Separate features from target and make a stratified split."""
    X, y = df[cfg.features], df[cfg.target]
    return train_test_split(
        X, y,
        test_size=cfg.test_size,
        stratify=y,
        random_state=cfg.random_state,
    )


def distribution_report(y_all, y_train, y_test) -> pd.DataFrame:
    """Class proportions and row counts for the full, train and test sets."""
    parts = {"full": y_all, "train": y_train, "test": y_test}
    table = pd.DataFrame(
        {name: y.value_counts(normalize=True) for name, y in parts.items()}
    ).T.sort_index(axis=1)
    table["n_rows"] = [len(y) for y in parts.values()]
    return table


def check_stratification(y_all, y_train, y_test, tol: float = 0.005) -> None:
    """Raise if train or test class proportions drift from the full dataset."""
    full = y_all.value_counts(normalize=True)
    for name, y in (("train", y_train), ("test", y_test)):
        drift = (y.value_counts(normalize=True) - full).abs().max()
        if drift > tol:
            raise ValueError(
                f"{name} class balance differs from the full data by {drift:.4f} "
                f"(tolerance {tol})"
            )
