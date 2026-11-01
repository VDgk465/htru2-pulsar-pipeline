"""Data validation: fail fast with a clear message before any training happens."""
from __future__ import annotations

import logging

import pandas as pd

from .config import Config

logger = logging.getLogger(__name__)


class DataValidationError(ValueError):
    """Raised when the input data does not meet the pipeline's expectations."""


def validate(df: pd.DataFrame, cfg: Config) -> dict:
    """Check schema, types, nulls, labels and size. Return a summary report."""
    missing_cols = [c for c in cfg.columns if c not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Missing required columns: {missing_cols}")

    errors: list[str] = []

    if len(df) < cfg.min_rows:
        errors.append(f"Only {len(df)} rows; expected at least {cfg.min_rows}")

    nulls = df[cfg.columns].isnull().sum()
    nulls = nulls[nulls > 0]
    if not nulls.empty:
        errors.append(f"Null values found: {nulls.to_dict()}")

    non_numeric = [
        c for c in cfg.features if not pd.api.types.is_numeric_dtype(df[c])
    ]
    if non_numeric:
        errors.append(f"Non-numeric feature columns: {non_numeric}")

    present = set(df[cfg.target].dropna().unique())
    unexpected = sorted(present - set(cfg.valid_labels))
    if unexpected:
        errors.append(f"Unexpected labels {unexpected}; allowed {cfg.valid_labels}")
    absent = sorted(set(cfg.valid_labels) - present)
    if absent:
        errors.append(f"Labels missing from the data: {absent} (cannot stratify)")

    if errors:
        raise DataValidationError("; ".join(errors))

    counts = df[cfg.target].value_counts()
    report = {
        "n_rows": int(len(df)),
        "n_features": len(cfg.features),
        "dtypes": {c: str(t) for c, t in df[cfg.columns].dtypes.items()},
        "missing_values": {c: int(v) for c, v in df[cfg.columns].isnull().sum().items()},
        "n_duplicate_rows": int(df.duplicated().sum()),
        "class_counts": {int(k): int(v) for k, v in counts.items()},
        "class_fractions": {
            int(k): round(float(v), 4)
            for k, v in df[cfg.target].value_counts(normalize=True).items()
        },
        "feature_stats": df[cfg.features].describe().T.round(4).to_dict(orient="index"),
    }
    logger.info(
        "Validation passed: %d rows, class fractions %s",
        report["n_rows"], report["class_fractions"],
    )
    return report
