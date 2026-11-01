"""Turn metrics into flat table rows and a Markdown results summary."""
from __future__ import annotations

import pandas as pd

SUMMARY_COLUMNS = [
    "variant", "n_train", "best_cv_score", "accuracy", "roc_auc", "pr_auc",
    "precision", "recall", "f1",
]


def flat_row(variant, n_train, best_params, best_cv_score, metrics, k=None) -> dict:
    """One comparable row per trained variant (positive-class precision/recall/F1)."""
    cm = metrics["confusion_matrix"]
    return {
        "variant": variant,
        "k": k,
        "n_train": int(n_train),
        "best_cv_score": round(float(best_cv_score), 4),
        "n_estimators": best_params.get("n_estimators"),
        "min_samples_leaf": best_params.get("min_samples_leaf"),
        "max_depth": best_params.get("max_depth"),
        "accuracy": metrics["accuracy"],
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "precision": metrics["positive"]["precision"],
        "recall": metrics["positive"]["recall"],
        "f1": metrics["positive"]["f1"],
        "tp": cm["tp"], "fp": cm["fp"], "fn": cm["fn"],
    }


def to_markdown(df: pd.DataFrame) -> str:
    """Minimal GitHub-flavoured table (avoids the optional `tabulate` dependency)."""
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}"
        return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)

    header = "| " + " | ".join(df.columns) + " |"
    rule = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, rule, *body])


CI_METRICS = ["pr_auc", "roc_auc", "precision", "recall", "f1"]


def ci_table(ci_df: pd.DataFrame) -> pd.DataFrame:
    """One row per variant; each cell reads 'estimate [low, high]'."""
    rows = []
    for variant, grp in ci_df.groupby("variant", sort=False):
        by_metric = grp.set_index("metric")
        row = {"variant": variant}
        for m in CI_METRICS:
            r = by_metric.loc[m]
            row[m] = f"{r['estimate']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
        rows.append(row)
    return pd.DataFrame(rows)


def paired_table(paired_df: pd.DataFrame) -> pd.DataFrame:
    """Difference vs the baseline with its interval, for the headline metrics."""
    sub = paired_df[paired_df["metric"].isin(CI_METRICS)]
    return pd.DataFrame({
        "variant": sub["variant"].to_numpy(),
        "metric": sub["metric"].to_numpy(),
        "difference": [f"{d:+.4f}" for d in sub["difference"]],
        "interval": [f"[{lo:+.4f}, {hi:+.4f}]" for lo, hi in zip(sub["ci_low"], sub["ci_high"])],
        "distinguishable from 0": ["yes" if x else "no" for x in sub["excludes_zero"]],
    })


def results_markdown(summary: pd.DataFrame, importances: pd.DataFrame, decision: dict,
                     uncertainty=None) -> str:
    """Auto-generated results page: every number comes from the run that produced it."""
    cols = [c for c in SUMMARY_COLUMNS if c in summary.columns]
    parts = [
        "# Results (auto-generated)",
        "",
        "All metrics are computed on the held-out test set. `precision`, `recall` and "
        "`f1` refer to the positive (pulsar) class.",
        "",
        "## Model comparison",
        "",
        to_markdown(summary[cols]),
        "",
    ]
    if uncertainty is not None:
        ci_df, paired_df = uncertainty
        parts += [
            "## Confidence intervals",
            "",
            "Percentile bootstrap over the test rows. These intervals capture test-set "
            "sampling noise only, not variation from retraining with other seeds.",
            "",
            to_markdown(ci_table(ci_df)),
            "",
        ]
        if len(paired_df):
            parts += [
                "## Differences from the full-data model (paired bootstrap)",
                "",
                "Both models are scored on the same resamples. If the interval contains 0 "
                "the difference is within sampling noise. With several variants and "
                "metrics, a few intervals can exclude 0 by chance.",
                "",
                to_markdown(paired_table(paired_df)),
                "",
            ]
    parts += [
        "## Feature importance (full-data model)",
        "",
        to_markdown(importances.round(4)),
        "",
        "## Promotion decision",
        "",
        f"- Promoted: **{decision['promoted']}**",
        f"- Reason: {decision['reason']}",
        "",
    ]
    return "\n".join(parts)
