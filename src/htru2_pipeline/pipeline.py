"""End-to-end orchestration: ingest -> validate -> split -> train -> evaluate ->
promote -> k-means experiments -> figures and a results summary."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from .config import Config
from .evaluate import evaluate, save_metrics
from .ingest import load_raw
from .promotion import promote_if_better
from .report import flat_row, results_markdown
from .sampling import run_kmeans_experiments
from .split import check_stratification, distribution_report, split_data
from .train import train_full_data_model
from .uncertainty import run_uncertainty
from .validate import validate
from .visualize import (
    plot_feature_importance,
    plot_tsne,
    plot_variant_comparison,
    save_figure,
    tsne_embedding,
)

logger = logging.getLogger(__name__)


def _dump(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
    return path


def run_pipeline(
    cfg: Config,
    refresh: bool = False,
    skip_kmeans: bool = False,
    skip_tsne: bool = False,
    skip_uncertainty: bool = False,
    spark_parity: bool = False,
) -> dict:
    """Run every stage and return a summary dict (also written to reports/)."""
    out = cfg.reports_dir

    # 1. Ingest and validate
    df = load_raw(cfg, refresh=refresh)
    validation = validate(df, cfg)
    _dump(validation, out / "data" / "validation_report.json")

    # 2. Stratified split, with a check that the class balance survived
    X_train, X_test, y_train, y_test = split_data(df, cfg)
    check_stratification(df[cfg.target], y_train, y_test)
    distribution_report(df[cfg.target], y_train, y_test).to_csv(
        out / "data" / "class_distribution.csv"
    )

    # 3. Train + evaluate the full-data model, then apply the promotion gate
    result = train_full_data_model(X_train, y_train, cfg)
    metrics = evaluate(result.model, X_test, y_test, cfg)
    save_metrics(metrics, out / "evaluation" / "full_data_metrics.json")
    decision = promote_if_better(
        result.model, metrics, cfg, name="rf_full_data", params=result.best_params
    )

    # 4. k-means representative-sample experiments
    rows = [flat_row("full_data", len(X_train), result.best_params,
                     result.best_cv_score, metrics)]
    summary = pd.DataFrame(rows)
    if not skip_kmeans:
        kmeans = run_kmeans_experiments(X_train, y_train, X_test, y_test, cfg)
        summary = pd.concat([summary, kmeans], ignore_index=True)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "summary.csv", index=False)

    # 5. Uncertainty: bootstrap intervals and paired comparisons against the baseline
    uncertainty = None
    if not skip_uncertainty:
        uncertainty = run_uncertainty(summary["variant"].tolist(), X_test, y_test, cfg)

    # 6. Figures and the Markdown results page
    figures = {
        "feature_importance": save_figure(
            plot_feature_importance(result.importances),
            out / "figures" / "feature_importance.png"),
    }
    if len(summary) > 1:
        figures["variant_comparison"] = save_figure(
            plot_variant_comparison(summary), out / "figures" / "variant_comparison.png")
    if not skip_tsne:
        emb = tsne_embedding(X_train, X_test, cfg)
        figures["tsne"] = save_figure(plot_tsne(emb, y_test, cfg), out / "figures" / "tsne.png")
    (out / "results.md").write_text(
        results_markdown(summary, result.importances, decision.to_dict(), uncertainty)
    )

    # 7. Optional: Spark MLlib parity check against the scikit-learn baseline
    parity = None
    if spark_parity:
        try:
            from .spark_pipeline import parity_check
        except ImportError as exc:
            raise RuntimeError(
                "Spark support needs pyspark and Java: python -m pip install pyspark pyarrow"
            ) from exc
        parity = parity_check(X_train, y_train, X_test, y_test, cfg, sk_metrics=metrics)

    run_summary = {
        "n_rows": validation["n_rows"],
        "class_fractions": validation["class_fractions"],
        "best_params": result.best_params,
        "best_cv_score": round(result.best_cv_score, 4),
        "test_metrics": metrics,
        "promotion": decision.to_dict(),
        "variants": summary.to_dict(orient="records"),
        "figures": {k: str(v) for k, v in figures.items()},
        "uncertainty": None if uncertainty is None else {
            "confidence_intervals": uncertainty[0].to_dict(orient="records"),
            "paired_vs_full_data": uncertainty[1].to_dict(orient="records"),
        },
        "spark_parity": parity,
    }
    _dump(run_summary, out / "run_summary.json")
    logger.info("Done. Results written to %s", out)
    return run_summary
