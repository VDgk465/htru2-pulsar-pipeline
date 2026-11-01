import pandas as pd
import pytest
from matplotlib.figure import Figure

from htru2_pipeline.report import (
    ci_table,
    flat_row,
    paired_table,
    results_markdown,
    to_markdown,
)
from htru2_pipeline.split import split_data
from htru2_pipeline.visualize import (
    plot_feature_importance,
    plot_tsne,
    plot_variant_comparison,
    save_figure,
    tsne_embedding,
)

METRICS = {
    "accuracy": 0.97, "roc_auc": 0.98, "pr_auc": 0.90,
    "positive": {"precision": 0.85, "recall": 0.88, "f1": 0.86},
    "negative": {"precision": 0.99, "recall": 0.98, "f1": 0.98},
    "confusion_matrix": {"tn": 3200, "fp": 40, "fn": 40, "tp": 300},
}


def make_summary():
    rows = [
        flat_row("full_data", 14318, {"n_estimators": 50, "max_depth": 10}, 0.979, METRICS),
        flat_row("kmeans_k10", 110, {"n_estimators": 100}, 0.95, METRICS, k=10),
    ]
    return pd.DataFrame(rows)


def test_flat_row_has_comparable_fields():
    row = flat_row("full_data", 100, {"n_estimators": 50}, 0.9791234, METRICS)
    assert row["precision"] == 0.85 and row["recall"] == 0.88 and row["tp"] == 300
    assert row["best_cv_score"] == 0.9791
    assert row["max_depth"] is None  # missing params stay empty, not invented


def test_to_markdown_is_a_valid_table():
    table = to_markdown(pd.DataFrame({"a": [1, 2], "b": [0.5, 0.25]}))
    lines = table.splitlines()
    assert lines[0] == "| a | b |"
    assert lines[1] == "| --- | --- |"
    assert lines[2] == "| 1 | 0.5000 |"


def test_results_markdown_contains_every_variant_and_the_decision():
    imp = pd.DataFrame({"feature": ["f1", "f2"], "importance": [0.7, 0.3]})
    md = results_markdown(
        make_summary(), imp, {"promoted": True, "reason": "no production model yet"}
    )
    assert "full_data" in md and "kmeans_k10" in md
    assert "no production model yet" in md and "f1" in md


def test_figures_are_built_and_saved(tmp_path):
    imp = pd.DataFrame({"feature": ["f1", "f2"], "importance": [0.7, 0.3]})
    for fig in (plot_feature_importance(imp), plot_variant_comparison(make_summary())):
        assert isinstance(fig, Figure)
        path = save_figure(fig, tmp_path / "nested" / "fig.png")
        assert path.exists() and path.stat().st_size > 0


def test_tsne_embedding_and_plot(clf_df, small_cfg, tmp_path):
    X = clf_df[small_cfg.features]
    y = clf_df[small_cfg.target]
    emb = tsne_embedding(X.iloc[:100], X.iloc[100:200], small_cfg)
    assert emb.shape == (100, 2)
    fig = plot_tsne(emb, y.iloc[100:200], small_cfg)
    assert save_figure(fig, tmp_path / "tsne.png").exists()


def test_tsne_scaler_uses_training_statistics_only(clf_df, small_cfg):
    X_tr, X_te, _, _ = split_data(clf_df, small_cfg)
    a = tsne_embedding(X_tr.iloc[:150], X_te.iloc[:80], small_cfg)
    b = tsne_embedding(X_tr.iloc[:150], X_te.iloc[:80], small_cfg)
    assert a == pytest.approx(b)  # deterministic given the seed


def make_uncertainty():
    metrics = ["pr_auc", "roc_auc", "precision", "recall", "f1", "accuracy"]
    ci = pd.DataFrame([
        {"variant": v, "metric": m, "estimate": 0.9, "ci_low": 0.85, "ci_high": 0.95}
        for v in ("full_data", "kmeans_k5") for m in metrics
    ])
    paired = pd.DataFrame([
        {"variant": "kmeans_k5", "baseline": "full_data", "metric": m, "difference": -0.02,
         "ci_low": -0.06, "ci_high": 0.02, "excludes_zero": False}
        for m in metrics
    ])
    return ci, paired


def test_ci_table_formats_estimate_and_interval():
    table = ci_table(make_uncertainty()[0])
    assert list(table["variant"]) == ["full_data", "kmeans_k5"]
    assert table.loc[0, "recall"] == "0.9000 [0.8500, 0.9500]"
    assert "accuracy" not in table.columns


def test_paired_table_flags_distinguishable_differences():
    table = paired_table(make_uncertainty()[1])
    assert set(table["distinguishable from 0"]) == {"no"}
    assert table.loc[0, "difference"] == "-0.0200"
    assert table.loc[0, "interval"] == "[-0.0600, +0.0200]"


def test_results_markdown_includes_uncertainty_sections_when_given():
    imp = pd.DataFrame({"feature": ["f1"], "importance": [1.0]})
    decision = {"promoted": False, "reason": "tie"}
    with_ci = results_markdown(make_summary(), imp, decision, make_uncertainty())
    assert "## Confidence intervals" in with_ci
    assert "## Differences from the full-data model" in with_ci
    assert "0.9000 [0.8500, 0.9500]" in with_ci
    without = results_markdown(make_summary(), imp, decision)
    assert "Confidence intervals" not in without
