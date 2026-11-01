from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from htru2_pipeline.config import EvaluationConfig
from htru2_pipeline.evaluate import evaluate
from htru2_pipeline.sampling import run_kmeans_experiments
from htru2_pipeline.split import split_data
from htru2_pipeline.train import build_rf_pipeline, train_full_data_model
from htru2_pipeline.uncertainty import (
    METRICS,
    bootstrap_samples,
    ci_rows,
    paired_rows,
    point_estimates,
    predict_arrays,
    run_uncertainty,
)


@pytest.fixture
def boot_cfg(small_cfg):
    return replace(small_cfg, evaluation=EvaluationConfig(bootstrap_samples=200))


@pytest.fixture
def fitted(clf_df, boot_cfg):
    X_tr, X_te, y_tr, y_te = split_data(clf_df, boot_cfg)
    model = build_rf_pipeline(boot_cfg).fit(X_tr, y_tr)
    return model, X_tr, y_tr, X_te, y_te


def test_point_estimates_match_the_evaluation_module(fitted, boot_cfg):
    model, _, _, X_te, y_te = fitted
    m = evaluate(model, X_te, y_te, boot_cfg)
    est = point_estimates(y_te, *predict_arrays(model, X_te, boot_cfg), boot_cfg)
    assert est["pr_auc"] == pytest.approx(m["pr_auc"], abs=1e-4)
    assert est["roc_auc"] == pytest.approx(m["roc_auc"], abs=1e-4)
    assert est["accuracy"] == pytest.approx(m["accuracy"], abs=1e-4)
    assert est["precision"] == pytest.approx(m["positive"]["precision"], abs=1e-4)
    assert est["recall"] == pytest.approx(m["positive"]["recall"], abs=1e-4)
    assert est["f1"] == pytest.approx(m["positive"]["f1"], abs=1e-4)


def test_bootstrap_has_one_row_per_resample_and_is_deterministic(fitted, boot_cfg):
    model, _, _, X_te, y_te = fitted
    arrays = predict_arrays(model, X_te, boot_cfg)
    a = bootstrap_samples(y_te, *arrays, boot_cfg)
    b = bootstrap_samples(y_te, *arrays, boot_cfg)
    assert a.shape == (200, len(METRICS))
    assert a.equals(b)


def test_intervals_are_ordered_and_bracket_the_estimate(fitted, boot_cfg):
    model, _, _, X_te, y_te = fitted
    arrays = predict_arrays(model, X_te, boot_cfg)
    samples = bootstrap_samples(y_te, *arrays, boot_cfg)
    est = point_estimates(y_te, *arrays, boot_cfg)
    for row in ci_rows("v", samples, est, boot_cfg):
        assert row["ci_low"] <= row["ci_high"]
        assert row["ci_low"] - 1e-3 <= row["estimate"] <= row["ci_high"] + 1e-3


def test_more_test_data_gives_narrower_intervals(fitted, boot_cfg):
    model, _, _, X_te, y_te = fitted
    y_pred, y_score = predict_arrays(model, X_te, boot_cfg)

    def width(n):
        s = bootstrap_samples(y_te.iloc[:n], y_pred[:n], y_score[:n], boot_cfg)
        return s["pr_auc"].quantile(0.975) - s["pr_auc"].quantile(0.025)

    assert width(100) > width(len(y_te))


def test_paired_difference_of_a_model_with_itself_is_zero(fitted, boot_cfg):
    model, _, _, X_te, y_te = fitted
    arrays = predict_arrays(model, X_te, boot_cfg)
    samples = bootstrap_samples(y_te, *arrays, boot_cfg)
    est = point_estimates(y_te, *arrays, boot_cfg)
    for row in paired_rows("same", "same", samples, samples, est, est, boot_cfg):
        assert row["difference"] == 0 and row["ci_low"] == 0 and row["ci_high"] == 0
        assert row["excludes_zero"] is False


def test_paired_bootstrap_detects_a_clearly_worse_model(fitted, boot_cfg):
    good, X_tr, y_tr, X_te, y_te = fitted
    bad = build_rf_pipeline(boot_cfg).fit(X_tr, y_tr.sample(frac=1, random_state=0).to_numpy())

    g_arrays = predict_arrays(good, X_te, boot_cfg)
    b_arrays = predict_arrays(bad, X_te, boot_cfg)
    g = bootstrap_samples(y_te, *g_arrays, boot_cfg)
    b = bootstrap_samples(y_te, *b_arrays, boot_cfg)
    rows = paired_rows(
        "bad", "good", b, g,
        point_estimates(y_te, *b_arrays, boot_cfg), point_estimates(y_te, *g_arrays, boot_cfg),
        boot_cfg,
    )
    pr = next(r for r in rows if r["metric"] == "pr_auc")
    assert pr["difference"] < 0 and pr["excludes_zero"] is True


def test_run_uncertainty_end_to_end(clf_df, boot_cfg):
    X_tr, X_te, y_tr, y_te = split_data(clf_df, boot_cfg)
    train_full_data_model(X_tr, y_tr, boot_cfg)
    run_kmeans_experiments(X_tr, y_tr, X_te, y_te, boot_cfg, ks=[5])

    ci, paired = run_uncertainty(["full_data", "kmeans_k5"], X_te, y_te, boot_cfg)
    assert len(ci) == 2 * len(METRICS)
    assert len(paired) == len(METRICS)
    assert set(paired["variant"]) == {"kmeans_k5"}
    assert set(paired["baseline"]) == {"full_data"}
    out = boot_cfg.reports_dir / "evaluation"
    assert (out / "confidence_intervals.csv").exists()
    assert (out / "paired_comparison.csv").exists()
    assert pd.read_csv(out / "paired_comparison.csv")["metric"].tolist() == METRICS


def test_run_uncertainty_needs_trained_models(clf_df, boot_cfg):
    _, X_te, _, y_te = split_data(clf_df, boot_cfg)
    with pytest.raises(FileNotFoundError, match="run the training stage"):
        run_uncertainty(["full_data"], X_te, y_te, boot_cfg)


def test_baseline_must_be_among_the_variants(clf_df, boot_cfg):
    _, X_te, _, y_te = split_data(clf_df, boot_cfg)
    with pytest.raises(ValueError, match="baseline"):
        run_uncertainty(["kmeans_k5"], X_te, y_te, boot_cfg)


def test_resamples_use_the_same_rows_for_every_model(clf_df, boot_cfg):
    """Paired comparisons are only valid if both models see identical resamples."""
    from htru2_pipeline.uncertainty import bootstrap_indices

    n = 300
    assert np.array_equal(bootstrap_indices(n, boot_cfg), bootstrap_indices(n, boot_cfg))
