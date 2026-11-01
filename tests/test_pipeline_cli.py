import json

import pytest
import yaml

from htru2_pipeline.cli import main
from htru2_pipeline.pipeline import run_pipeline


@pytest.fixture
def cached_cfg(clf_df, small_cfg):
    clf_df.to_csv(small_cfg.data.cache_path, index=False, header=False)
    return small_cfg


def test_pipeline_runs_end_to_end(cached_cfg):
    cfg = cached_cfg
    summary = run_pipeline(cfg)
    out = cfg.reports_dir

    for rel in (
        "data/validation_report.json", "data/class_distribution.csv",
        "evaluation/full_data_metrics.json", "kmeans_experiments.csv", "summary.csv",
        "results.md", "run_summary.json", "figures/feature_importance.png",
        "figures/variant_comparison.png", "figures/tsne.png",
        "evaluation/confidence_intervals.csv", "evaluation/paired_comparison.csv",
    ):
        assert (out / rel).exists(), rel
    assert (cfg.models_dir / "production" / "model.joblib").exists()

    assert summary["promotion"]["promoted"] is True
    assert [v["variant"] for v in summary["variants"]] == [
        "full_data", "kmeans_k5", "kmeans_k10",
    ]
    saved = json.loads((out / "run_summary.json").read_text())
    assert saved["test_metrics"]["roc_auc"] == summary["test_metrics"]["roc_auc"]
    assert "full_data" in (out / "results.md").read_text()
    assert "## Confidence intervals" in (out / "results.md").read_text()
    assert summary["uncertainty"]["paired_vs_full_data"]


def test_rerunning_on_the_same_data_does_not_re_promote(cached_cfg):
    run_pipeline(cached_cfg, skip_kmeans=True, skip_tsne=True)
    second = run_pipeline(cached_cfg, skip_kmeans=True, skip_tsne=True)
    assert second["promotion"]["promoted"] is False
    history = (cached_cfg.models_dir / "production" / "history.jsonl").read_text().splitlines()
    assert len(history) == 2


def test_skip_flags(cached_cfg):
    summary = run_pipeline(
        cached_cfg, skip_kmeans=True, skip_tsne=True, skip_uncertainty=True
    )
    assert len(summary["variants"]) == 1
    assert summary["uncertainty"] is None
    assert not (cached_cfg.reports_dir / "evaluation" / "confidence_intervals.csv").exists()
    assert "Confidence intervals" not in (cached_cfg.reports_dir / "results.md").read_text()
    assert not (cached_cfg.reports_dir / "figures" / "tsne.png").exists()
    assert not (cached_cfg.reports_dir / "figures" / "variant_comparison.png").exists()


def _write_config(path, tmp_path, columns):
    path.write_text(yaml.safe_dump({
        "data": {"url": "file:///unused.zip", "csv_member": "HTRU_2.csv",
                 "cache_path": str(tmp_path / "HTRU_2.csv")},
        "columns": columns, "target": "Target", "valid_labels": [0, 1],
        "random_state": 42, "validation": {"min_rows": 1000},
        "reports_dir": str(tmp_path / "reports"), "models_dir": str(tmp_path / "models"),
        "model": {"cv_folds": 3, "n_jobs": 1, "n_estimators_grid": [10],
                  "min_samples_leaf_grid": [1], "max_depth_grid": [5]},
        "sampling": {"ks": [5], "kmeans_n_init": 3},
        "promotion": {"min_metric": 0.0},
        "evaluation": {"bootstrap_samples": 50},
    }))
    return path


def test_cli_runs_and_prints_a_summary(cached_cfg, tmp_path, capsys):
    cfg_path = _write_config(tmp_path / "cfg.yaml", tmp_path, cached_cfg.columns)
    assert main(["--config", str(cfg_path), "--skip-tsne"]) == 0
    printed = capsys.readouterr().out
    assert "Promotion: True" in printed and "ROC-AUC" in printed
    assert (tmp_path / "reports" / "results.md").exists()


def test_cli_returns_1_on_bad_data(small_cfg, tmp_path, capsys):
    small_cfg.data.cache_path.write_text("1,2\n3,4\n")  # wrong number of columns
    cfg_path = _write_config(tmp_path / "cfg.yaml", tmp_path, small_cfg.columns)
    assert main(["--config", str(cfg_path)]) == 1
