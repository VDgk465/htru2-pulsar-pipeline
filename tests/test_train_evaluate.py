import json
from dataclasses import replace

import joblib
import pytest

from htru2_pipeline.config import ModelConfig
from htru2_pipeline.evaluate import evaluate, save_metrics
from htru2_pipeline.split import split_data
from htru2_pipeline.train import (
    build_rf_pipeline,
    feature_importances,
    param_grid,
    train_full_data_model,
    tune_rf,
)


def test_pipeline_scaling_is_optional(cfg):
    assert [n for n, _ in build_rf_pipeline(cfg).steps] == ["rf"]
    scaled = replace(cfg, model=ModelConfig(scale_features=True))
    assert [n for n, _ in build_rf_pipeline(scaled).steps] == ["scaler", "rf"]


def test_pipeline_uses_class_weight_and_seed(cfg):
    rf = build_rf_pipeline(cfg).named_steps["rf"]
    assert rf.class_weight == "balanced"
    assert rf.random_state == 42


def test_param_grid_keys(small_cfg):
    g = param_grid(small_cfg)
    assert set(g) == {"rf__n_estimators", "rf__min_samples_leaf", "rf__max_depth"}
    assert None in g["rf__max_depth"]


def test_tune_rf_selects_from_grid_and_scores_well(clf_df, small_cfg):
    X_tr, _, y_tr, _ = split_data(clf_df, small_cfg)
    grid = tune_rf(X_tr, y_tr, small_cfg)
    assert grid.best_params_["rf__n_estimators"] in [10, 20]
    assert grid.best_params_["rf__max_depth"] in [5, None]
    assert 0.8 < grid.best_score_ <= 1.0


def test_feature_importances_sorted_and_normalised(clf_df, small_cfg):
    X_tr, _, y_tr, _ = split_data(clf_df, small_cfg)
    model = build_rf_pipeline(small_cfg).fit(X_tr, y_tr)
    imp = feature_importances(model, list(X_tr.columns))
    assert imp["importance"].is_monotonic_decreasing
    assert imp["importance"].sum() == pytest.approx(1.0)
    assert set(imp["feature"]) == set(small_cfg.features)


def test_train_full_data_model_saves_artifacts(clf_df, small_cfg):
    X_tr, X_te, y_tr, _ = split_data(clf_df, small_cfg)
    result = train_full_data_model(X_tr, y_tr, small_cfg)
    rep = small_cfg.reports_dir / "training" / "full_data"
    for name in ("best_params.json", "cv_results.csv", "feature_importance.csv"):
        assert (rep / name).exists()
    saved = json.loads((rep / "best_params.json").read_text())
    assert saved["n_train"] == len(X_tr)
    loaded = joblib.load(small_cfg.models_dir / "candidates" / "rf_full_data.joblib")
    assert (loaded.predict(X_te) == result.model.predict(X_te)).all()


def test_evaluate_metrics_are_consistent(clf_df, small_cfg, tmp_path):
    X_tr, X_te, y_tr, y_te = split_data(clf_df, small_cfg)
    model = build_rf_pipeline(small_cfg).fit(X_tr, y_tr)
    m = evaluate(model, X_te, y_te, small_cfg)

    cm = m["confusion_matrix"]
    assert sum(cm.values()) == m["n_rows"] == len(y_te)
    assert cm["tp"] + cm["fn"] == int((y_te == 1).sum())
    assert m["positive"]["recall"] == pytest.approx(cm["tp"] / (cm["tp"] + cm["fn"]), abs=1e-3)
    for key in ("accuracy", "roc_auc", "pr_auc"):
        assert 0 <= m[key] <= 1

    out = save_metrics(m, tmp_path / "metrics.json")
    assert json.loads(out.read_text())["n_rows"] == len(y_te)
