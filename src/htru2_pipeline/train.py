"""Training: Random Forest pipeline, 5-fold grid search and feature importances.

`tune_rf` takes any (X, y), so the k-means representative-sample experiments
in the next step reuse exactly the same search as the full-data baseline.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import Config

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    model: Pipeline
    best_params: dict
    best_cv_score: float
    cv_results: pd.DataFrame
    importances: pd.DataFrame


def candidate_path(cfg: Config, name: str):
    """Where the trained model for a variant is stored."""
    return cfg.models_dir / "candidates" / f"rf_{name}.joblib"


def build_rf_pipeline(cfg: Config) -> Pipeline:
    """Optional scaler followed by a Random Forest."""
    steps = []
    if cfg.model.scale_features:
        steps.append(("scaler", StandardScaler()))
    steps.append((
        "rf",
        RandomForestClassifier(
            random_state=cfg.random_state,
            class_weight=cfg.model.class_weight,
        ),
    ))
    return Pipeline(steps)


def param_grid(cfg: Config) -> dict:
    return {
        "rf__n_estimators": cfg.model.n_estimators_grid,
        "rf__min_samples_leaf": cfg.model.min_samples_leaf_grid,
        "rf__max_depth": cfg.model.max_depth_grid,
    }


def tune_rf(X, y, cfg: Config) -> GridSearchCV:
    """Grid search with stratified k-fold CV, scored as set in the config."""
    cv = StratifiedKFold(
        n_splits=cfg.model.cv_folds, shuffle=True, random_state=cfg.random_state
    )
    grid = GridSearchCV(
        estimator=build_rf_pipeline(cfg),
        param_grid=param_grid(cfg),
        scoring=cfg.model.scoring,
        cv=cv,
        n_jobs=cfg.model.n_jobs,
    )
    grid.fit(X, y)
    logger.info(
        "Best params %s (CV %s = %.4f)",
        grid.best_params_, cfg.model.scoring, grid.best_score_,
    )
    return grid


def feature_importances(model: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    """Impurity-based importances of the fitted forest, sorted high to low."""
    imp = model.named_steps["rf"].feature_importances_
    return (
        pd.DataFrame({"feature": feature_names, "importance": imp})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def train_full_data_model(
    X_train, y_train, cfg: Config, name: str = "full_data", save: bool = True
) -> TrainingResult:
    """Tune and fit the RF on the given training data and optionally persist it."""
    logger.info("Tuning RF '%s' on %d rows", name, len(X_train))
    grid = tune_rf(X_train, y_train, cfg)
    model = grid.best_estimator_  # refit on the whole training set by GridSearchCV

    result = TrainingResult(
        model=model,
        best_params={k.removeprefix("rf__"): v for k, v in grid.best_params_.items()},
        best_cv_score=float(grid.best_score_),
        cv_results=pd.DataFrame(grid.cv_results_),
        importances=feature_importances(model, list(X_train.columns)),
    )
    if save:
        save_training_artifacts(result, cfg, name, n_train=len(X_train))
    return result


def save_training_artifacts(
    result: TrainingResult, cfg: Config, name: str, n_train: int
) -> None:
    rep = cfg.reports_dir / "training" / name
    rep.mkdir(parents=True, exist_ok=True)
    model_path = candidate_path(cfg, name)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    result.cv_results[
        [c for c in result.cv_results.columns
         if c.startswith("param_") or c in
         ("mean_test_score", "std_test_score", "rank_test_score")]
    ].to_csv(rep / "cv_results.csv", index=False)
    result.importances.to_csv(rep / "feature_importance.csv", index=False)
    (rep / "best_params.json").write_text(json.dumps(
        {
            "best_params": result.best_params,
            "best_cv_score": round(result.best_cv_score, 4),
            "scoring": cfg.model.scoring,
            "n_train": n_train,
        },
        indent=2,
        default=str,
    ))
    joblib.dump(result.model, model_path)
    logger.info("Saved reports to %s and model to %s", rep, model_path)
