from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from htru2_pipeline.sampling import (
    build_representative_set,
    ratio_from_counts,
    run_kmeans_experiments,
    select_representatives,
)
from htru2_pipeline.split import split_data


def test_ratio_modes_match_the_htru2_counts():
    # 16,259 negatives and 1,639 positives -> 9.92: rounding keeps the ratio, floor loses it.
    assert ratio_from_counts(1639, 16259, "round") == 10
    assert ratio_from_counts(1639, 16259, "floor") == 9


def test_ratio_edge_cases():
    assert ratio_from_counts(10, 4, "round") == 1  # never below 1
    with pytest.raises(ValueError, match="positive"):
        ratio_from_counts(0, 10)
    with pytest.raises(ValueError, match="ratio_mode"):
        ratio_from_counts(1, 10, "ceil")


def test_select_representatives_returns_real_rows(clf_df, small_cfg):
    X = clf_df[small_cfg.features]
    reps = select_representatives(X, 12, small_cfg)
    assert len(reps) == 12
    assert reps.index.isin(X.index).all()
    assert reps.equals(X.loc[reps.index])  # original, unscaled values


def test_select_representatives_rejects_bad_k(clf_df, small_cfg):
    X = clf_df[small_cfg.features].head(10)
    with pytest.raises(ValueError, match="k must be"):
        select_representatives(X, 11, small_cfg)


def test_one_representative_per_well_separated_cluster(small_cfg):
    rng = np.random.default_rng(0)
    centres = [-10.0, 0.0, 10.0]
    blocks = []
    for c in centres:
        block = pd.DataFrame(rng.normal(scale=0.3, size=(40, 8)), columns=small_cfg.features)
        block += c  # separate the blobs in every feature
        blocks.append(block)
    X = pd.concat(blocks, ignore_index=True)

    reps = select_representatives(X, 3, small_cfg)
    blob_of_rep = sorted(int(round(v / 10)) for v in reps.iloc[:, 0])
    assert blob_of_rep == [-1, 0, 1]


def test_representative_set_keeps_the_class_ratio(clf_df, small_cfg):
    X_tr, _, y_tr, _ = split_data(clf_df, small_cfg)
    X_rep, y_rep = build_representative_set(X_tr, y_tr, 5, small_cfg)
    ratio = round((y_tr == 0).sum() / (y_tr == 1).sum())
    assert (y_rep == 1).sum() == 5
    assert (y_rep == 0).sum() == 5 * ratio
    assert y_rep.equals(y_tr.loc[X_rep.index])


def test_too_few_positives_for_cv_is_rejected(clf_df, small_cfg):
    strict = replace(small_cfg, model=replace(small_cfg.model, cv_folds=10))
    X_tr, _, y_tr, _ = split_data(clf_df, strict)
    with pytest.raises(ValueError, match="cv_folds"):
        build_representative_set(X_tr, y_tr, 5, strict)


def test_run_kmeans_experiments_table_and_files(clf_df, small_cfg):
    X_tr, X_te, y_tr, y_te = split_data(clf_df, small_cfg)
    table = run_kmeans_experiments(X_tr, y_tr, X_te, y_te, small_cfg)
    assert list(table["k"]) == [5, 10]
    assert list(table["variant"]) == ["kmeans_k5", "kmeans_k10"]
    ratio = round((y_tr == 0).sum() / (y_tr == 1).sum())
    assert list(table["n_train"]) == [5 * (1 + ratio), 10 * (1 + ratio)]
    assert table[["roc_auc", "pr_auc", "precision", "recall", "f1"]].notna().all().all()
    assert (small_cfg.reports_dir / "kmeans_experiments.csv").exists()
    assert (small_cfg.reports_dir / "evaluation" / "kmeans_k5_metrics.json").exists()
