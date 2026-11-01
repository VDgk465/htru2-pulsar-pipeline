"""Spark tests. Skipped automatically when pyspark or Java is not available."""
import json

import pandas as pd
import pytest
from sklearn.datasets import make_classification

pytest.importorskip("pyspark")

from pyspark.sql import functions as F  # noqa: E402

from htru2_pipeline.benchmark import run_benchmark  # noqa: E402
from htru2_pipeline.sampling import ratio_from_counts  # noqa: E402
from htru2_pipeline.spark_pipeline import (  # noqa: E402
    add_class_weights,
    assemble,
    build_representative_set_spark,
    evaluate_spark,
    parity_check,
    select_representatives_spark,
    to_spark,
    tune_rf_spark,
)
from htru2_pipeline.split import split_data  # noqa: E402


@pytest.fixture
def easy_df(small_cfg):
    X, y = make_classification(
        n_samples=1500, n_features=8, n_informative=4, weights=[0.91, 0.09],
        class_sep=3.0, random_state=0,
    )
    out = pd.DataFrame(X, columns=small_cfg.features)
    out[small_cfg.target] = y
    return out


def test_assemble_builds_vector_and_double_label(spark_session, easy_df, small_cfg):
    sdf = assemble(to_spark(spark_session, easy_df[small_cfg.features],
                            easy_df[small_cfg.target], small_cfg), small_cfg)
    assert sdf.columns == ["features", "label"]
    assert dict(sdf.dtypes)["label"] == "double"
    assert sdf.count() == len(easy_df)


def test_class_weights_balance_the_classes(spark_session, easy_df, small_cfg):
    sdf = add_class_weights(assemble(to_spark(
        spark_session, easy_df[small_cfg.features], easy_df[small_cfg.target], small_cfg
    ), small_cfg))
    rows = sdf.groupBy("label").agg(F.sum("weight").alias("w")).collect()
    totals = {r["label"]: r["w"] for r in rows}
    assert totals[0.0] == pytest.approx(totals[1.0])
    assert totals[0.0] == pytest.approx(len(easy_df) / 2)


def test_tune_and_evaluate_spark(spark_session, easy_df, small_cfg):
    X_tr, X_te, y_tr, y_te = split_data(easy_df, small_cfg)
    train = add_class_weights(assemble(to_spark(spark_session, X_tr, y_tr, small_cfg), small_cfg))
    test = assemble(to_spark(spark_session, X_te, y_te, small_cfg), small_cfg)

    cv_model, params, cv_score = tune_rf_spark(train, small_cfg)
    assert params["n_estimators"] in small_cfg.spark.num_trees_grid
    assert cv_score > 0.9

    m = evaluate_spark(cv_model.bestModel, test, small_cfg)
    cm = m["confusion_matrix"]
    assert sum(cm.values()) == m["n_rows"] == len(y_te)
    assert cm["tp"] + cm["fn"] == int((y_te == 1).sum())
    assert m["roc_auc"] > 0.9


def test_spark_representatives_one_per_cluster(spark_session, small_cfg):
    import numpy as np

    rng = np.random.default_rng(0)
    blocks = []
    for centre in (-10.0, 0.0, 10.0):
        block = pd.DataFrame(rng.normal(scale=0.3, size=(40, 8)), columns=small_cfg.features)
        block += centre  # separate the blobs in every feature
        blocks.append(block)
    X = pd.concat(blocks, ignore_index=True)
    y = pd.Series([0] * len(X))

    sdf = assemble(to_spark(spark_session, X, y, small_cfg), small_cfg)
    reps = select_representatives_spark(sdf, 3, small_cfg).collect()
    blobs = sorted(int(round(r["features"][0] / 10)) for r in reps)
    assert blobs == [-1, 0, 1]


def test_spark_representative_set_keeps_the_class_ratio(spark_session, easy_df, small_cfg):
    X_tr, _, y_tr, _ = split_data(easy_df, small_cfg)
    sdf = assemble(to_spark(spark_session, X_tr, y_tr, small_cfg), small_cfg)
    reps = build_representative_set_spark(sdf, 5, small_cfg)
    ratio = ratio_from_counts(int((y_tr == 1).sum()), int((y_tr == 0).sum()))
    counts = {r["label"]: r["count"] for r in reps.groupBy("label").count().collect()}
    assert counts == {1.0: 5, 0.0: 5 * ratio}


def test_parity_check_between_engines(spark_session, easy_df, small_cfg):
    X_tr, X_te, y_tr, y_te = split_data(easy_df, small_cfg)
    report = parity_check(X_tr, y_tr, X_te, y_te, small_cfg, spark=spark_session)
    assert report["passed"], report
    saved = json.loads((small_cfg.reports_dir / "spark" / "parity.json").read_text())
    assert set(saved["abs_diff"]) == {"roc_auc", "pr_auc", "f1"}


def test_benchmark_produces_a_timing_table(spark_session, easy_df, small_cfg):
    X_tr, _, y_tr, _ = split_data(easy_df, small_cfg)
    table = run_benchmark(X_tr, y_tr, small_cfg, factors=[1, 2], k=5, spark=spark_session)
    assert len(table) == 2 * 4
    assert set(table["engine"]) == {"scikit-learn", "spark"}
    assert set(table["task"]) == {"rf_fit", "kmeans_representatives"}
    assert (table["seconds"] > 0).all()
    assert list(table.groupby("factor")["rows"].first()) == [len(X_tr), 2 * len(X_tr)]
    assert (small_cfg.reports_dir / "benchmark.csv").exists()
