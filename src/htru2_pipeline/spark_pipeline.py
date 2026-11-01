"""Spark MLlib implementation of the same workflow (optional: needs Java + pyspark).

Honest framing: HTRU2 has ~18k rows, which scikit-learn handles in seconds. This
module exists to show the same logic running on a distributed engine, to check
that both engines agree (`parity_check`), and to measure where Spark starts to
pay off (see `benchmark.py`).

Differences from scikit-learn, all deliberate and documented:
  * Spark's RandomForest caps tree depth at 30, so "unlimited" becomes 30.
  * CrossValidator folds are random, not stratified.
  * Class balancing uses a weight column (n / (n_classes * n_class)), which is
    the formula behind scikit-learn's class_weight="balanced".
"""
from __future__ import annotations

import logging

import pandas as pd
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.clustering import KMeans as SparkKMeans
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StandardScaler as SparkStandardScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import DenseVector
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

from .config import Config
from .evaluate import evaluate
from .pipeline import _dump
from .sampling import ratio_from_counts
from .train import train_full_data_model

logger = logging.getLogger(__name__)


def get_spark(cfg: Config) -> SparkSession:
    return (
        SparkSession.builder.master(cfg.spark.master)
        .appName(cfg.spark.app_name)
        .config("spark.driver.memory", cfg.spark.driver_memory)
        .config("spark.sql.shuffle.partitions", str(cfg.spark.shuffle_partitions))
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )


def to_spark(spark: SparkSession, X: pd.DataFrame, y: pd.Series, cfg: Config) -> DataFrame:
    pdf = X.reset_index(drop=True).copy()
    pdf[cfg.target] = y.to_numpy().astype(float)
    return spark.createDataFrame(pdf)


def assemble(df: DataFrame, cfg: Config) -> DataFrame:
    """Pack feature columns into one vector column; label as double."""
    return (
        VectorAssembler(inputCols=cfg.features, outputCol="features")
        .transform(df)
        .select("features", F.col(cfg.target).cast("double").alias("label"))
    )


def add_class_weights(df: DataFrame) -> DataFrame:
    """Balanced weights: n_samples / (n_classes * n_samples_in_class)."""
    counts = {r["label"]: r["count"] for r in df.groupBy("label").count().collect()}
    n, n_classes = sum(counts.values()), len(counts)
    mapping = F.create_map(
        *[F.lit(x) for lab, c in counts.items() for x in (lab, n / (n_classes * c))]
    )
    return df.withColumn("weight", mapping[F.col("label")])


def tune_rf_spark(train_df: DataFrame, cfg: Config):
    """Cross-validated grid search (ROC-AUC). Returns (cv_model, best_params, cv_score)."""
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", seed=cfg.random_state)
    if "weight" in train_df.columns and cfg.model.class_weight == "balanced":
        rf = rf.setWeightCol("weight")
    grid = (
        ParamGridBuilder()
        .addGrid(rf.numTrees, cfg.spark.num_trees_grid)
        .addGrid(rf.minInstancesPerNode, cfg.spark.min_instances_grid)
        .addGrid(rf.maxDepth, cfg.spark.max_depth_grid)
        .build()
    )
    evaluator = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )
    cv = CrossValidator(
        estimator=rf, estimatorParamMaps=grid, evaluator=evaluator,
        numFolds=cfg.spark.cv_folds, seed=cfg.random_state,
    )
    cv_model = cv.fit(train_df)
    best = cv_model.bestModel
    params = {
        "n_estimators": len(best.trees),
        "min_samples_leaf": best.getMinInstancesPerNode(),
        "max_depth": best.getMaxDepth(),
    }
    return cv_model, params, float(max(cv_model.avgMetrics))


def evaluate_spark(model, test_df: DataFrame, cfg: Config) -> dict:
    """Same metric schema as `evaluate.evaluate` so the two engines are comparable."""
    if cfg.positive_label != 1:
        raise ValueError("Spark's BinaryClassificationEvaluator assumes positive label 1")
    pred = model.transform(test_df)
    counts = {
        (r["label"], r["prediction"]): r["count"]
        for r in pred.groupBy("label", "prediction").count().collect()
    }
    tp, fp = counts.get((1.0, 1.0), 0), counts.get((0.0, 1.0), 0)
    fn, tn = counts.get((1.0, 0.0), 0), counts.get((0.0, 0.0), 0)

    def prf(hit, false_pos, false_neg):
        p = hit / (hit + false_pos) if hit + false_pos else 0.0
        r = hit / (hit + false_neg) if hit + false_neg else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}

    def area(metric):
        return BinaryClassificationEvaluator(
            labelCol="label", rawPredictionCol="rawPrediction", metricName=metric
        ).evaluate(pred)

    total = tp + fp + fn + tn
    return {
        "n_rows": int(total),
        "accuracy": round((tp + tn) / total, 4),
        "roc_auc": round(area("areaUnderROC"), 4),
        "pr_auc": round(area("areaUnderPR"), 4),
        "positive": prf(tp, fp, fn),
        "negative": prf(tn, fn, fp),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def select_representatives_spark(df: DataFrame, k: int, cfg: Config) -> DataFrame:
    """Spark version of the k-means step: one real row nearest to each centroid."""
    scaler = SparkStandardScaler(
        inputCol="features", outputCol="scaled", withMean=True, withStd=True
    ).fit(df)
    scaled = scaler.transform(df)
    km = SparkKMeans(
        k=k, seed=cfg.random_state, featuresCol="scaled", predictionCol="cluster", maxIter=50
    ).fit(scaled)
    centers = [DenseVector(c) for c in km.clusterCenters()]

    @F.udf(DoubleType())
    def sq_dist(vec, cluster):
        return float(vec.squared_distance(centers[cluster]))

    ranked = Window.partitionBy("cluster").orderBy("dist")
    return (
        km.transform(scaled)
        .withColumn("dist", sq_dist("scaled", "cluster"))
        .withColumn("rn", F.row_number().over(ranked))
        .filter(F.col("rn") == 1)
        .select("features", "label")
    )


def build_representative_set_spark(df: DataFrame, k: int, cfg: Config) -> DataFrame:
    """k positive + ratio*k negative representatives, as in the scikit-learn path."""
    pos_label = float(cfg.positive_label)
    pos = df.filter(F.col("label") == pos_label)
    neg = df.filter(F.col("label") != pos_label)
    ratio = ratio_from_counts(pos.count(), neg.count(), cfg.sampling.ratio_mode)
    return select_representatives_spark(pos, k, cfg).unionByName(
        select_representatives_spark(neg, k * ratio, cfg)
    )


def parity_check(
    X_train, y_train, X_test, y_test, cfg: Config,
    sk_metrics: dict | None = None, spark: SparkSession | None = None, save: bool = True,
) -> dict:
    """Train the Spark RF on the SAME split and compare it with scikit-learn.

    This is a sanity check, not an equality test: the engines use different
    algorithms internally, so metrics should agree within `parity_tolerance`.
    """
    own_session = spark is None
    spark = spark or get_spark(cfg)
    try:
        train_df = add_class_weights(assemble(to_spark(spark, X_train, y_train, cfg), cfg))
        test_df = assemble(to_spark(spark, X_test, y_test, cfg), cfg)
        train_df.cache()
        test_df.cache()

        cv_model, params, cv_score = tune_rf_spark(train_df, cfg)
        spark_metrics = evaluate_spark(cv_model.bestModel, test_df, cfg)
    finally:
        if own_session:
            spark.stop()

    if sk_metrics is None:
        result = train_full_data_model(X_train, y_train, cfg, save=False)
        sk_metrics = evaluate(result.model, X_test, y_test, cfg)

    def pick(m):
        return {"roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"], "f1": m["positive"]["f1"]}

    sk, sp = pick(sk_metrics), pick(spark_metrics)
    diff = {k: round(abs(sk[k] - sp[k]), 4) for k in sk}
    report = {
        "sklearn": sk, "spark": sp, "abs_diff": diff,
        "tolerance": cfg.spark.parity_tolerance,
        "passed": all(d <= cfg.spark.parity_tolerance for d in diff.values()),
        "spark_best_params": params,
        "spark_best_cv_roc_auc": round(cv_score, 4),
    }
    if save:
        _dump(report, cfg.reports_dir / "spark" / "parity.json")
    logger.info("Spark parity %s: %s", "PASSED" if report["passed"] else "FAILED", diff)
    return report
