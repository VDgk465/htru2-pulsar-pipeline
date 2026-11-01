"""Scale benchmark: scikit-learn vs Spark on replicated (noisy) copies of the data.

Usage:  python -m htru2_pipeline.benchmark --factors 1 10 50

Timings depend heavily on your hardware, and Spark here runs in local mode on a
single machine, so treat the results as "where does the overhead stop
dominating", not as a cluster benchmark. Spark gets a warm-up run first so JVM
start-up is not counted against it.
"""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .config import Config, load_config
from .ingest import load_raw
from .sampling import build_representative_set
from .split import split_data

logger = logging.getLogger(__name__)

TREES, DEPTH, NOISE = 50, 10, 0.05


def _seconds(fn) -> float:
    start = time.perf_counter()
    fn()
    return round(time.perf_counter() - start, 3)


def scaled_pandas(X: pd.DataFrame, y: pd.Series, factor: int, seed: int):
    """`factor` noisy copies of the data (noise = 5% of each feature's std)."""
    rng = np.random.default_rng(seed)
    Xb = np.tile(X.to_numpy(), (factor, 1))
    Xb = Xb + rng.normal(scale=NOISE * X.std().to_numpy(), size=Xb.shape)
    return (
        pd.DataFrame(Xb, columns=X.columns),
        pd.Series(np.tile(y.to_numpy(), factor), name=y.name),
    )


def run_benchmark(X, y, cfg: Config, factors=(1, 10, 50), k: int = 10, spark=None,
                  save: bool = True) -> pd.DataFrame:
    from pyspark.sql import functions as F
    from pyspark.ml.classification import RandomForestClassifier as SparkRF

    from .spark_pipeline import (
        assemble,
        build_representative_set_spark,
        get_spark,
        to_spark,
    )

    own = spark is None
    spark = spark or get_spark(cfg)
    rows = []
    try:
        base = to_spark(spark, X, y, cfg)
        std = X.std()

        # Warm-up so JVM start-up and first-job overhead are not timed.
        SparkRF(numTrees=2, maxDepth=2).fit(assemble(base, cfg))

        for factor in factors:
            Xb, yb = scaled_pandas(X, y, factor, cfg.random_state)
            n = len(Xb)

            big = base.crossJoin(spark.range(factor).withColumnRenamed("id", "_rep"))
            for i, c in enumerate(cfg.features):
                big = big.withColumn(
                    c, F.col(c) + F.randn(cfg.random_state + i) * float(NOISE * std[c])
                )
            sdf = assemble(big.drop("_rep"), cfg).cache()
            sdf.count()  # materialise so the timings measure compute, not generation

            sk_rf = RandomForestClassifier(
                n_estimators=TREES, max_depth=DEPTH, n_jobs=-1,
                class_weight="balanced", random_state=cfg.random_state,
            )
            tasks = [
                ("rf_fit", "scikit-learn", lambda: sk_rf.fit(Xb, yb)),
                ("rf_fit", "spark", lambda: SparkRF(
                    numTrees=TREES, maxDepth=DEPTH, seed=cfg.random_state).fit(sdf)),
                ("kmeans_representatives", "scikit-learn",
                 lambda: build_representative_set(Xb, yb, k, cfg)),
                ("kmeans_representatives", "spark",
                 lambda: build_representative_set_spark(sdf, k, cfg).count()),
            ]
            for task, engine, fn in tasks:
                secs = _seconds(fn)
                rows.append({"rows": n, "factor": factor, "task": task,
                             "engine": engine, "seconds": secs})
                logger.info("%9d rows | %-24s | %-12s | %8.2fs", n, task, engine, secs)
            sdf.unpersist()
    finally:
        if own:
            spark.stop()

    table = pd.DataFrame(rows)
    if save:
        path = cfg.reports_dir / "benchmark.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)
        logger.info("Benchmark written to %s", path)
    return table


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="scikit-learn vs Spark scale benchmark")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--factors", type=int, nargs="+", default=[1, 10, 50])
    p.add_argument("--k", type=int, default=10)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = load_config(args.config)
    X_train, _, y_train, _ = split_data(load_raw(cfg), cfg)
    table = run_benchmark(X_train, y_train, cfg, factors=args.factors, k=args.k)
    print(table.pivot_table(index=["rows", "task"], columns="engine", values="seconds"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
