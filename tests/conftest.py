import os
import shutil
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from htru2_pipeline.config import (
    Config,
    DataConfig,
    EvaluationConfig,
    ModelConfig,
    PromotionConfig,
    SamplingConfig,
    SparkConfig,
)

COLUMNS = [
    "Mean_Integrated_Profile", "Std_Integrated_Profile",
    "Excess_Kurtosis_Integrated_Profile", "Skewness_Integrated_Profile",
    "Mean_DM_SNR", "Std_DM_SNR", "Excess_Kurtosis_DM_SNR",
    "Skewness_DM_SNR", "Target",
]


@pytest.fixture
def df():
    """Synthetic data with the real schema and ~9% positives (pure noise)."""
    rng = np.random.default_rng(0)
    n = 2000
    y = np.array([1] * 180 + [0] * (n - 180))
    rng.shuffle(y)
    data = {c: rng.normal(size=n) for c in COLUMNS[:-1]}
    data["Target"] = y
    return pd.DataFrame(data)


@pytest.fixture
def cfg(tmp_path):
    return Config(
        data=DataConfig("file:///unused.zip", "HTRU_2.csv", tmp_path / "HTRU_2.csv"),
        columns=COLUMNS,
        target="Target",
        valid_labels=[0, 1],
        random_state=42,
        min_rows=1000,
        label_names={0: "Non-pulsar", 1: "Pulsar"},
        reports_dir=tmp_path / "reports",
        models_dir=tmp_path / "models",
    )


@pytest.fixture
def clf_df(cfg):
    """Synthetic but learnable data: ~9% positives, 8 informative-ish features."""
    X, y = make_classification(
        n_samples=2000, n_features=8, n_informative=4, weights=[0.91, 0.09],
        class_sep=1.5, random_state=0,
    )
    out = pd.DataFrame(X, columns=cfg.features)
    out[cfg.target] = y
    return out


@pytest.fixture
def small_cfg(cfg):
    """The test config with tiny grids so the whole suite stays fast."""
    return replace(
        cfg,
        model=ModelConfig(
            cv_folds=3, n_jobs=1,
            n_estimators_grid=[10, 20], min_samples_leaf_grid=[1, 5],
            max_depth_grid=[5, None],
        ),
        sampling=SamplingConfig(ks=[5, 10], kmeans_n_init=3),
        promotion=PromotionConfig(min_metric=0.0),
        evaluation=EvaluationConfig(bootstrap_samples=60),
        spark=SparkConfig(
            master="local[2]", driver_memory="1g", shuffle_partitions=4, cv_folds=2,
            num_trees_grid=[5, 10], min_instances_grid=[1], max_depth_grid=[5],
            parity_tolerance=0.15,
        ),
    )


@pytest.fixture(scope="session")
def spark_session():
    """One shared local Spark session; the Spark tests are skipped without pyspark/Java."""
    pytest.importorskip("pyspark")
    if shutil.which("java") is None and not os.environ.get("JAVA_HOME"):
        pytest.skip("Java not found; Spark tests skipped")
    from pyspark.sql import SparkSession

    try:
        spark = (
            SparkSession.builder.master("local[2]").appName("htru2-tests")
            .config("spark.sql.shuffle.partitions", "4")
            .config("spark.ui.enabled", "false")
            .config("spark.driver.host", "127.0.0.1")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .getOrCreate()
        )
    except Exception as exc:  # JVM / environment problems are not test failures
        pytest.skip(f"Could not start Spark: {exc}")
    yield spark
    spark.stop()
