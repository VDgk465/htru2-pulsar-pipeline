import numpy as np
import pytest

from htru2_pipeline.validate import DataValidationError, validate


def test_validate_passes_and_reports(df, cfg):
    report = validate(df, cfg)
    assert report["n_rows"] == len(df)
    assert report["n_features"] == 8
    assert report["class_counts"] == {0: 1820, 1: 180}
    assert report["class_fractions"][1] == pytest.approx(0.09)
    assert all(v == 0 for v in report["missing_values"].values())
    assert set(report["feature_stats"]) == set(cfg.features)


def test_rejects_missing_column(df, cfg):
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate(df.drop(columns=["Mean_DM_SNR"]), cfg)


def test_rejects_nulls(df, cfg):
    df.loc[0, "Mean_DM_SNR"] = np.nan
    with pytest.raises(DataValidationError, match="Null values"):
        validate(df, cfg)


def test_rejects_non_numeric_feature(df, cfg):
    df["Std_DM_SNR"] = df["Std_DM_SNR"].astype(str)
    with pytest.raises(DataValidationError, match="Non-numeric"):
        validate(df, cfg)


def test_rejects_unexpected_label(df, cfg):
    df.loc[0, "Target"] = 5
    with pytest.raises(DataValidationError, match="Unexpected labels"):
        validate(df, cfg)


def test_rejects_missing_class(df, cfg):
    df["Target"] = 0
    with pytest.raises(DataValidationError, match="Labels missing"):
        validate(df, cfg)


def test_rejects_too_few_rows(df, cfg):
    with pytest.raises(DataValidationError, match="rows"):
        validate(df.head(100), cfg)


def test_duplicates_are_reported_not_fatal(df, cfg):
    dup = df.iloc[[0]]
    import pandas as pd
    report = validate(pd.concat([df, dup]), cfg)
    assert report["n_duplicate_rows"] == 1
