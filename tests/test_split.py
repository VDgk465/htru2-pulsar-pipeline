import pytest

from htru2_pipeline.split import check_stratification, distribution_report, split_data


def test_split_sizes_and_disjoint(df, cfg):
    X_tr, X_te, y_tr, y_te = split_data(df, cfg)
    assert len(X_te) == pytest.approx(0.2 * len(df), abs=1)
    assert set(X_tr.index).isdisjoint(X_te.index)
    assert cfg.target not in X_tr.columns
    assert list(X_tr.columns) == cfg.features


def test_split_is_stratified(df, cfg):
    _, _, y_tr, y_te = split_data(df, cfg)
    check_stratification(df["Target"], y_tr, y_te)  # should not raise
    table = distribution_report(df["Target"], y_tr, y_te)
    assert list(table.index) == ["full", "train", "test"]
    assert table[1].max() - table[1].min() < 0.005


def test_split_is_deterministic(df, cfg):
    a = split_data(df, cfg)[1].index
    b = split_data(df, cfg)[1].index
    assert list(a) == list(b)


def test_check_stratification_detects_drift(df):
    y = df["Target"]
    skewed_test = y[y == 1].head(150)  # nearly all positives
    with pytest.raises(ValueError, match="class balance differs"):
        check_stratification(y, y.iloc[:1500], skewed_test)
