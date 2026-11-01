import zipfile
from dataclasses import replace

import pandas as pd
import pytest

from htru2_pipeline.ingest import DataIngestionError, load_raw


def _make_zip(tmp_path, df, member="HTRU2/HTRU_2.csv"):
    path = tmp_path / "htru2.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(member, df.to_csv(index=False, header=False))
    return path


def _with_url(cfg, url):
    return replace(cfg, data=replace(cfg.data, url=url))


def test_download_extract_and_cache(df, cfg, tmp_path):
    cfg = _with_url(cfg, _make_zip(tmp_path, df).as_uri())
    out = load_raw(cfg)
    assert list(out.columns) == cfg.columns
    assert len(out) == len(df)
    assert cfg.data.cache_path.exists()


def test_cache_is_used_without_network(df, cfg, tmp_path):
    load_raw(_with_url(cfg, _make_zip(tmp_path, df).as_uri()))
    offline = _with_url(cfg, (tmp_path / "missing.zip").as_uri())
    assert len(load_raw(offline)) == len(df)


def test_refresh_with_bad_url_fails_clearly(df, cfg, tmp_path):
    load_raw(_with_url(cfg, _make_zip(tmp_path, df).as_uri()))
    offline = _with_url(cfg, (tmp_path / "missing.zip").as_uri())
    with pytest.raises(DataIngestionError, match="place it at"):
        load_raw(offline, refresh=True)


def test_download_failure_suggests_manual_placement(cfg, tmp_path):
    bad = _with_url(cfg, (tmp_path / "nope.zip").as_uri())
    with pytest.raises(DataIngestionError, match="Download HTRU_2.csv manually"):
        load_raw(bad)


def test_missing_csv_in_archive(df, cfg, tmp_path):
    zip_path = _make_zip(tmp_path, df, member="other.csv")
    with pytest.raises(DataIngestionError, match="not found in archive"):
        load_raw(_with_url(cfg, zip_path.as_uri()))


def test_not_a_zip(cfg, tmp_path):
    junk = tmp_path / "junk.zip"
    junk.write_bytes(b"this is not a zip file")
    with pytest.raises(DataIngestionError, match="valid zip"):
        load_raw(_with_url(cfg, junk.as_uri()))


def test_wrong_column_count(cfg):
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(
        cfg.data.cache_path, index=False, header=False
    )
    with pytest.raises(DataIngestionError, match="expected 9"):
        load_raw(cfg)
