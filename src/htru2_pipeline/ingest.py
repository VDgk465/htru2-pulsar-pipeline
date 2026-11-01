"""Ingestion: download HTRU_2.csv from UCI once and cache it locally."""
from __future__ import annotations

import io
import logging
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from .config import Config

logger = logging.getLogger(__name__)


class DataIngestionError(RuntimeError):
    """Raised when the dataset cannot be fetched or has an unexpected shape."""


def _download_csv_bytes(cfg: Config) -> bytes:
    """Download the zip archive and return the raw bytes of the CSV inside it."""
    url, member = cfg.data.url, cfg.data.csv_member
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            payload = resp.read()
    except OSError as exc:  # URLError and timeouts are OSError subclasses
        raise DataIngestionError(
            f"Could not download {url} ({exc}). Download {member} manually "
            f"and place it at {cfg.data.cache_path}."
        ) from exc

    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as exc:
        raise DataIngestionError(f"{url} did not return a valid zip archive") from exc

    with archive:
        matches = [
            n for n in archive.namelist() if Path(n).name.lower() == member.lower()
        ]
        if not matches:
            raise DataIngestionError(
                f"{member} not found in archive; it contains {archive.namelist()}"
            )
        return archive.read(matches[0])


def load_raw(cfg: Config, refresh: bool = False) -> pd.DataFrame:
    """Return the dataset with named columns, using the local cache if present."""
    path = cfg.data.cache_path
    if path.exists() and not refresh:
        logger.info("Loading cached dataset from %s", path)
    else:
        logger.info("Downloading HTRU2 from %s", cfg.data.url)
        data = _download_csv_bytes(cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)  # keep the original file untouched in the cache
        logger.info("Cached %d bytes to %s", len(data), path)

    df = pd.read_csv(path, header=None)
    if df.shape[1] != len(cfg.columns):
        raise DataIngestionError(
            f"{path} has {df.shape[1]} columns; expected {len(cfg.columns)}"
        )
    df.columns = cfg.columns
    return df
