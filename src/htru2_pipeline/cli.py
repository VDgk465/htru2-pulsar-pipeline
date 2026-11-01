"""Command-line entry point: `python -m htru2_pipeline` or `htru2-pipeline`."""
from __future__ import annotations

import argparse
import logging
import sys

from .config import load_config
from .ingest import DataIngestionError
from .pipeline import run_pipeline
from .validate import DataValidationError

logger = logging.getLogger("htru2_pipeline")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="htru2-pipeline",
        description="Train, evaluate and (conditionally) promote an HTRU2 pulsar classifier.",
    )
    p.add_argument("--config", default="config/config.yaml", help="path to the YAML config")
    p.add_argument("--refresh", action="store_true", help="re-download the dataset")
    p.add_argument("--skip-kmeans", action="store_true", help="skip the k-means experiments")
    p.add_argument("--skip-tsne", action="store_true", help="skip the t-SNE plot")
    p.add_argument("--skip-uncertainty", action="store_true",
                   help="skip the bootstrap confidence intervals")
    p.add_argument("--spark-parity", action="store_true",
                   help="also run the Spark MLlib parity check (needs Java + pyspark)")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    try:
        cfg = load_config(args.config)
        summary = run_pipeline(
            cfg,
            refresh=args.refresh,
            skip_kmeans=args.skip_kmeans,
            skip_tsne=args.skip_tsne,
            skip_uncertainty=args.skip_uncertainty,
            spark_parity=args.spark_parity,
        )
    except (DataIngestionError, DataValidationError) as exc:
        logger.error("%s", exc)
        return 1

    m = summary["test_metrics"]
    print("\n=== Full-data model, held-out test set ===")
    print(f"ROC-AUC {m['roc_auc']}  PR-AUC {m['pr_auc']}  accuracy {m['accuracy']}")
    print(f"Pulsar class: {m['positive']}")
    print(f"Promotion: {summary['promotion']['promoted']} ({summary['promotion']['reason']})")
    print(f"Full results: {cfg.reports_dir / 'results.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
