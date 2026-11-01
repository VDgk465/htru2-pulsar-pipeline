"""Promotion gate: a candidate model replaces "production" only if it earns it.

A candidate must (1) clear an absolute metric floor and (2) beat the current
production model by the configured margin. Every decision is appended to an
audit log. Caveat: the gate uses the same held-out test set each run, so in a
real system you would add a separate validation set to avoid optimistic bias.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib

from .config import Config

logger = logging.getLogger(__name__)

ALLOWED_METRICS = ("roc_auc", "pr_auc", "accuracy")


@dataclass
class PromotionDecision:
    promoted: bool
    reason: str
    metric: str
    candidate_value: float
    production_value: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def production_dir(cfg: Config) -> Path:
    return cfg.models_dir / "production"


def load_production_metadata(cfg: Config) -> dict | None:
    path = production_dir(cfg) / "metadata.json"
    return json.loads(path.read_text()) if path.exists() else None


def decide(candidate_metrics: dict, production_meta: dict | None, cfg: Config):
    metric = cfg.promotion.metric
    if metric not in ALLOWED_METRICS:
        raise ValueError(f"promotion.metric must be one of {ALLOWED_METRICS}, got {metric!r}")

    cand = float(candidate_metrics[metric])
    floor, margin = cfg.promotion.min_metric, cfg.promotion.min_improvement
    prod = None if production_meta is None else float(production_meta["metrics"][metric])

    if cand < floor:
        return PromotionDecision(
            False, f"{metric} {cand:.4f} is below the required minimum {floor:.4f}",
            metric, cand, prod,
        )
    if prod is None:
        return PromotionDecision(True, "no production model yet", metric, cand, None)
    if cand > prod + margin:
        return PromotionDecision(
            True, f"{metric} improved from {prod:.4f} to {cand:.4f}", metric, cand, prod,
        )
    return PromotionDecision(
        False,
        f"{metric} {cand:.4f} does not beat production {prod:.4f} by more than {margin}",
        metric, cand, prod,
    )


def promote_if_better(model, metrics: dict, cfg: Config, name: str, params=None):
    """Apply the gate; write the model + metadata only when promoted."""
    decision = decide(metrics, load_production_metadata(cfg), cfg)
    out = production_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if decision.promoted:
        joblib.dump(model, out / "model.joblib")
        (out / "metadata.json").write_text(json.dumps(
            {"name": name, "promoted_at": now, "params": params or {}, "metrics": metrics},
            indent=2, default=str,
        ))
        logger.info("PROMOTED %s: %s", name, decision.reason)
    else:
        logger.info("NOT promoted %s: %s", name, decision.reason)

    with (out / "history.jsonl").open("a") as fh:
        fh.write(json.dumps({"time": now, "candidate_name": name, **decision.to_dict()}) + "\n")
    return decision
