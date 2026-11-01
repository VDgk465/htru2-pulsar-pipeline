import json
from dataclasses import replace

import joblib
import pytest

from htru2_pipeline.config import PromotionConfig
from htru2_pipeline.promotion import decide, load_production_metadata, promote_if_better


def m(v):
    return {"roc_auc": v, "pr_auc": v, "accuracy": v}


def with_promotion(cfg, **kw):
    return replace(cfg, promotion=PromotionConfig(**kw))


def test_first_candidate_is_promoted(small_cfg):
    d = decide(m(0.9), None, small_cfg)
    assert d.promoted and "no production model" in d.reason


def test_candidate_below_floor_is_rejected(small_cfg):
    cfg = with_promotion(small_cfg, min_metric=0.85)
    d = decide(m(0.80), None, cfg)
    assert not d.promoted and "below the required minimum" in d.reason


def test_candidate_must_beat_production(small_cfg):
    prod = {"metrics": m(0.90)}
    assert not decide(m(0.90), prod, small_cfg).promoted  # a tie is not an improvement
    assert not decide(m(0.85), prod, small_cfg).promoted
    assert decide(m(0.95), prod, small_cfg).promoted


def test_improvement_margin_is_respected(small_cfg):
    cfg = with_promotion(small_cfg, min_metric=0.0, min_improvement=0.02)
    prod = {"metrics": m(0.90)}
    assert not decide(m(0.91), prod, cfg).promoted
    assert decide(m(0.93), prod, cfg).promoted


def test_unknown_metric_is_rejected(small_cfg):
    cfg = with_promotion(small_cfg, metric="vibes")
    with pytest.raises(ValueError, match="promotion.metric"):
        decide(m(0.9), None, cfg)


def test_promote_if_better_writes_model_metadata_and_history(small_cfg):
    prod = small_cfg.models_dir / "production"

    first = promote_if_better({"model": 1}, m(0.90), small_cfg, name="v1", params={"a": 1})
    assert first.promoted
    assert joblib.load(prod / "model.joblib") == {"model": 1}
    assert load_production_metadata(small_cfg)["name"] == "v1"

    second = promote_if_better({"model": 2}, m(0.90), small_cfg, name="v2")
    assert not second.promoted
    assert joblib.load(prod / "model.joblib") == {"model": 1}  # untouched
    assert load_production_metadata(small_cfg)["name"] == "v1"

    third = promote_if_better({"model": 3}, m(0.95), small_cfg, name="v3")
    assert third.promoted
    assert joblib.load(prod / "model.joblib") == {"model": 3}

    log = [json.loads(line) for line in (prod / "history.jsonl").read_text().splitlines()]
    assert [e["candidate_name"] for e in log] == ["v1", "v2", "v3"]
    assert [e["promoted"] for e in log] == [True, False, True]
    assert log[1]["candidate_value"] == 0.9 and log[1]["production_value"] == 0.9
