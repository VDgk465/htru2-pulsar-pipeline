from pathlib import Path

from htru2_pipeline.config import load_config


def test_real_config_loads():
    cfg = load_config(Path(__file__).parents[1] / "config" / "config.yaml")
    assert len(cfg.features) == 8
    assert cfg.target == "Target"
    assert cfg.target not in cfg.features
    assert cfg.data.url.endswith("htru2.zip")
    assert cfg.random_state == 42


def test_model_section_loads_with_null_depth():
    cfg = load_config(Path(__file__).parents[1] / "config" / "config.yaml")
    assert cfg.model.max_depth_grid == [10, None]
    assert cfg.model.scoring == "roc_auc"
    assert cfg.model.scale_features is False
    assert cfg.positive_label == 1


def test_pipeline_sections_load():
    cfg = load_config(Path(__file__).parents[1] / "config" / "config.yaml")
    assert cfg.sampling.ks == [5, 10, 25]
    assert cfg.sampling.ratio_mode == "round"
    assert cfg.promotion.metric == "pr_auc"
    assert cfg.spark.max_depth_grid == [10, 30]


def test_evaluation_section_loads():
    cfg = load_config(Path(__file__).parents[1] / "config" / "config.yaml")
    assert cfg.evaluation.bootstrap_samples == 1000
    assert cfg.evaluation.confidence == 0.95
