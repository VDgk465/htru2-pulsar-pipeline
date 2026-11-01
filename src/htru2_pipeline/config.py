"""Typed configuration loaded from YAML, so no magic values live in the code."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DataConfig:
    url: str
    csv_member: str
    cache_path: Path


@dataclass(frozen=True)
class ModelConfig:
    cv_folds: int = 5
    scoring: str = "roc_auc"
    n_jobs: int = 1
    class_weight: str | None = "balanced"
    # Random Forests are scale-invariant, so scaling is off by default.
    scale_features: bool = False
    n_estimators_grid: list[int] = field(default_factory=lambda: [50, 100])
    min_samples_leaf_grid: list[int] = field(default_factory=lambda: [1, 5])
    max_depth_grid: list[int | None] = field(default_factory=lambda: [10, None])


@dataclass(frozen=True)
class SamplingConfig:
    ks: list[int] = field(default_factory=lambda: [5, 10, 25])
    # "round" keeps the negative:positive ratio closest to the real one (~9.9 -> 10);
    # "floor" truncates it (-> 9).
    ratio_mode: str = "round"
    kmeans_n_init: int = 10


@dataclass(frozen=True)
class PromotionConfig:
    metric: str = "pr_auc"
    min_metric: float = 0.80      # absolute floor a candidate must clear
    min_improvement: float = 0.0  # margin required over the current production model


@dataclass(frozen=True)
class EvaluationConfig:
    bootstrap_samples: int = 1000   # resamples of the test set for confidence intervals
    confidence: float = 0.95


@dataclass(frozen=True)
class SparkConfig:
    app_name: str = "htru2-pipeline"
    master: str = "local[*]"
    driver_memory: str = "2g"
    shuffle_partitions: int = 8
    cv_folds: int = 5
    num_trees_grid: list[int] = field(default_factory=lambda: [50, 100])
    min_instances_grid: list[int] = field(default_factory=lambda: [1, 5])
    # Spark caps tree depth at 30, so "unlimited" (None in scikit-learn) becomes 30.
    max_depth_grid: list[int] = field(default_factory=lambda: [10, 30])
    parity_tolerance: float = 0.05


@dataclass(frozen=True)
class Config:
    data: DataConfig
    columns: list[str]
    target: str
    valid_labels: list[int]
    random_state: int
    min_rows: int
    test_size: float = 0.2
    positive_label: int = 1
    label_names: dict[int, str] = field(default_factory=dict)
    reports_dir: Path = Path("reports")
    models_dir: Path = Path("models")
    model: ModelConfig = field(default_factory=ModelConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    promotion: PromotionConfig = field(default_factory=PromotionConfig)
    spark: SparkConfig = field(default_factory=SparkConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @property
    def features(self) -> list[str]:
        return [c for c in self.columns if c != self.target]


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    return Config(
        data=DataConfig(
            url=raw["data"]["url"],
            csv_member=raw["data"]["csv_member"],
            cache_path=Path(raw["data"]["cache_path"]),
        ),
        columns=list(raw["columns"]),
        target=raw["target"],
        valid_labels=[int(x) for x in raw["valid_labels"]],
        random_state=int(raw["random_state"]),
        min_rows=int(raw["validation"]["min_rows"]),
        test_size=float(raw.get("test_size", 0.2)),
        positive_label=int(raw.get("positive_label", 1)),
        label_names={int(k): v for k, v in raw.get("label_names", {}).items()},
        reports_dir=Path(raw.get("reports_dir", "reports")),
        models_dir=Path(raw.get("models_dir", "models")),
        # Unknown keys raise a TypeError, so typos in the YAML fail fast.
        model=ModelConfig(**raw.get("model", {})),
        sampling=SamplingConfig(**raw.get("sampling", {})),
        promotion=PromotionConfig(**raw.get("promotion", {})),
        spark=SparkConfig(**raw.get("spark", {})),
        evaluation=EvaluationConfig(**raw.get("evaluation", {})),
    )
