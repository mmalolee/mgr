from dataclasses import dataclass


@dataclass
class OcclusionConfig:
    strides: tuple = (3, 6, 6)
    sliding_window_shapes: tuple = (3, 18, 18)
    baseline_value: float = -1.0


@dataclass
class IGConfig:
    n_steps: int = 50
    baseline_value: float = -1.0


@dataclass
class ExperimentConfig:
    top_k_percent: float = 0.10
    deletion_steps: int = 20
    baseline_value: float = -1.0
