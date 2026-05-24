from dataclasses import dataclass


@dataclass
class IGConfig:
    n_steps: int = 200
    baseline_value: float = -1.0


@dataclass
class OcclusionConfig:
    strides: tuple = (3, 8, 8)
    sliding_window_shapes: tuple = (3, 24, 24)
    baseline_value: float = -1.0


@dataclass
class ExperimentConfig:
    gaussian_sigmas: tuple[float] = (0.01, 0.05, 0.1, 0.5)
    top_k_percent: float = 0.10
    deletion_steps: int = 20
    baseline_value: float = -1.0


@dataclass
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 0.001
