from dataclasses import dataclass


@dataclass
class IGConfig:
    n_steps: int = 50
    baseline_value: float = -1.0


@dataclass
class OcclusionConfig:
    strides: tuple = (3, 8, 8)
    sliding_window_shapes: tuple = (
        0.01,
        0.03,
        0.05,
        0.075,
        0.10,
        0.15,
        0.20,
        0.30,
        0.50,
        0.80,
    )
    baseline_value: float = -1.0


@dataclass
class ExperimentConfig:
    gaussian_sigmas: tuple[float] = (0.01, 0.05, 0.1, 0.5)
    fgsm_epsilons: tuple[float] = (
        0.01,
        0.03,
        0.05,
        0.075,
        0.10,
        0.15,
        0.20,
        0.30,
        0.40,
        0.50,
    )
    top_k_percent: float = 0.10
    deletion_steps: int = 20
    baseline_value: float = -1.0


@dataclass
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 0.001
