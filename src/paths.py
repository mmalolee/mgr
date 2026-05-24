from pathlib import Path


class Paths:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODEL_DIR = PROJECT_ROOT / "models"
    NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"

    MNIST_MODEL = MODEL_DIR / "resnet18_mnist.pth"
    CIFAR10_MODEL = MODEL_DIR / "resnet18_cifar10.pth"
