import numpy as np
import torch
import torch.nn as nn
from torchvision import models


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tensor_to_img(tensor):
    img_np = tensor.squeeze(0).detach().cpu().numpy()

    if img_np.ndim == 3:
        if img_np.shape[0] == 1:
            img_np = img_np.squeeze(0)
        else:
            img_np = np.transpose(img_np, (1, 2, 0))

    img_np = (img_np + 1) / 2
    img_np = np.clip(img_np, 0, 1)

    return img_np


def plot_image(ax, img, title, cmap=None):
    ax.imshow(img, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")


def initialize_resnet18(num_classes=10):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes,
    )

    return model
