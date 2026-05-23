import numpy as np
import torch
from sklearn.metrics import auc


class Metrics:
    @staticmethod
    def cosine_similarity(a, b):
        a = a.flatten()
        b = b.flatten()

        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

    @staticmethod
    def topk_iou(a, b, top_k_percent=0.10):
        a = a.flatten()
        b = b.flatten()

        k = int(len(a) * top_k_percent)

        top_a = np.argpartition(a, -k)[-k:]
        top_b = np.argpartition(b, -k)[-k:]

        set_a = set(top_a)
        set_b = set(top_b)

        intersection = len(set_a & set_b)
        union = len(set_a | set_b)

        return intersection / (union + 1e-8)

    @staticmethod
    def deletion_auc(
        model,
        input_tensor,
        attribution_map,
        target_class,
        steps=20,
        baseline_value=-1.0,
    ):
        x_deleted = input_tensor.clone().detach()
        _, C, H, W = x_deleted.shape

        total_pixels = H * W

        attr_flat = attribution_map.flatten()

        importance = np.maximum(attr_flat, 0)

        if np.all(importance == 0):
            importance = np.abs(attr_flat)

        sorted_indices = np.argsort(importance)[::-1]

        confidences = []
        fractions_removed = []

        with torch.no_grad():
            for step in range(steps + 1):
                fraction_removed = step / steps

                output = model(x_deleted)
                prob = torch.softmax(output, dim=1)[0, target_class].item()

                confidences.append(prob)
                fractions_removed.append(fraction_removed)

                if step == steps:
                    break

                start = int((step / steps) * total_pixels)
                end = int(((step + 1) / steps) * total_pixels)

                pixels_to_remove = sorted_indices[start:end]

                rows = pixels_to_remove // W
                cols = pixels_to_remove % W

                x_deleted[:, :, rows, cols] = baseline_value

        return auc(fractions_removed, confidences)
