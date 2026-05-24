import numpy as np
import torch
from captum.attr import IntegratedGradients, Occlusion


class IGExplainer:
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        self.ig = IntegratedGradients(model)

    def explain(self, input_tensor, target_class):
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True
        baseline = torch.full_like(
            input_tensor,
            self.config.baseline_value,
        ).to(self.device)

        attr = self.ig.attribute(
            input_tensor,
            baselines=baseline,
            target=target_class,
            n_steps=self.config.n_steps,
        )

        return self.process_raw_attribution(attr)

    def process_raw_attribution(self, attr_tensor):
        attr_np = attr_tensor.squeeze(0).detach().cpu().numpy()

        signed_map = attr_np.sum(axis=0)
        magnitude_map = np.abs(attr_np).sum(axis=0)
        deletion_map = np.clip(signed_map, 0, None)

        return {
            "raw": signed_map,
            "abs": magnitude_map,
            "deletion": deletion_map,
        }


class OcclusionExplainer:
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        self.occlusion = Occlusion(model)

    def explain(self, input_tensor, target_class):
        input_tensor = input_tensor.to(self.device)
        baseline = torch.full_like(
            input_tensor,
            self.config.baseline_value,
        ).to(self.device)

        attr = self.occlusion.attribute(
            input_tensor,
            strides=self.config.strides,
            sliding_window_shapes=self.config.sliding_window_shapes,
            baselines=baseline,
            target=target_class,
        )

        return self.process_raw_attribution(attr)

    def process_raw_attribution(self, attr_tensor):
        attr_np = attr_tensor.squeeze(0).detach().cpu().numpy()

        signed_map = attr_np.sum(axis=0)
        magnitude_map = np.abs(attr_np).sum(axis=0)
        deletion_map = np.clip(signed_map, 0, None)

        return {
            "raw": signed_map,
            "abs": magnitude_map,
            "deletion": deletion_map,
        }
