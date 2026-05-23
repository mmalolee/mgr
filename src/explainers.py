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

        return process_attribution_raw(attr)


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

        return process_attribution_raw(attr)


def process_attribution_raw(attr_tensor):
    attr_np = attr_tensor.squeeze(0).detach().cpu().numpy()

    if attr_np.ndim == 3:
        attr_np = attr_np.sum(axis=0)

    return attr_np
