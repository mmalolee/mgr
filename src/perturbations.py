import torch
import torch.nn.functional as F


class Perturbations:
    def __init__(self, model, device, clamp_min=-1.0, clamp_max=1.0):
        self.model = model
        self.device = device
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def gaussian(self, input_tensor, sigma):
        x = input_tensor.clone().detach().to(self.device)

        noise = torch.randn_like(x) * sigma
        x_noised = x + noise

        return torch.clamp(
            x_noised,
            self.clamp_min,
            self.clamp_max,
        ).detach()

    def fgsm(self, input_tensor, target_class, epsilon):
        x = input_tensor.clone().detach().to(self.device)
        x.requires_grad = True

        output = self.model(x)

        loss = F.cross_entropy(
            output,
            torch.tensor([target_class], device=self.device),
        )

        self.model.zero_grad()
        loss.backward()

        perturbation = epsilon * x.grad.sign()
        x_adv = x + perturbation

        self.model.zero_grad()

        return torch.clamp(
            x_adv,
            self.clamp_min,
            self.clamp_max,
        ).detach()
