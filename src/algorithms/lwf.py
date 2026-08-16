import copy

import torch
import torch.nn.functional as F


class LwF:
    def __init__(
        self,
        model,
        temperature=2.0,
        alpha=1.0
    ):
        self.old_model = copy.deepcopy(model)

        self.old_model.eval()

        for parameter in self.old_model.parameters():
            parameter.requires_grad = False

        self.temperature = temperature
        self.alpha = alpha

    def distillation_loss(self, model, images):
        with torch.no_grad():
            old_outputs = self.old_model(images)

        new_outputs = model(images)

        temperature = self.temperature

        old_probabilities = F.softmax(
            old_outputs / temperature,
            dim=1
        )

        new_log_probabilities = F.log_softmax(
            new_outputs / temperature,
            dim=1
        )

        loss = F.kl_div(
            new_log_probabilities,
            old_probabilities,
            reduction="batchmean"
        )

        return (
            self.alpha
            * (temperature ** 2)
            * loss
        )