import torch


class EWC:
    def __init__(self, model, data_loader, device):
        self.device = device

        self.parameters = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

        self.fisher = self._compute_fisher(
            model,
            data_loader
        )

    def _compute_fisher(self, model, data_loader):
        fisher = {
            name: torch.zeros_like(parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

        model.eval()

        num_batches = 0

        for images, labels in data_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            model.zero_grad()

            outputs = model(images)

            loss = torch.nn.functional.cross_entropy(
                outputs,
                labels
            )

            loss.backward()

            for name, parameter in model.named_parameters():
                if (
                    parameter.requires_grad
                    and parameter.grad is not None
                ):
                    fisher[name] += (
                        parameter.grad.detach() ** 2
                    )

            num_batches += 1

        if num_batches > 0:
            for name in fisher:
                fisher[name] /= num_batches

        return fisher

    def penalty(self, model):
        loss = torch.tensor(
            0.0,
            device=self.device
        )

        for name, parameter in model.named_parameters():
            if name in self.fisher:
                loss += (
                    self.fisher[name]
                    * (
                        parameter
                        - self.parameters[name]
                    ) ** 2
                ).sum()

        return loss