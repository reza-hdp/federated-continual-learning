import copy
import torch.nn as nn
import torch.optim as optim

from src.utils.training import train_one_epoch


class FederatedClient:
    def __init__(self, client_id, train_loader, device):
        self.client_id = client_id
        self.train_loader = train_loader
        self.device = device

    def train(self, global_model, local_epochs=1, learning_rate=0.001):
        if local_epochs < 1:
            raise ValueError("local_epochs must be at least 1")

        local_model = copy.deepcopy(global_model).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            local_model.parameters(),
            lr=learning_rate
        )

        train_loss = 0.0
        train_accuracy = 0.0

        for _ in range(local_epochs):
            train_loss, train_accuracy = train_one_epoch(
                local_model,
                self.train_loader,
                criterion,
                optimizer,
                self.device
            )

        return local_model.state_dict(), train_loss, train_accuracy