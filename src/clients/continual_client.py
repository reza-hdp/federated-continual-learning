import copy
import random

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import ConcatDataset, DataLoader, Subset


class ContinualFederatedClient:
    def __init__(
        self,
        client_id,
        device,
        memory_size=500,
        seed=42
    ):
        self.client_id = client_id
        self.device = device

        self.memory_size = memory_size
        self.seed = seed

        self.memory_datasets = []

    def add_memory(
            self,
            dataset
    ):
        if len(dataset) == 0:
            return

        rng = random.Random(
            self.seed
            + self.client_id
            + len(self.memory_datasets)
        )

        # Add the new task temporarily.
        self.memory_datasets.append(
            dataset
        )

        num_tasks = len(
            self.memory_datasets
        )

        # MEMORY_SIZE is now the TOTAL memory
        # budget for this client.
        samples_per_task = (
                self.memory_size // num_tasks
        )

        new_memories = []

        for task_dataset in self.memory_datasets:
            num_samples = min(
                samples_per_task,
                len(task_dataset)
            )

            indices = rng.sample(
                range(len(task_dataset)),
                num_samples
            )

            memory_subset = Subset(
                task_dataset,
                indices
            )

            new_memories.append(
                memory_subset
            )

        self.memory_datasets = new_memories

    def create_replay_loader(
        self,
        current_dataset,
        batch_size=64
    ):
        datasets = [
            current_dataset
        ]

        datasets.extend(
            self.memory_datasets
        )

        combined_dataset = ConcatDataset(
            datasets
        )

        return DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=True
        )

    def train(
        self,
        global_model,
        train_loader,
        local_epochs=1,
        learning_rate=0.001
    ):
        local_model = copy.deepcopy(
            global_model
        ).to(self.device)

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(
            local_model.parameters(),
            lr=learning_rate
        )

        local_model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for _ in range(local_epochs):

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                outputs = local_model(images)

                loss = criterion(
                    outputs,
                    labels
                )

                loss.backward()
                optimizer.step()

                total_loss += (
                    loss.item()
                    * images.size(0)
                )

                _, predicted = torch.max(
                    outputs,
                    dim=1
                )

                total += labels.size(0)

                matches = torch.eq(
                    predicted,
                    labels
                )

                correct += torch.sum(
                    matches
                ).item()

        average_loss = total_loss / total
        accuracy = 100.0 * correct / total

        return (
            local_model.state_dict(),
            average_loss,
            accuracy
        )