import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import ConcatDataset, DataLoader, Subset


class ReplayLwFClient:
    def __init__(
        self,
        client_id,
        device,
        memory_size=500,
        seed=42,
        temperature=2.0,
        distillation_weight=1.0
    ):
        self.client_id = client_id
        self.device = device

        self.memory_size = memory_size
        self.seed = seed

        self.temperature = temperature
        self.distillation_weight = (
            distillation_weight
        )

        self.memory_datasets = []

        self.teacher_model = None

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

        self.memory_datasets.append(
            dataset
        )

        num_tasks = len(
            self.memory_datasets
        )

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

    def update_teacher(
        self,
        model
    ):
        self.teacher_model = copy.deepcopy(
            model
        ).to(self.device)

        self.teacher_model.eval()

        for parameter in (
            self.teacher_model.parameters()
        ):
            parameter.requires_grad = False

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

                images = images.to(
                    self.device
                )

                labels = labels.to(
                    self.device
                )

                optimizer.zero_grad()

                outputs = local_model(
                    images
                )

                classification_loss = (
                    criterion(
                        outputs,
                        labels
                    )
                )

                loss = classification_loss

                if self.teacher_model is not None:

                    with torch.no_grad():
                        teacher_outputs = (
                            self.teacher_model(
                                images
                            )
                        )

                    temperature = (
                        self.temperature
                    )

                    teacher_probabilities = (
                        F.softmax(
                            teacher_outputs
                            / temperature,
                            dim=1
                        )
                    )

                    student_log_probabilities = (
                        F.log_softmax(
                            outputs
                            / temperature,
                            dim=1
                        )
                    )

                    distillation_loss = (
                        F.kl_div(
                            student_log_probabilities,
                            teacher_probabilities,
                            reduction="batchmean"
                        )
                        * temperature ** 2
                    )

                    loss = (
                        classification_loss
                        + self.distillation_weight
                        * distillation_loss
                    )

                loss.backward()
                optimizer.step()

                total_loss += (
                    classification_loss.item()
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

        average_loss = (
            total_loss / total
        )

        accuracy = (
            100.0 * correct / total
        )

        return (
            local_model.state_dict(),
            average_loss,
            accuracy
        )