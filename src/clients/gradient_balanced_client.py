import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim

from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Subset
)

from src.algorithms.gradient_balanced_retention import (
    GradientBalancedRetention
)


class GradientBalancedClient:
    def __init__(
        self,
        client_id,
        device,
        memory_size=500,
        seed=42,
        temperature=2.0,
        min_weight=0.2,
        max_weight=2.0
    ):
        self.client_id = client_id
        self.device = device

        self.memory_size = memory_size
        self.seed = seed

        self.temperature = temperature

        self.memory_datasets = []
        self.teacher_model = None

        self.retention_controller = (
            GradientBalancedRetention(
                min_weight=min_weight,
                max_weight=max_weight
            )
        )

    # ========================================================
    # Fixed total replay-memory budget
    # ========================================================

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
            self.memory_size
            // num_tasks
        )

        new_memories = []

        for task_dataset in (
            self.memory_datasets
        ):
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

        self.memory_datasets = (
            new_memories
        )

    # ========================================================
    # Replay loader
    # ========================================================

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

        combined_dataset = (
            ConcatDataset(
                datasets
            )
        )

        return DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=True
        )

    # ========================================================
    # Old-memory loader
    # ========================================================

    def create_memory_loader(
        self,
        batch_size=64
    ):
        if not self.memory_datasets:
            return None

        memory_dataset = (
            ConcatDataset(
                self.memory_datasets
            )
        )

        return DataLoader(
            memory_dataset,
            batch_size=batch_size,
            shuffle=True
        )

    # ========================================================
    # Teacher update
    # ========================================================

    def update_teacher(
        self,
        model
    ):
        self.teacher_model = (
            copy.deepcopy(
                model
            ).to(
                self.device
            )
        )

        self.teacher_model.eval()

        for parameter in (
            self.teacher_model.parameters()
        ):
            parameter.requires_grad = False

    # ========================================================
    # Calculate gradient norm
    # ========================================================

    def calculate_gradient_norm(
        self,
        model,
        data_loader
    ):
        if data_loader is None:
            return 0.0

        gradient_model = (
            copy.deepcopy(
                model
            ).to(
                self.device
            )
        )

        gradient_model.train()

        criterion = (
            nn.CrossEntropyLoss()
        )

        gradient_model.zero_grad(
            set_to_none=True
        )

        total_samples = 0

        for images, labels in (
            data_loader
        ):
            images = images.to(
                self.device
            )

            labels = labels.to(
                self.device
            )

            outputs = gradient_model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

            current_batch_size = int(
                labels.numel()
            )

            weighted_loss = (
                loss
                * current_batch_size
            )

            weighted_loss.backward()

            total_samples += (
                current_batch_size
            )

        if total_samples == 0:
            return 0.0

        squared_norm = 0.0

        for parameter in (
            gradient_model.parameters()
        ):
            if parameter.grad is None:
                continue

            parameter.grad.div_(
                float(total_samples)
            )

            parameter_squared_norm = (
                torch.sum(
                    parameter.grad.detach()
                    ** 2
                ).item()
            )

            squared_norm += float(
                parameter_squared_norm
            )

        gradient_norm = (
            squared_norm ** 0.5
        )

        return float(
            gradient_norm
        )

    # ========================================================
    # Calculate adaptive retention weight
    # ========================================================

    def calculate_retention_weight(
        self,
        global_model,
        current_loader,
        batch_size=64
    ):
        memory_loader = (
            self.create_memory_loader(
                batch_size=batch_size
            )
        )

        if memory_loader is None:
            return {
                "old_gradient_norm": 0.0,
                "new_gradient_norm": 0.0,
                "magnitude_ratio": 0.0,
                "balance_score": 0.0,
                "retention_weight": 0.0
            }

        old_gradient_norm = (
            self.calculate_gradient_norm(
                global_model,
                memory_loader
            )
        )

        new_gradient_norm = (
            self.calculate_gradient_norm(
                global_model,
                current_loader
            )
        )

        diagnostics = (
            self.retention_controller
            .calculate_diagnostics(
                old_gradient_norm,
                new_gradient_norm
            )
        )

        return diagnostics

    # ========================================================
    # Local training
    # ========================================================

    def train(
        self,
        global_model,
        current_loader,
        training_loader,
        local_epochs=1,
        learning_rate=0.001,
        batch_size=64
    ):
        local_model = (
            copy.deepcopy(
                global_model
            ).to(
                self.device
            )
        )

        criterion = (
            nn.CrossEntropyLoss()
        )

        optimizer = optim.Adam(
            local_model.parameters(),
            lr=learning_rate
        )

        # ----------------------------------------------------
        # Calculate adaptive weight before local optimization
        # ----------------------------------------------------

        if (
            self.teacher_model is None
            or not self.memory_datasets
        ):
            diagnostics = {
                "old_gradient_norm": 0.0,
                "new_gradient_norm": 0.0,
                "magnitude_ratio": 0.0,
                "balance_score": 0.0,
                "retention_weight": 0.0
            }

        else:
            diagnostics = (
                self.calculate_retention_weight(
                    global_model,
                    current_loader,
                    batch_size=batch_size
                )
            )

        retention_weight = (
            diagnostics[
                "retention_weight"
            ]
        )

        local_model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for _ in range(
            local_epochs
        ):
            for images, labels in (
                training_loader
            ):
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

                loss = (
                    classification_loss
                )

                if (
                    self.teacher_model
                    is not None
                    and retention_weight > 0.0
                ):
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
                        functional.softmax(
                            teacher_outputs
                            / temperature,
                            dim=1
                        )
                    )

                    student_log_probabilities = (
                        functional.log_softmax(
                            outputs
                            / temperature,
                            dim=1
                        )
                    )

                    distillation_loss = (
                        functional.kl_div(
                            student_log_probabilities,
                            teacher_probabilities,
                            reduction="batchmean"
                        )
                        * temperature ** 2
                    )

                    loss = (
                        classification_loss
                        + retention_weight
                        * distillation_loss
                    )

                loss.backward()
                optimizer.step()

                total_loss += (
                    classification_loss.item()
                    * images.size(0)
                )

                predictions = (
                    torch.argmax(
                        outputs,
                        dim=1
                    )
                )

                matches = torch.eq(
                    predictions,
                    labels
                )

                batch_correct = (
                    torch.count_nonzero(
                        matches
                    ).item()
                )

                correct += int(
                    batch_correct
                )

                total += int(
                    labels.numel()
                )

        if total == 0:
            raise ValueError(
                "Training loader contains "
                "no samples."
            )

        average_loss = (
            total_loss
            / total
        )

        accuracy = (
            100.0
            * correct
            / total
        )

        return (
            local_model.state_dict(),
            average_loss,
            accuracy,
            diagnostics
        )