import copy
import random

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset


# ============================================================
# Buffer dataset
# ============================================================

class BufferDataset(Dataset):

    def __init__(
        self,
        samples
    ):
        self.samples = samples

    def __len__(
        self
    ):
        return len(
            self.samples
        )

    def __getitem__(
        self,
        index
    ):
        image, label = (
            self.samples[
                index
            ]
        )

        return (
            image,
            label
        )


# ============================================================
# Fed-A-GEM client
# ============================================================

class FedAGEMClient:

    def __init__(
        self,
        client_id,
        device,
        memory_size=500,
        seed=42
    ):
        self.client_id = (
            client_id
        )

        self.device = (
            device
        )

        self.memory_size = (
            memory_size
        )

        self.seed = (
            seed
        )

        self.buffer = []

        self.num_seen_samples = 0

        self.rng = random.Random(
            seed + client_id
        )


    # ========================================================
    # Reservoir sampling
    # ========================================================

    def update_buffer(
        self,
        dataset
    ):
        """
        Add samples using reservoir sampling.

        Each sample seen over the continual stream has an
        equal probability of being retained in the fixed-size
        memory buffer.
        """

        for index in range(
            len(dataset)
        ):

            image, label = (
                dataset[index]
            )

            image = (
                image
                .detach()
                .cpu()
                .clone()
            )

            label = int(
                label
            )

            self.num_seen_samples += 1

            # ------------------------------------------------
            # Buffer not full yet
            # ------------------------------------------------

            if len(
                self.buffer
            ) < self.memory_size:

                self.buffer.append(
                    (
                        image,
                        label
                    )
                )

                continue


            # ------------------------------------------------
            # Standard reservoir replacement
            # ------------------------------------------------

            replacement_index = (
                self.rng.randint(
                    0,
                    self.num_seen_samples - 1
                )
            )

            if (
                replacement_index
                < self.memory_size
            ):

                self.buffer[
                    replacement_index
                ] = (
                    image,
                    label
                )


    # ========================================================
    # Buffer information
    # ========================================================

    def buffer_size(
        self
    ):
        return len(
            self.buffer
        )


    # ========================================================
    # Compute local buffer gradient
    # ========================================================

    def compute_buffer_gradient(
        self,
        global_model,
        batch_size=64
    ):
        """
        Compute the mean gradient of the global model on this
        client's replay buffer.

        Returns:
            flattened_gradient or None
        """

        if len(
            self.buffer
        ) == 0:

            return None


        model = copy.deepcopy(
            global_model
        ).to(
            self.device
        )

        model.train()

        criterion = (
            nn.CrossEntropyLoss()
        )


        buffer_dataset = (
            BufferDataset(
                self.buffer
            )
        )

        buffer_loader = (
            DataLoader(
                buffer_dataset,
                batch_size=batch_size,
                shuffle=False
            )
        )


        # ----------------------------------------------------
        # Clear gradients
        # ----------------------------------------------------

        model.zero_grad(
            set_to_none=True
        )


        total_samples = len(
            buffer_dataset
        )


        # ----------------------------------------------------
        # Compute exact mean gradient over buffer
        # ----------------------------------------------------

        for images, labels in (
            buffer_loader
        ):

            images = images.to(
                self.device
            )

            labels = labels.to(
                self.device
            )

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )


            batch_fraction = (
                images.size(0)
                / total_samples
            )

            weighted_loss = (
                loss
                * batch_fraction
            )

            weighted_loss.backward()


        # ----------------------------------------------------
        # Flatten gradient
        # ----------------------------------------------------

        gradient_parts = []

        for parameter in (
            model.parameters()
        ):

            if parameter.grad is None:

                gradient_parts.append(
                    torch.zeros_like(
                        parameter
                    ).view(-1)
                )

            else:

                gradient_parts.append(
                    parameter.grad
                    .detach()
                    .clone()
                    .view(-1)
                )


        flattened_gradient = (
            torch.cat(
                gradient_parts
            )
        )


        return (
            flattened_gradient
        )


    # ========================================================
    # Replace model gradients with projected gradient
    # ========================================================

    @staticmethod
    def _write_flat_gradient(
        model,
        flat_gradient
    ):
        offset = 0

        for parameter in (
            model.parameters()
        ):

            number_of_values = (
                parameter.numel()
            )

            gradient_slice = (
                flat_gradient[
                    offset:
                    offset + number_of_values
                ]
            )

            gradient_slice = (
                gradient_slice
                .view_as(
                    parameter
                )
            )

            if parameter.grad is None:

                parameter.grad = (
                    gradient_slice
                    .detach()
                    .clone()
                )

            else:

                parameter.grad.copy_(
                    gradient_slice
                )

            offset += (
                number_of_values
            )


    # ========================================================
    # Flatten current model gradient
    # ========================================================

    @staticmethod
    def _flatten_model_gradient(
        model
    ):
        gradient_parts = []

        for parameter in (
            model.parameters()
        ):

            if parameter.grad is None:

                gradient_parts.append(
                    torch.zeros_like(
                        parameter
                    ).view(-1)
                )

            else:

                gradient_parts.append(
                    parameter.grad
                    .view(-1)
                )


        return torch.cat(
            gradient_parts
        )


    # ========================================================
    # Fed-A-GEM local training
    # ========================================================

    def train(
        self,
        global_model,
        current_loader,
        global_reference_gradient=None,
        local_epochs=1,
        learning_rate=0.01
    ):
        """
        Train on current-task data.

        If the current gradient conflicts with the global
        reference gradient, apply the A-GEM projection:

            g_projected =
                g
                - (g dot g_ref / ||g_ref||^2)
                  * g_ref

        Projection is used only when:

            g dot g_ref < 0
        """

        local_model = (
            copy.deepcopy(
                global_model
            )
            .to(
                self.device
            )
        )


        criterion = (
            nn.CrossEntropyLoss()
        )


        optimizer = optim.SGD(
            local_model.parameters(),
            lr=learning_rate
        )


        local_model.train()


        total_loss = 0.0

        correct = 0

        total = 0


        projection_count = 0

        batch_count = 0

        dot_products = []


        # ====================================================
        # Local epochs
        # ====================================================

        for _ in range(
            local_epochs
        ):

            for images, labels in (
                current_loader
            ):

                batch_count += 1


                images = images.to(
                    self.device
                )

                labels = labels.to(
                    self.device
                )


                optimizer.zero_grad(
                    set_to_none=True
                )


                outputs = local_model(
                    images
                )


                loss = criterion(
                    outputs,
                    labels
                )


                loss.backward()


                # ============================================
                # Fed-A-GEM projection
                # ============================================

                if (
                    global_reference_gradient
                    is not None
                ):

                    current_gradient = (
                        self._flatten_model_gradient(
                            local_model
                        )
                    )


                    reference_gradient = (
                        global_reference_gradient
                        .to(
                            self.device
                        )
                    )


                    dot_product = (
                        torch.dot(
                            current_gradient,
                            reference_gradient
                        )
                    )


                    dot_products.append(
                        float(
                            dot_product.item()
                        )
                    )


                    # ----------------------------------------
                    # Conflict condition
                    # ----------------------------------------

                    if (
                        dot_product.item()
                        < 0.0
                    ):

                        reference_norm_squared = (
                            torch.dot(
                                reference_gradient,
                                reference_gradient
                            )
                        )


                        if (
                            reference_norm_squared.item()
                            > 1e-12
                        ):

                            projection_coefficient = (
                                dot_product
                                / reference_norm_squared
                            )


                            projected_gradient = (
                                current_gradient
                                - projection_coefficient
                                * reference_gradient
                            )


                            self._write_flat_gradient(
                                local_model,
                                projected_gradient
                            )


                            projection_count += 1


                optimizer.step()


                # ============================================
                # Statistics
                # ============================================

                total_loss += (
                    loss.item()
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


                correct += int(
                    torch.count_nonzero(
                        matches
                    ).item()
                )


                total += int(
                    labels.numel()
                )


        # ====================================================
        # Final local statistics
        # ====================================================

        if total > 0:

            average_loss = (
                total_loss
                / total
            )

            accuracy = (
                100.0
                * correct
                / total
            )

        else:

            average_loss = 0.0

            accuracy = 0.0


        if batch_count > 0:

            projection_rate = (
                100.0
                * projection_count
                / batch_count
            )

        else:

            projection_rate = 0.0


        if dot_products:

            average_dot_product = (
                sum(
                    dot_products
                )
                / len(
                    dot_products
                )
            )

        else:

            average_dot_product = 0.0


        diagnostics = {
            "projection_count": (
                projection_count
            ),
            "batch_count": (
                batch_count
            ),
            "projection_rate": (
                projection_rate
            ),
            "average_dot_product": (
                average_dot_product
            ),
            "buffer_size": (
                len(
                    self.buffer
                )
            )
        }


        return (
            local_model.state_dict(),
            average_loss,
            accuracy,
            diagnostics
        )