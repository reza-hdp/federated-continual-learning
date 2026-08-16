import numpy as np

from torch.utils.data import DataLoader, Subset


def create_federated_continual_tasks(
    dataset,
    num_clients=5,
    class_groups=None,
    batch_size=64,
    alpha=0.5,
    seed=42,
    min_samples_per_client=1,
    max_partition_attempts=1000
):
    """
    Create federated class-incremental tasks using
    Dirichlet non-IID partitioning.

    For every continual task, the Dirichlet split is
    resampled until every client receives at least
    min_samples_per_client samples.

    This prevents empty-client DataLoader errors when
    using strongly heterogeneous settings such as
    alpha=0.1.
    """

    if class_groups is None:
        class_groups = [
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
            [8, 9]
        ]

    if num_clients < 1:
        raise ValueError(
            "num_clients must be at least 1"
        )

    if alpha <= 0:
        raise ValueError(
            "alpha must be greater than 0"
        )

    if min_samples_per_client < 1:
        raise ValueError(
            "min_samples_per_client "
            "must be at least 1"
        )

    if max_partition_attempts < 1:
        raise ValueError(
            "max_partition_attempts "
            "must be at least 1"
        )

    rng = np.random.default_rng(
        seed
    )

    targets = np.asarray(
        dataset.targets
    )

    federated_tasks = []

    # ========================================================
    # Create each continual task
    # ========================================================

    for task_id, classes in enumerate(
        class_groups
    ):

        successful_partition = False
        task_client_indices = None

        # ====================================================
        # Resample until every client has data
        # ====================================================

        for attempt in range(
            1,
            max_partition_attempts + 1
        ):

            candidate_client_indices = [
                []
                for _ in range(
                    num_clients
                )
            ]

            # ================================================
            # Partition each class independently
            # ================================================

            for class_id in classes:

                class_indices = np.where(
                    targets == class_id
                )[0].copy()

                rng.shuffle(
                    class_indices
                )

                proportions = rng.dirichlet(
                    np.full(
                        num_clients,
                        alpha,
                        dtype=float
                    )
                )

                split_points = (
                    np.cumsum(
                        proportions
                    )[:-1]
                    * len(
                        class_indices
                    )
                ).astype(
                    int
                )

                splits = np.split(
                    class_indices,
                    split_points
                )

                for client_id, split in enumerate(
                    splits
                ):

                    candidate_client_indices[
                        client_id
                    ].extend(
                        split.tolist()
                    )

            # ================================================
            # Check client sample counts
            # ================================================

            client_sample_counts = [
                len(indices)
                for indices
                in candidate_client_indices
            ]

            valid_partition = all(
                sample_count
                >= min_samples_per_client
                for sample_count
                in client_sample_counts
            )

            if valid_partition:

                task_client_indices = (
                    candidate_client_indices
                )

                successful_partition = True

                break

        # ====================================================
        # Safety check
        # ====================================================

        if not successful_partition:

            raise RuntimeError(
                f"Unable to create a valid "
                f"Dirichlet partition for "
                f"Task {task_id + 1} after "
                f"{max_partition_attempts} attempts. "
                f"Try increasing alpha or reducing "
                f"min_samples_per_client."
            )

        # ====================================================
        # Show partition information
        # ====================================================

        print()

        print(
            f"Task {task_id + 1} "
            f"Dirichlet Partition "
            f"(alpha={alpha})"
        )

        print("-" * 60)

        for client_id, indices in enumerate(
            task_client_indices
        ):

            client_targets = targets[
                indices
            ]

            class_counts = {
                int(class_id): int(
                    np.sum(
                        client_targets
                        == class_id
                    )
                )
                for class_id in classes
            }

            print(
                f"Client {client_id} | "
                f"Samples: {len(indices)} | "
                f"Classes: {class_counts}"
            )

        # ====================================================
        # Create DataLoaders
        # ====================================================

        client_loaders = []

        for client_id in range(
            num_clients
        ):

            indices = list(
                task_client_indices[
                    client_id
                ]
            )

            rng.shuffle(
                indices
            )

            client_dataset = Subset(
                dataset,
                indices
            )

            client_loader = DataLoader(
                client_dataset,
                batch_size=batch_size,
                shuffle=True
            )

            client_loaders.append(
                client_loader
            )

        federated_tasks.append(
            client_loaders
        )

    return federated_tasks