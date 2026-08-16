import numpy as np
from src.data.cifar10 import get_cifar10_loaders
from src.data.federated_continual import (
    create_federated_continual_tasks
)


train_loader, _ = get_cifar10_loaders(
    batch_size=64
)

train_dataset = train_loader.dataset


class_groups = [
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
    [8, 9]
]


federated_tasks = create_federated_continual_tasks(
    dataset=train_dataset,
    num_clients=5,
    class_groups=class_groups,
    batch_size=64,
    alpha=0.5,
    seed=42
)


print(
    "Number of continual tasks:",
    len(federated_tasks)
)

print()


for task_id, client_loaders in enumerate(
    federated_tasks,
    start=1
):
    print(
        f"Task {task_id} | "
        f"Classes: {class_groups[task_id - 1]}"
    )

    task_total = 0

    for client_id, loader in enumerate(
        client_loaders
    ):
        num_samples = len(loader.dataset)
        indices = loader.dataset.indices

        labels = np.array(
            train_dataset.targets
        )[indices]

        class_counts = np.bincount(
            labels,
            minlength=10
        )
        task_total += num_samples

        print(
            f"  Client {client_id}: "
            f"{num_samples} samples | "
            f"Class counts: {class_counts.tolist()}"
        )

    print(
        f"  Total Task Samples: "
        f"{task_total}"
    )

    print()