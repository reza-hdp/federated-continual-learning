import numpy as np
from torch.utils.data import DataLoader, Subset


def create_continual_tasks(
    dataset,
    class_groups,
    batch_size=64,
    shuffle=True
):
    targets = np.array(dataset.targets)

    task_loaders = []

    for classes in class_groups:
        indices = np.where(
            np.isin(targets, classes)
        )[0]

        task_dataset = Subset(
            dataset,
            indices.tolist()
        )

        task_loader = DataLoader(
            task_dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )

        task_loaders.append(task_loader)

    return task_loaders