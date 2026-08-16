from torch.utils.data import DataLoader, Subset


def create_iid_clients(dataset, num_clients=5, batch_size=64):
    if num_clients < 1:
        raise ValueError("num_clients must be at least 1")

    indices = list(range(len(dataset)))

    client_size = len(indices) // num_clients

    client_loaders = []

    for client_id in range(num_clients):
        start = client_id * client_size

        if client_id == num_clients - 1:
            end = len(indices)
        else:
            end = start + client_size

        client_indices = indices[start:end]

        client_dataset = Subset(dataset, client_indices)

        client_loader = DataLoader(
            client_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        client_loaders.append(client_loader)

    return client_loaders


import numpy as np

def create_noniid_clients(
    dataset,
    num_clients=5,
    batch_size=64,
    alpha=0.5,
    seed=42
):
    if num_clients < 1:
        raise ValueError("num_clients must be at least 1")

    if alpha <= 0:
        raise ValueError("alpha must be greater than 0")

    rng = np.random.default_rng(seed)

    targets = np.array(dataset.targets)
    num_classes = len(np.unique(targets))

    client_indices = [[] for _ in range(num_clients)]

    for class_id in range(num_classes):
        class_indices = np.where(targets == class_id)[0]

        rng.shuffle(class_indices)

        proportions = rng.dirichlet(
            np.repeat(alpha, num_clients)
        )

        split_points = (
            np.cumsum(proportions)[:-1] * len(class_indices)
        ).astype(int)

        splits = np.split(class_indices, split_points)

        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())

    client_loaders = []

    for indices in client_indices:
        rng.shuffle(indices)

        client_dataset = Subset(dataset, indices)

        client_loader = DataLoader(
            client_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        client_loaders.append(client_loader)

    return client_loaders