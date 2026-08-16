from collections import Counter

from src.data.cifar10 import get_cifar10_loaders
from src.data.federated_continual import (
    create_federated_continual_tasks
)
from src.algorithms.client_risk import (
    calculate_shift_score
)
from src.utils.seed import set_seed


# --------------------------------
# Configuration
# --------------------------------

SEED = 42
NUM_CLIENTS = 5
NUM_CLASSES = 10
BATCH_SIZE = 64
ALPHA = 0.5

set_seed(SEED)


# --------------------------------
# Load CIFAR-10
# --------------------------------

train_loader, _ = get_cifar10_loaders(
    batch_size=BATCH_SIZE
)

train_dataset = train_loader.dataset


# --------------------------------
# Continual tasks
# --------------------------------

class_groups = [
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
    [8, 9]
]


# --------------------------------
# Create federated continual tasks
# --------------------------------

federated_tasks = create_federated_continual_tasks(
    dataset=train_dataset,
    num_clients=NUM_CLIENTS,
    class_groups=class_groups,
    batch_size=BATCH_SIZE,
    alpha=ALPHA,
    seed=SEED
)


# --------------------------------
# Helper function
# --------------------------------

def get_class_counts(loader):
    counts = Counter()

    for _, labels in loader:
        for label in labels.tolist():
            counts[int(label)] += 1

    return [
        counts[class_id]
        for class_id in range(NUM_CLASSES)
    ]


# --------------------------------
# Store client distributions
# --------------------------------

client_task_counts = [
    []
    for _ in range(NUM_CLIENTS)
]


for task_id, client_loaders in enumerate(
    federated_tasks
):

    for client_id, loader in enumerate(
        client_loaders
    ):

        counts = get_class_counts(
            loader
        )

        client_task_counts[
            client_id
        ].append(
            counts
        )


# --------------------------------
# Print distributions
# --------------------------------

print()
print("Client Class Distributions")
print("=" * 75)

for client_id in range(NUM_CLIENTS):

    print()
    print(f"Client {client_id}")

    for task_id, counts in enumerate(
        client_task_counts[client_id],
        start=1
    ):

        print(
            f"Task {task_id}: "
            f"{counts}"
        )


# --------------------------------
# Consecutive-task shift
# --------------------------------

print()
print("Consecutive-Task Distribution Shift")
print("=" * 75)

all_shift_scores = []

for client_id in range(NUM_CLIENTS):

    print()
    print(f"Client {client_id}")

    client_scores = []

    for task_id in range(
        1,
        len(class_groups)
    ):

        previous_counts = (
            client_task_counts[
                client_id
            ][task_id - 1]
        )

        current_counts = (
            client_task_counts[
                client_id
            ][task_id]
        )

        shift_score = calculate_shift_score(
            previous_counts,
            current_counts
        )

        client_scores.append(
            shift_score
        )

        all_shift_scores.append(
            shift_score
        )

        print(
            f"Task {task_id} -> "
            f"Task {task_id + 1}: "
            f"{shift_score:.4f}"
        )

    average_shift = (
        sum(client_scores)
        / len(client_scores)
    )

    print(
        f"Average Shift: "
        f"{average_shift:.4f}"
    )


# --------------------------------
# Overall diagnostics
# --------------------------------

print()
print("Shift Diagnostic Summary")
print("=" * 75)

minimum_shift = min(
    all_shift_scores
)

maximum_shift = max(
    all_shift_scores
)

average_shift = (
    sum(all_shift_scores)
    / len(all_shift_scores)
)

print(
    f"Minimum Shift: "
    f"{minimum_shift:.4f}"
)

print(
    f"Maximum Shift: "
    f"{maximum_shift:.4f}"
)

print(
    f"Average Shift: "
    f"{average_shift:.4f}"
)

unique_rounded_scores = {
    round(score, 4)
    for score in all_shift_scores
}

print(
    f"Number of unique rounded scores: "
    f"{len(unique_rounded_scores)}"
)


# --------------------------------
# Saturation warning
# --------------------------------

near_maximum = sum(
    1
    for score in all_shift_scores
    if score >= 0.99
)

saturation_ratio = (
    near_maximum
    / len(all_shift_scores)
)

print(
    f"Scores >= 0.99: "
    f"{near_maximum}/"
    f"{len(all_shift_scores)}"
)

print(
    f"Saturation Ratio: "
    f"{100.0 * saturation_ratio:.2f}%"
)

if saturation_ratio >= 0.8:
    print()
    print(
        "WARNING: Class-distribution shift "
        "is strongly saturated."
    )

    print(
        "A different shift estimator should "
        "be used for adaptive retention."
    )

else:
    print()
    print(
        "Class-distribution shift contains "
        "useful variation."
    )