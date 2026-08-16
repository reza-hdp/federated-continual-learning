import copy

import torch
import torch.nn as nn
import torch.optim as optim

from src.models.cnn import SimpleCNN
from src.data.cifar10 import get_cifar10_loaders
from src.data.federated_continual import (
    create_federated_continual_tasks
)
from src.algorithms.update_conflict import (
    flatten_model_update,
    calculate_reference_update,
    calculate_conflict_score
)
from src.server.server import FederatedServer
from src.utils.seed import set_seed


# --------------------------------
# Configuration
# --------------------------------

SEED = 42
NUM_CLIENTS = 5
BATCH_SIZE = 64
ALPHA = 0.5
ROUNDS_TASK_1 = 3
LEARNING_RATE = 0.001

set_seed(SEED)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)
print("Seed:", SEED)


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
# Create federated tasks
# --------------------------------

federated_tasks = (
    create_federated_continual_tasks(
        dataset=train_dataset,
        num_clients=NUM_CLIENTS,
        class_groups=class_groups,
        batch_size=BATCH_SIZE,
        alpha=ALPHA,
        seed=SEED
    )
)

task_1_loaders = federated_tasks[0]
task_2_loaders = federated_tasks[1]


# --------------------------------
# Model and server
# --------------------------------

initial_global_model = SimpleCNN()

server = FederatedServer(
    global_model=initial_global_model,
    device=device
)

criterion = nn.CrossEntropyLoss()


# --------------------------------
# Copy model state
# --------------------------------

def copy_model_state(model):
    return {
        name: value.detach().cpu().clone()
        for name, value
        in model.state_dict().items()
    }


# --------------------------------
# Local training
# --------------------------------

def train_local_model(
    initial_state,
    data_loader
):
    trained_model = SimpleCNN().to(
        device
    )

    trained_model.load_state_dict(
        initial_state
    )

    optimizer = optim.Adam(
        trained_model.parameters(),
        lr=LEARNING_RATE
    )

    trained_model.train()

    for images, labels in data_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = trained_model(
            images
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()
        optimizer.step()

    return trained_model


# --------------------------------
# Evaluate accuracy
# --------------------------------

def evaluate_accuracy(
    model,
    data_loader
):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in data_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predicted = torch.argmax(
                outputs,
                dim=1
            )

            matches = torch.eq(
                predicted,
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
        return 0.0

    accuracy = (
        100.0
        * correct
        / total
    )

    return accuracy


# --------------------------------
# Learn Task 1 globally
# --------------------------------

print()
print(
    "Training Global Model on Task 1"
)
print("=" * 75)

for round_number in range(
    1,
    ROUNDS_TASK_1 + 1
):

    print(
        f"Federated Round "
        f"{round_number}/"
        f"{ROUNDS_TASK_1}"
    )

    current_global_state = (
        copy_model_state(
            server.global_model
        )
    )

    client_states = []
    client_sizes = []

    for client_id in range(
        NUM_CLIENTS
    ):

        task_1_client_model = (
            train_local_model(
                current_global_state,
                task_1_loaders[
                    client_id
                ]
            )
        )

        task_1_client_state = (
            copy_model_state(
                task_1_client_model
            )
        )

        client_states.append(
            task_1_client_state
        )

        sample_count = len(
            task_1_loaders[
                client_id
            ].dataset
        )

        client_sizes.append(
            sample_count
        )

    server.aggregate(
        client_states,
        client_sizes
    )


# --------------------------------
# State after Task 1
# --------------------------------

task_1_global_state = (
    copy_model_state(
        server.global_model
    )
)


# --------------------------------
# Generate Task 2 updates
# --------------------------------

print()
print(
    "Generating Task 2 Updates"
)
print("=" * 75)

task_2_client_models = []
task_2_updates = []
task_2_sizes = []

for client_id in range(
    NUM_CLIENTS
):

    task_2_client_model = (
        train_local_model(
            task_1_global_state,
            task_2_loaders[
                client_id
            ]
        )
    )

    task_2_client_state = (
        copy_model_state(
            task_2_client_model
        )
    )

    update_vector = (
        flatten_model_update(
            task_1_global_state,
            task_2_client_state
        )
    )

    task_2_client_models.append(
        copy.deepcopy(
            task_2_client_model
        )
    )

    task_2_updates.append(
        update_vector
    )

    sample_count = len(
        task_2_loaders[
            client_id
        ].dataset
    )

    task_2_sizes.append(
        sample_count
    )


# --------------------------------
# Leave-one-out conflicts
# --------------------------------

conflict_scores = []

for client_id in range(
    NUM_CLIENTS
):

    other_updates = [
        update
        for index, update
        in enumerate(
            task_2_updates
        )
        if index != client_id
    ]

    other_sizes = [
        size
        for index, size
        in enumerate(
            task_2_sizes
        )
        if index != client_id
    ]

    other_reference = (
        calculate_reference_update(
            other_updates,
            other_sizes
        )
    )

    conflict = (
        calculate_conflict_score(
            task_2_updates[
                client_id
            ],
            other_reference
        )
    )

    conflict_scores.append(
        float(conflict)
    )


# --------------------------------
# Measure Task-1 forgetting
# --------------------------------

print()
print(
    "Conflict vs Forgetting"
)
print("=" * 75)

forgetting_scores = []

for client_id in range(
    NUM_CLIENTS
):

    task_1_reference_model = (
        SimpleCNN().to(device)
    )

    task_1_reference_model.load_state_dict(
        task_1_global_state
    )

    before_accuracy = (
        evaluate_accuracy(
            task_1_reference_model,
            task_1_loaders[
                client_id
            ]
        )
    )

    after_accuracy = (
        evaluate_accuracy(
            task_2_client_models[
                client_id
            ],
            task_1_loaders[
                client_id
            ]
        )
    )

    forgetting = max(
        0.0,
        before_accuracy
        - after_accuracy
    )

    forgetting_scores.append(
        float(forgetting)
    )

    print(
        f"Client {client_id} | "
        f"Conflict: "
        f"{conflict_scores[client_id]:.4f} | "
        f"Before: "
        f"{before_accuracy:.2f}% | "
        f"After: "
        f"{after_accuracy:.2f}% | "
        f"Forgetting: "
        f"{forgetting:.2f} pp"
    )


# --------------------------------
# Pearson correlation
# --------------------------------

def pearson_correlation(
    values_x,
    values_y
):
    if len(values_x) != len(
        values_y
    ):
        raise ValueError(
            "Input lists must have "
            "the same length."
        )

    if len(values_x) < 2:
        return 0.0

    mean_x = (
        sum(values_x)
        / len(values_x)
    )

    mean_y = (
        sum(values_y)
        / len(values_y)
    )

    numerator = sum(
        (x - mean_x)
        * (y - mean_y)
        for x, y in zip(
            values_x,
            values_y
        )
    )

    squared_x = [
        (x - mean_x) ** 2
        for x in values_x
    ]

    squared_y = [
        (y - mean_y) ** 2
        for y in values_y
    ]

    denominator_x = sum(
        squared_x
    )

    denominator_y = sum(
        squared_y
    )

    denominator = (
        denominator_x
        * denominator_y
    ) ** 0.5

    if denominator == 0.0:
        return 0.0

    return (
        numerator
        / denominator
    )


correlation = (
    pearson_correlation(
        conflict_scores,
        forgetting_scores
    )
)


# --------------------------------
# Summary
# --------------------------------

print()
print(
    "Conflict-Forgetting Summary"
)
print("=" * 75)

print(
    f"Minimum Conflict: "
    f"{min(conflict_scores):.4f}"
)

print(
    f"Maximum Conflict: "
    f"{max(conflict_scores):.4f}"
)

print(
    f"Minimum Forgetting: "
    f"{min(forgetting_scores):.2f} pp"
)

print(
    f"Maximum Forgetting: "
    f"{max(forgetting_scores):.2f} pp"
)

print(
    f"Pearson Correlation: "
    f"{correlation:+.4f}"
)


# --------------------------------
# Interpretation
# --------------------------------

print()

if correlation >= 0.70:

    print(
        "RESULT: Strong positive "
        "relationship between update "
        "conflict and forgetting."
    )

elif correlation >= 0.40:

    print(
        "RESULT: Moderate positive "
        "relationship between update "
        "conflict and forgetting."
    )

elif correlation > 0.0:

    print(
        "RESULT: Weak positive "
        "relationship between update "
        "conflict and forgetting."
    )

else:

    print(
        "RESULT: No positive "
        "relationship was observed "
        "in this diagnostic."
    )