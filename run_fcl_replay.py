import csv

import torch
import torch.nn as nn

from src.models.cnn import SimpleCNN
from src.data.cifar10 import get_cifar10_loaders
from src.data.continual import create_continual_tasks
from src.data.federated_continual import (
    create_federated_continual_tasks
)
from src.clients.continual_client import (
    ContinualFederatedClient
)
from src.server.server import FederatedServer
from src.utils.training import evaluate
from src.utils.seed import set_seed


# --------------------------------
# Configuration
# --------------------------------

SEED = 2026
NUM_CLIENTS = 5
NUM_TASKS = 5
ROUNDS_PER_TASK = 3
BATCH_SIZE = 64
ALPHA = 0.5
MEMORY_SIZE = 500

set_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)
print("Seed:", SEED)


# --------------------------------
# Load CIFAR-10
# --------------------------------

train_loader, test_loader = get_cifar10_loaders(
    batch_size=BATCH_SIZE
)

train_dataset = train_loader.dataset
test_dataset = test_loader.dataset


# --------------------------------
# Define continual tasks
# --------------------------------

class_groups = [
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
    [8, 9]
]


# --------------------------------
# Create non-IID federated tasks
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
# Create continual test loaders
# --------------------------------

test_task_loaders = create_continual_tasks(
    dataset=test_dataset,
    class_groups=class_groups,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# --------------------------------
# Global model and server
# --------------------------------

global_model = SimpleCNN()

server = FederatedServer(
    global_model=global_model,
    device=device
)

criterion = nn.CrossEntropyLoss()


# --------------------------------
# Persistent replay clients
# --------------------------------

clients = [
    ContinualFederatedClient(
        client_id=client_id,
        device=device,
        memory_size=MEMORY_SIZE,
        seed=SEED
    )
    for client_id in range(NUM_CLIENTS)
]


accuracy_matrix = []


# --------------------------------
# Federated continual learning
# --------------------------------

for current_task in range(NUM_TASKS):

    print()
    print("=" * 50)

    print(
        f"Learning Task {current_task + 1} "
        f"| Classes {class_groups[current_task]}"
    )

    print("=" * 50)

    current_client_loaders = (
        federated_tasks[current_task]
    )

    for round_number in range(
        1,
        ROUNDS_PER_TASK + 1
    ):

        print()

        print(
            f"Federated Round "
            f"{round_number}/{ROUNDS_PER_TASK}"
        )

        client_states = []
        client_sizes = []

        for client_id in range(
            NUM_CLIENTS
        ):

            current_loader = (
                current_client_loaders[
                    client_id
                ]
            )

            if current_task == 0:
                training_loader = (
                    current_loader
                )
            else:
                training_loader = (
                    clients[
                        client_id
                    ].create_replay_loader(
                        current_loader.dataset,
                        batch_size=BATCH_SIZE
                    )
                )

            state, loss, accuracy = (
                clients[client_id].train(
                    global_model=(
                        server.global_model
                    ),
                    train_loader=training_loader,
                    local_epochs=1,
                    learning_rate=0.001
                )
            )

            client_states.append(
                state
            )

            training_size = len(
                training_loader.dataset
            )

            client_sizes.append(
                training_size
            )

            print(
                f"Client {client_id} | "
                f"Training Samples: "
                f"{training_size} | "
                f"Loss: {loss:.4f} | "
                f"Accuracy: {accuracy:.2f}%"
            )

        server.aggregate(
            client_states,
            client_sizes
        )


    # --------------------------------
    # Update fixed replay memory
    # --------------------------------

    for client_id in range(
        NUM_CLIENTS
    ):

        current_dataset = (
            current_client_loaders[
                client_id
            ].dataset
        )

        clients[client_id].add_memory(
            current_dataset
        )


    # --------------------------------
    # Evaluate global model
    # --------------------------------

    print()
    print("Global Model Evaluation:")

    task_accuracies = []

    for test_task in range(
        current_task + 1
    ):

        _, test_accuracy = evaluate(
            server.global_model,
            test_task_loaders[test_task],
            criterion,
            device
        )

        task_accuracies.append(
            test_accuracy
        )

        print(
            f"Task {test_task + 1} "
            f"Accuracy: "
            f"{test_accuracy:.2f}%"
        )

    accuracy_matrix.append(
        task_accuracies
    )


# --------------------------------
# Accuracy matrix
# --------------------------------

print()
print("FCL Replay Accuracy Matrix")

for task_id, accuracies in enumerate(
    accuracy_matrix,
    start=1
):

    formatted = [
        f"{accuracy:.2f}%"
        for accuracy in accuracies
    ]

    print(
        f"After Task {task_id}: "
        f"{formatted}"
    )


# --------------------------------
# Calculate forgetting
# --------------------------------

forgetting_scores = []

for task_id in range(
    len(accuracy_matrix) - 1
):

    best_previous_accuracy = max(
        row[task_id]
        for row in accuracy_matrix
        if len(row) > task_id
    )

    final_accuracy = (
        accuracy_matrix[-1][task_id]
    )

    forgetting = max(
        0.0,
        best_previous_accuracy
        - final_accuracy
    )

    forgetting_scores.append(
        forgetting
    )

    print(
        f"Task {task_id + 1} "
        f"Forgetting: "
        f"{forgetting:.2f} "
        f"percentage points"
    )


average_forgetting = (
    sum(forgetting_scores)
    / len(forgetting_scores)
)

final_average_accuracy = (
    sum(accuracy_matrix[-1])
    / len(accuracy_matrix[-1])
)


print(
    f"Final Average Accuracy: "
    f"{final_average_accuracy:.2f}%"
)

print(
    f"Average Forgetting: "
    f"{average_forgetting:.2f} "
    f"percentage points"
)


# --------------------------------
# Save results
# --------------------------------

with open(
    "results/fcl_replay_r3_seed2026.csv",
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "after_task",
        "task_1",
        "task_2",
        "task_3",
        "task_4",
        "task_5"
    ])

    for task_id, accuracies in enumerate(
        accuracy_matrix,
        start=1
    ):

        row: list[int | float | str] = [
            task_id,
            *accuracies
        ]

        while len(row) < 6:
            row.append("")

        writer.writerow(row)


# --------------------------------
# Save final model
# --------------------------------

torch.save(
    server.global_model.state_dict(),
    "results/fcl_replay_r3_seed2026_model.pth"
)


print()
print(
    "Results saved to "
    "results/fcl_replay_r3_seed2026.csv"
)

print(
    "Model saved to "
    "results/fcl_replay_r3_seed2026_model.pth"
)