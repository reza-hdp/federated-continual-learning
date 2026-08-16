import csv

import torch
import torch.nn as nn

from src.models.cnn import SimpleCNN
from src.data.cifar10 import get_cifar10_loaders
from src.data.continual import create_continual_tasks
from src.data.federated_continual import (
    create_federated_continual_tasks
)
from src.clients.replay_lwf_client import ReplayLwFClient
from src.server.server import FederatedServer
from src.utils.training import evaluate
from src.utils.seed import set_seed


# ============================================================
# Configuration
# ============================================================

SEED = 42

NUM_CLIENTS = 5
NUM_TASKS = 5

ROUNDS_PER_TASK = 3
LOCAL_EPOCHS = 1

BATCH_SIZE = 64

# Milder non-IID heterogeneity
ALPHA = 1.0

MEMORY_SIZE = 500

LEARNING_RATE = 0.001

TEMPERATURE = 2.0
DISTILLATION_WEIGHT = 1.0


RESULTS_FILE = (
    "results/"
    "fcl_replay_lwf_alpha1.0_w1.0_r3_seed42.csv"
)

MODEL_FILE = (
    "results/"
    "fcl_replay_lwf_alpha1.0_w1.0_r3_seed42_model.pth"
)


# ============================================================
# Setup
# ============================================================

set_seed(
    SEED
)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print()

print(
    "Fixed Replay + LwF FCL"
)

print("=" * 80)

print(
    f"Device: {device}"
)

print(
    f"Seed: {SEED}"
)

print(
    f"Dirichlet alpha: "
    f"{ALPHA}"
)

print(
    f"Rounds per task: "
    f"{ROUNDS_PER_TASK}"
)

print(
    f"Memory size: "
    f"{MEMORY_SIZE}"
)

print(
    f"Temperature: "
    f"{TEMPERATURE}"
)

print(
    f"Distillation weight: "
    f"{DISTILLATION_WEIGHT}"
)


# ============================================================
# Load CIFAR-10
# ============================================================

train_loader, test_loader = (
    get_cifar10_loaders(
        batch_size=BATCH_SIZE
    )
)

train_dataset = (
    train_loader.dataset
)

test_dataset = (
    test_loader.dataset
)


# ============================================================
# Continual tasks
# ============================================================

CLASS_GROUPS = [
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
    [8, 9]
]


federated_tasks = (
    create_federated_continual_tasks(
        dataset=train_dataset,
        num_clients=NUM_CLIENTS,
        class_groups=CLASS_GROUPS,
        batch_size=BATCH_SIZE,
        alpha=ALPHA,
        seed=SEED
    )
)


# ============================================================
# Continual test loaders
# ============================================================

test_task_loaders = (
    create_continual_tasks(
        dataset=test_dataset,
        class_groups=CLASS_GROUPS,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
)


# ============================================================
# Model and server
# ============================================================

global_model = (
    SimpleCNN()
)

server = (
    FederatedServer(
        global_model=global_model,
        device=device
    )
)

criterion = (
    nn.CrossEntropyLoss()
)


# ============================================================
# Persistent clients
# ============================================================

clients = [
    ReplayLwFClient(
        client_id=client_id,
        device=device,
        memory_size=MEMORY_SIZE,
        seed=SEED,
        temperature=TEMPERATURE,
        distillation_weight=DISTILLATION_WEIGHT
    )
    for client_id in range(
        NUM_CLIENTS
    )
]


accuracy_matrix = []


# ============================================================
# Federated Continual Learning
# ============================================================

for current_task in range(
    NUM_TASKS
):

    task_number = (
        current_task + 1
    )

    print()
    print("=" * 80)

    print(
        f"TASK {task_number} | "
        f"Classes: "
        f"{CLASS_GROUPS[current_task]}"
    )

    print("=" * 80)

    current_client_loaders = (
        federated_tasks[
            current_task
        ]
    )


    # ========================================================
    # Federated rounds
    # ========================================================

    for round_number in range(
        1,
        ROUNDS_PER_TASK + 1
    ):

        print()

        print(
            f"Federated Round "
            f"{round_number}/"
            f"{ROUNDS_PER_TASK}"
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


            # ------------------------------------------------
            # Task 1 has no replay memory yet
            # ------------------------------------------------

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


            # ------------------------------------------------
            # Local training
            # ------------------------------------------------

            (
                client_state,
                client_loss,
                client_accuracy
            ) = clients[
                client_id
            ].train(
                global_model=(
                    server.global_model
                ),
                train_loader=(
                    training_loader
                ),
                local_epochs=(
                    LOCAL_EPOCHS
                ),
                learning_rate=(
                    LEARNING_RATE
                )
            )


            client_states.append(
                client_state
            )

            training_size = len(
                training_loader.dataset
            )

            client_sizes.append(
                training_size
            )


            print(
                f"Client {client_id} | "
                f"Samples: "
                f"{training_size} | "
                f"Loss: "
                f"{client_loss:.4f} | "
                f"Accuracy: "
                f"{client_accuracy:.2f}%"
            )


        # ====================================================
        # Weighted FedAvg
        # ====================================================

        server.aggregate(
            client_states,
            client_sizes
        )


    # ========================================================
    # Update replay memory
    # ========================================================

    for client_id in range(
        NUM_CLIENTS
    ):

        current_dataset = (
            current_client_loaders[
                client_id
            ].dataset
        )

        clients[
            client_id
        ].add_memory(
            current_dataset
        )


    # ========================================================
    # Update teacher models
    # ========================================================

    for client in clients:

        client.update_teacher(
            server.global_model
        )


    # ========================================================
    # Evaluate
    # ========================================================

    print()

    print(
        f"Evaluation after Task "
        f"{task_number}"
    )

    task_accuracies = []


    for test_task in range(
        current_task + 1
    ):

        _, test_accuracy = (
            evaluate(
                server.global_model,
                test_task_loaders[
                    test_task
                ],
                criterion,
                device
            )
        )

        task_accuracies.append(
            test_accuracy
        )


        print(
            f"Task "
            f"{test_task + 1}: "
            f"{test_accuracy:.2f}%"
        )


    accuracy_matrix.append(
        task_accuracies
    )


# ============================================================
# Accuracy matrix
# ============================================================

print()

print(
    "FCL Replay + LwF Accuracy Matrix"
)

print("=" * 80)


for task_id, accuracies in enumerate(
    accuracy_matrix,
    start=1
):

    formatted = [
        f"{accuracy:.2f}%"
        for accuracy in accuracies
    ]

    print(
        f"After Task "
        f"{task_id}: "
        f"{formatted}"
    )


# ============================================================
# Forgetting
# ============================================================

forgetting_scores = []


for task_id in range(
    NUM_TASKS - 1
):

    best_previous_accuracy = max(
        row[task_id]
        for row in accuracy_matrix
        if len(row) > task_id
    )

    final_accuracy = (
        accuracy_matrix[-1][
            task_id
        ]
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
        f"Task "
        f"{task_id + 1} "
        f"Forgetting: "
        f"{forgetting:.2f} "
        f"percentage points"
    )


# ============================================================
# Final metrics
# ============================================================

final_average_accuracy = (
    sum(
        accuracy_matrix[-1]
    )
    / len(
        accuracy_matrix[-1]
    )
)

average_forgetting = (
    sum(
        forgetting_scores
    )
    / len(
        forgetting_scores
    )
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


# ============================================================
# Save results
# ============================================================

with open(
    RESULTS_FILE,
    "w",
    newline=""
) as output_file:

    writer = csv.writer(
        output_file
    )


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

        row = [
            task_id,
            *accuracies
        ]


        while len(row) < 6:

            row.append(
                ""
            )


        writer.writerow(
            row
        )


# ============================================================
# Save model
# ============================================================

torch.save(
    server.global_model.state_dict(),
    MODEL_FILE
)


# ============================================================
# Finished
# ============================================================

print()

print(
    f"Results saved to "
    f"{RESULTS_FILE}"
)

print(
    f"Model saved to "
    f"{MODEL_FILE}"
)