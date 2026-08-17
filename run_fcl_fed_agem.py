import csv
import os

import torch
import torch.nn as nn

from src.models.cnn import SimpleCNN
from src.data.cifar10 import get_cifar10_loaders
from src.data.continual import create_continual_tasks
from src.data.federated_continual import (
    create_federated_continual_tasks
)
from src.clients.fed_agem_client import (
    FedAGEMClient
)
from src.server.server import (
    FederatedServer
)
from src.utils.training import evaluate
from src.utils.seed import set_seed


# ============================================================
# Configuration
# ============================================================

SEED = 1001

NUM_CLIENTS = 5
NUM_TASKS = 5

ROUNDS_PER_TASK = 3
LOCAL_EPOCHS = 1

BATCH_SIZE = 64

ALPHA = 0.5

MEMORY_SIZE = 500

LEARNING_RATE = 0.001


EXPERIMENT_NAME = (
    "fcl_fed_agem_"
    "lr0.001_alpha0.5_r3_seed1001"
)

RESULTS_FILE = (
    f"results/"
    f"{EXPERIMENT_NAME}.csv"
)

DIAGNOSTICS_FILE = (
    "results/"
    "fcl_fed_agem_diagnostics_"
    "lr0.001_alpha0.5_r3_seed1001.csv"
)

MODEL_FILE = (
    f"results/"
    f"{EXPERIMENT_NAME}_model.pth"
)


# ============================================================
# Setup
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)

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
    "Fed-A-GEM FCL"
)

print("=" * 80)

print(
    f"Device: {device}"
)

print(
    f"Seed: {SEED}"
)

print(
    f"Dirichlet alpha: {ALPHA}"
)

print(
    f"Rounds per task: "
    f"{ROUNDS_PER_TASK}"
)

print(
    f"Local epochs: "
    f"{LOCAL_EPOCHS}"
)

print(
    f"Memory size: "
    f"{MEMORY_SIZE}"
)

print(
    f"Learning rate: "
    f"{LEARNING_RATE}"
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
# Tasks
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
    FedAGEMClient(
        client_id=client_id,
        device=device,
        memory_size=MEMORY_SIZE,
        seed=SEED
    )
    for client_id in range(
        NUM_CLIENTS
    )
]


# ============================================================
# Global reference-gradient aggregation
# ============================================================

def compute_global_reference_gradient():
    local_gradients = []


    for client in clients:

        gradient = (
            client.compute_buffer_gradient(
                global_model=(
                    server.global_model
                ),
                batch_size=BATCH_SIZE
            )
        )


        if gradient is not None:

            local_gradients.append(
                gradient
            )


    if not local_gradients:

        return None


    stacked_gradients = (
        torch.stack(
            local_gradients,
            dim=0
        )
    )


    global_gradient = (
        torch.mean(
            stacked_gradients,
            dim=0
        )
    )


    return (
        global_gradient
        .detach()
        .clone()
    )


# ============================================================
# Experiment storage
# ============================================================

accuracy_matrix = []

diagnostic_records = []


global_reference_gradient = None


# ============================================================
# Federated continual learning
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


        if (
            global_reference_gradient
            is None
        ):

            print(
                "Global reference gradient: "
                "not available"
            )

        else:

            reference_norm = (
                torch.linalg.vector_norm(
                    global_reference_gradient
                ).item()
            )

            print(
                f"Global reference gradient norm: "
                f"{reference_norm:.6f}"
            )


        client_states = []

        client_sizes = []


        # ====================================================
        # Local client updates
        # ====================================================

        for client_id in range(
            NUM_CLIENTS
        ):

            current_loader = (
                current_client_loaders[
                    client_id
                ]
            )


            (
                client_state,
                client_loss,
                client_accuracy,
                diagnostics
            ) = clients[
                client_id
            ].train(
                global_model=(
                    server.global_model
                ),
                current_loader=(
                    current_loader
                ),
                global_reference_gradient=(
                    global_reference_gradient
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


            current_size = len(
                current_loader.dataset
            )


            client_sizes.append(
                current_size
            )


            diagnostic_records.append({
                "seed": SEED,
                "alpha": ALPHA,
                "task": task_number,
                "round": round_number,
                "client": client_id,
                "projection_count": (
                    diagnostics[
                        "projection_count"
                    ]
                ),
                "batch_count": (
                    diagnostics[
                        "batch_count"
                    ]
                ),
                "projection_rate": (
                    diagnostics[
                        "projection_rate"
                    ]
                ),
                "average_dot_product": (
                    diagnostics[
                        "average_dot_product"
                    ]
                ),
                "buffer_size": (
                    diagnostics[
                        "buffer_size"
                    ]
                )
            })


            print(
                f"Client {client_id} | "
                f"Samples: "
                f"{current_size} | "
                f"Loss: "
                f"{client_loss:.4f} | "
                f"Accuracy: "
                f"{client_accuracy:.2f}% | "
                f"Projection: "
                f"{diagnostics['projection_rate']:.2f}% | "
                f"Buffer: "
                f"{diagnostics['buffer_size']}"
            )


        # ====================================================
        # FedAvg aggregation
        # ====================================================

        server.aggregate(
            client_states,
            client_sizes
        )


        # ====================================================
        # Recompute global buffer gradient
        # ====================================================

        global_reference_gradient = (
            compute_global_reference_gradient()
        )


    # ========================================================
    # Update buffers once after each task
    # ========================================================

    for client_id in range(
        NUM_CLIENTS
    ):

        clients[
            client_id
        ].update_buffer(
            current_client_loaders[
                client_id
            ].dataset
        )


    # ========================================================
    # New buffer gradient for next task
    # ========================================================

    global_reference_gradient = (
        compute_global_reference_gradient()
    )


    print()

    print(
        "Client buffer sizes:"
    )


    for client_id in range(
        NUM_CLIENTS
    ):

        print(
            f"Client {client_id}: "
            f"{clients[client_id].buffer_size()}"
        )


    # ========================================================
    # Evaluation
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
# Forgetting
# ============================================================

forgetting_scores = []


for task_id in range(
    NUM_TASKS - 1
):

    best_previous_accuracy = max(
        row[
            task_id
        ]
        for row in accuracy_matrix
        if len(
            row
        ) > task_id
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


# ============================================================
# Print final matrix
# ============================================================

print()

print(
    "Fed-A-GEM FCL Accuracy Matrix"
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


for task_id, forgetting in enumerate(
    forgetting_scores,
    start=1
):

    print(
        f"Task "
        f"{task_id} "
        f"Forgetting: "
        f"{forgetting:.2f} "
        f"percentage points"
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
# Save accuracy matrix
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


        while len(
            row
        ) < 6:

            row.append(
                ""
            )


        writer.writerow(
            row
        )


# ============================================================
# Save diagnostics
# ============================================================

with open(
    DIAGNOSTICS_FILE,
    "w",
    newline=""
) as output_file:

    fieldnames = [
        "seed",
        "alpha",
        "task",
        "round",
        "client",
        "projection_count",
        "batch_count",
        "projection_rate",
        "average_dot_product",
        "buffer_size"
    ]


    writer = csv.DictWriter(
        output_file,
        fieldnames=fieldnames
    )


    writer.writeheader()

    writer.writerows(
        diagnostic_records
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
    f"Diagnostics saved to "
    f"{DIAGNOSTICS_FILE}"
)

print(
    f"Model saved to "
    f"{MODEL_FILE}"
)