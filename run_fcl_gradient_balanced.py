import csv
import os

import torch

from src.models.cnn import SimpleCNN
from src.data.cifar10 import get_cifar10_loaders
from src.data.federated_continual import (
    create_federated_continual_tasks
)
from src.clients.gradient_balanced_client import (
    GradientBalancedClient
)
from src.server.server import FederatedServer
from src.utils.seed import set_seed


# ============================================================
# Configuration
# ============================================================

SEED = 42

NUM_CLIENTS = 5
NUM_TASKS = 5

BATCH_SIZE = 64

# Milder non-IID heterogeneity
ALPHA = 1.0

ROUNDS_PER_TASK = 3
LOCAL_EPOCHS = 1

LEARNING_RATE = 0.001

MEMORY_SIZE = 500
TEMPERATURE = 2.0

MIN_WEIGHT = 0.5
MAX_WEIGHT = 1.5


EXPERIMENT_NAME = (
    "fcl_gradient_balanced_"
    "alpha1.0_w0.5_1.5_r3_seed42"
)

RESULTS_FILE = (
    f"results/{EXPERIMENT_NAME}.csv"
)

WEIGHTS_FILE = (
    "results/"
    "fcl_gradient_balanced_weights_"
    "alpha1.0_w0.5_1.5_r3_seed42.csv"
)

MODEL_FILE = (
    f"results/{EXPERIMENT_NAME}_model.pth"
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
    "Gradient-Balanced Replay + LwF FCL"
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
    f"Memory size: "
    f"{MEMORY_SIZE}"
)

print(
    f"Temperature: "
    f"{TEMPERATURE}"
)

print(
    f"Weight range: "
    f"[{MIN_WEIGHT}, {MAX_WEIGHT}]"
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


federated_train_tasks = (
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
# Create test loaders
# ============================================================

def create_task_loader(
    dataset,
    class_group,
    batch_size
):
    targets = dataset.targets

    indices = [
        index
        for index, label
        in enumerate(targets)
        if int(label) in class_group
    ]

    subset = torch.utils.data.Subset(
        dataset,
        indices
    )

    return torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False
    )


task_test_loaders = [
    create_task_loader(
        test_dataset,
        class_group,
        BATCH_SIZE
    )
    for class_group in CLASS_GROUPS
]


# ============================================================
# Model and server
# ============================================================

global_model = SimpleCNN()

server = FederatedServer(
    global_model=global_model,
    device=device
)


# ============================================================
# Clients
# ============================================================

clients = [
    GradientBalancedClient(
        client_id=client_id,
        device=device,
        memory_size=MEMORY_SIZE,
        seed=SEED,
        temperature=TEMPERATURE,
        min_weight=MIN_WEIGHT,
        max_weight=MAX_WEIGHT
    )
    for client_id in range(
        NUM_CLIENTS
    )
]


# ============================================================
# Evaluation
# ============================================================

def evaluate_accuracy(
    model,
    data_loader
):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in data_loader:

            images = images.to(
                device
            )

            labels = labels.to(
                device
            )

            outputs = model(
                images
            )

            predictions = torch.argmax(
                outputs,
                dim=1
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

    if total == 0:
        return 0.0

    return (
        100.0
        * correct
        / total
    )


# ============================================================
# Training
# ============================================================

accuracy_matrix = []
weight_records = []


for task_index in range(
    NUM_TASKS
):

    task_number = (
        task_index + 1
    )

    print()
    print("=" * 80)

    print(
        f"TASK {task_number} | "
        f"Classes: "
        f"{CLASS_GROUPS[task_index]}"
    )

    print("=" * 80)

    current_client_loaders = (
        federated_train_tasks[
            task_index
        ]
    )


    # ========================================================
    # Federated rounds
    # ========================================================

    for round_index in range(
        ROUNDS_PER_TASK
    ):

        round_number = (
            round_index + 1
        )

        print()

        print(
            f"Round "
            f"{round_number}/"
            f"{ROUNDS_PER_TASK}"
        )

        client_states = []
        client_sizes = []
        round_weights = []


        for client_id in range(
            NUM_CLIENTS
        ):

            client = (
                clients[
                    client_id
                ]
            )

            current_loader = (
                current_client_loaders[
                    client_id
                ]
            )

            current_dataset = (
                current_loader.dataset
            )


            training_loader = (
                client.create_replay_loader(
                    current_dataset,
                    batch_size=BATCH_SIZE
                )
            )


            (
                client_state,
                client_loss,
                client_accuracy,
                diagnostics
            ) = client.train(
                global_model=(
                    server.global_model
                ),
                current_loader=(
                    current_loader
                ),
                training_loader=(
                    training_loader
                ),
                local_epochs=(
                    LOCAL_EPOCHS
                ),
                learning_rate=(
                    LEARNING_RATE
                ),
                batch_size=(
                    BATCH_SIZE
                )
            )


            client_states.append(
                client_state
            )

            client_size = len(
                training_loader.dataset
            )

            client_sizes.append(
                client_size
            )


            retention_weight = float(
                diagnostics[
                    "retention_weight"
                ]
            )

            round_weights.append(
                retention_weight
            )


            weight_records.append({
                "seed": SEED,
                "alpha": ALPHA,
                "task": task_number,
                "round": round_number,
                "client": client_id,
                "old_gradient_norm": (
                    diagnostics[
                        "old_gradient_norm"
                    ]
                ),
                "new_gradient_norm": (
                    diagnostics[
                        "new_gradient_norm"
                    ]
                ),
                "magnitude_ratio": (
                    diagnostics[
                        "magnitude_ratio"
                    ]
                ),
                "balance_score": (
                    diagnostics[
                        "balance_score"
                    ]
                ),
                "retention_weight": (
                    retention_weight
                )
            })


            print(
                f"Client {client_id} | "
                f"Loss: "
                f"{client_loss:.4f} | "
                f"Acc: "
                f"{client_accuracy:.2f}% | "
                f"OldNorm: "
                f"{diagnostics['old_gradient_norm']:.4f} | "
                f"NewNorm: "
                f"{diagnostics['new_gradient_norm']:.4f} | "
                f"Ratio: "
                f"{diagnostics['magnitude_ratio']:.4f} | "
                f"Weight: "
                f"{retention_weight:.4f}"
            )


        # ====================================================
        # FedAvg
        # ====================================================

        server.aggregate(
            client_states,
            client_sizes
        )


        if task_index > 0:

            minimum_weight = min(
                round_weights
            )

            maximum_weight = max(
                round_weights
            )

            average_weight = (
                sum(round_weights)
                / len(round_weights)
            )

            print(
                f"Round Weight Summary | "
                f"Min: "
                f"{minimum_weight:.4f} | "
                f"Max: "
                f"{maximum_weight:.4f} | "
                f"Mean: "
                f"{average_weight:.4f}"
            )


    # ========================================================
    # Evaluation
    # ========================================================

    task_accuracies = []

    print()

    print(
        f"Evaluation after Task "
        f"{task_number}"
    )


    for evaluation_task_index in range(
        task_index + 1
    ):

        task_accuracy = (
            evaluate_accuracy(
                server.global_model,
                task_test_loaders[
                    evaluation_task_index
                ]
            )
        )

        task_accuracies.append(
            task_accuracy
        )

        print(
            f"Task "
            f"{evaluation_task_index + 1}: "
            f"{task_accuracy:.2f}%"
        )


    accuracy_matrix.append(
        task_accuracies
    )


    # ========================================================
    # Update replay memory
    # ========================================================

    for client_id in range(
        NUM_CLIENTS
    ):

        clients[
            client_id
        ].add_memory(
            current_client_loaders[
                client_id
            ].dataset
        )


    # ========================================================
    # Update teacher
    # ========================================================

    for client in clients:

        client.update_teacher(
            server.global_model
        )


# ============================================================
# Forgetting
# ============================================================

forgetting_scores = []


for task_index in range(
    NUM_TASKS - 1
):

    previous_accuracies = [
        row[task_index]
        for row in accuracy_matrix
        if len(row) > task_index
    ]

    best_previous_accuracy = max(
        previous_accuracies
    )

    final_accuracy = (
        accuracy_matrix[-1][
            task_index
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

final_accuracies = (
    accuracy_matrix[-1]
)

final_average_accuracy = (
    sum(final_accuracies)
    / len(final_accuracies)
)

average_forgetting = (
    sum(forgetting_scores)
    / len(forgetting_scores)
)


# ============================================================
# Print results
# ============================================================

print()

print(
    "Gradient-Balanced FCL Accuracy Matrix"
)

print("=" * 80)


for task_index, row in enumerate(
    accuracy_matrix,
    start=1
):

    formatted_row = [
        f"{accuracy:.2f}%"
        for accuracy in row
    ]

    print(
        f"After Task "
        f"{task_index}: "
        f"{formatted_row}"
    )


for task_index, forgetting in enumerate(
    forgetting_scores,
    start=1
):

    print(
        f"Task "
        f"{task_index} "
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
# Save accuracy results
# ============================================================

with open(
    RESULTS_FILE,
    "w",
    newline=""
) as output_file:

    fieldnames = [
        "after_task",
        "task_1",
        "task_2",
        "task_3",
        "task_4",
        "task_5"
    ]

    writer = csv.DictWriter(
        output_file,
        fieldnames=fieldnames
    )

    writer.writeheader()


    for task_index, row in enumerate(
        accuracy_matrix,
        start=1
    ):

        csv_row = {
            "after_task": task_index,
            "task_1": "",
            "task_2": "",
            "task_3": "",
            "task_4": "",
            "task_5": ""
        }


        for accuracy_index, accuracy in enumerate(
            row,
            start=1
        ):

            csv_row[
                f"task_{accuracy_index}"
            ] = accuracy


        writer.writerow(
            csv_row
        )


# ============================================================
# Save adaptive-weight diagnostics
# ============================================================

with open(
    WEIGHTS_FILE,
    "w",
    newline=""
) as output_file:

    fieldnames = [
        "seed",
        "alpha",
        "task",
        "round",
        "client",
        "old_gradient_norm",
        "new_gradient_norm",
        "magnitude_ratio",
        "balance_score",
        "retention_weight"
    ]

    writer = csv.DictWriter(
        output_file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        weight_records
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
    f"Adaptive weights saved to "
    f"{WEIGHTS_FILE}"
)

print(
    f"Model saved to "
    f"{MODEL_FILE}"
)