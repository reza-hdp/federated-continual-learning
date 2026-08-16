import csv
import math

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


# ============================================================
# Configuration
# ============================================================

SEEDS = [42, 123, 2026]

NUM_CLIENTS = 5
NUM_TASKS = 5

BATCH_SIZE = 64
ALPHA = 0.5

ROUNDS_PER_TASK = 3
LEARNING_RATE = 0.001

OUTPUT_FILE = (
    "results/"
    "conflict_forgetting_diagnostic_all.csv"
)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

criterion = nn.CrossEntropyLoss()


# ============================================================
# Continual task definition
# ============================================================

CLASS_GROUPS = [
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
    [8, 9]
]


# ============================================================
# Helper: copy model state
# ============================================================

def copy_model_state(model):
    return {
        name: value.detach().cpu().clone()
        for name, value
        in model.state_dict().items()
    }


# ============================================================
# Helper: local training
# ============================================================

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


# ============================================================
# Helper: accuracy
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

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            matches = torch.eq(
                predictions,
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

    return (
        100.0
        * correct
        / total
    )


# ============================================================
# Pearson correlation
# ============================================================

def pearson_correlation(
    values_x,
    values_y
):
    if len(values_x) != len(values_y):
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

    differences_x = [
        value - mean_x
        for value in values_x
    ]

    differences_y = [
        value - mean_y
        for value in values_y
    ]

    numerator = sum(
        difference_x * difference_y
        for difference_x, difference_y
        in zip(
            differences_x,
            differences_y
        )
    )

    denominator_x = sum(
        difference ** 2
        for difference in differences_x
    )

    denominator_y = sum(
        difference ** 2
        for difference in differences_y
    )

    denominator = math.sqrt(
        denominator_x
        * denominator_y
    )

    if denominator == 0.0:
        return 0.0

    return numerator / denominator


# ============================================================
# Rank values for Spearman
# Handles tied values using average ranks.
# ============================================================

def rank_values(values):
    indexed_values = list(
        enumerate(values)
    )

    sorted_values = sorted(
        indexed_values,
        key=lambda item: item[1]
    )

    ranks = [
        0.0
        for _ in values
    ]

    position = 0

    while position < len(
        sorted_values
    ):

        end_position = position

        current_value = (
            sorted_values[
                position
            ][1]
        )

        while (
            end_position + 1
            < len(sorted_values)
            and sorted_values[
                end_position + 1
            ][1]
            == current_value
        ):
            end_position += 1

        average_rank = (
            position
            + end_position
        ) / 2.0 + 1.0

        for rank_position in range(
            position,
            end_position + 1
        ):

            original_index = (
                sorted_values[
                    rank_position
                ][0]
            )

            ranks[
                original_index
            ] = average_rank

        position = (
            end_position + 1
        )

    return ranks


# ============================================================
# Spearman correlation
# ============================================================

def spearman_correlation(
    values_x,
    values_y
):
    if len(values_x) != len(values_y):
        raise ValueError(
            "Input lists must have "
            "the same length."
        )

    ranks_x = rank_values(
        values_x
    )

    ranks_y = rank_values(
        values_y
    )

    return pearson_correlation(
        ranks_x,
        ranks_y
    )


# ============================================================
# Train one federated task
# ============================================================

def train_federated_task(
    server,
    client_loaders
):
    for _ in range(
        ROUNDS_PER_TASK
    ):

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

            trained_client_model = (
                train_local_model(
                    current_global_state,
                    client_loaders[
                        client_id
                    ]
                )
            )

            trained_client_state = (
                copy_model_state(
                    trained_client_model
                )
            )

            client_states.append(
                trained_client_state
            )

            sample_count = len(
                client_loaders[
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


# ============================================================
# Main diagnostic
# ============================================================

all_records = []


print()
print(
    "Multi-Seed Conflict-Forgetting Diagnostic"
)
print("=" * 80)

print(
    "Device:",
    device
)


for seed in SEEDS:

    print()
    print("=" * 80)
    print(f"SEED {seed}")
    print("=" * 80)

    set_seed(seed)

    train_loader, _ = (
        get_cifar10_loaders(
            batch_size=BATCH_SIZE
        )
    )

    train_dataset = (
        train_loader.dataset
    )

    federated_tasks = (
        create_federated_continual_tasks(
            dataset=train_dataset,
            num_clients=NUM_CLIENTS,
            class_groups=CLASS_GROUPS,
            batch_size=BATCH_SIZE,
            alpha=ALPHA,
            seed=seed
        )
    )

    seed_global_model = (
        SimpleCNN()
    )

    seed_server = (
        FederatedServer(
            global_model=seed_global_model,
            device=device
        )
    )

    # --------------------------------------------------------
    # Learn Task 1 first
    # --------------------------------------------------------

    print()
    print("Learning Task 1")

    train_federated_task(
        seed_server,
        federated_tasks[0]
    )

    # --------------------------------------------------------
    # Examine transitions:
    # Task 1 -> 2
    # Task 2 -> 3
    # Task 3 -> 4
    # Task 4 -> 5
    # --------------------------------------------------------

    for current_task in range(
        1,
        NUM_TASKS
    ):

        previous_task = (
            current_task - 1
        )

        print()
        print(
            f"Transition "
            f"{previous_task + 1} -> "
            f"{current_task + 1}"
        )

        pre_transition_state = (
            copy_model_state(
                seed_server.global_model
            )
        )

        current_task_loaders = (
            federated_tasks[
                current_task
            ]
        )

        previous_task_loaders = (
            federated_tasks[
                previous_task
            ]
        )

        # ----------------------------------------------------
        # One-step local models for diagnostic measurement
        # ----------------------------------------------------

        diagnostic_models = []
        update_vectors = []
        sample_counts = []

        for client_id in range(
            NUM_CLIENTS
        ):

            diagnostic_model = (
                train_local_model(
                    pre_transition_state,
                    current_task_loaders[
                        client_id
                    ]
                )
            )

            diagnostic_state = (
                copy_model_state(
                    diagnostic_model
                )
            )

            update_vector = (
                flatten_model_update(
                    pre_transition_state,
                    diagnostic_state
                )
            )

            diagnostic_models.append(
                diagnostic_model
            )

            update_vectors.append(
                update_vector
            )

            sample_count = len(
                current_task_loaders[
                    client_id
                ].dataset
            )

            sample_counts.append(
                sample_count
            )

        # ----------------------------------------------------
        # Leave-one-out conflict + forgetting
        # ----------------------------------------------------

        for client_id in range(
            NUM_CLIENTS
        ):

            other_updates = [
                update
                for index, update
                in enumerate(
                    update_vectors
                )
                if index != client_id
            ]

            other_sizes = [
                size
                for index, size
                in enumerate(
                    sample_counts
                )
                if index != client_id
            ]

            other_reference = (
                calculate_reference_update(
                    other_updates,
                    other_sizes
                )
            )

            conflict_score = (
                calculate_conflict_score(
                    update_vectors[
                        client_id
                    ],
                    other_reference
                )
            )

            reference_model = (
                SimpleCNN().to(
                    device
                )
            )

            reference_model.load_state_dict(
                pre_transition_state
            )

            before_accuracy = (
                evaluate_accuracy(
                    reference_model,
                    previous_task_loaders[
                        client_id
                    ]
                )
            )

            after_accuracy = (
                evaluate_accuracy(
                    diagnostic_models[
                        client_id
                    ],
                    previous_task_loaders[
                        client_id
                    ]
                )
            )

            forgetting = max(
                0.0,
                before_accuracy
                - after_accuracy
            )

            record = {
                "seed": seed,
                "transition": (
                    f"{previous_task + 1}"
                    f"->{current_task + 1}"
                ),
                "client": client_id,
                "conflict": float(
                    conflict_score
                ),
                "before_accuracy": float(
                    before_accuracy
                ),
                "after_accuracy": float(
                    after_accuracy
                ),
                "forgetting": float(
                    forgetting
                )
            }

            all_records.append(
                record
            )

            print(
                f"Client {client_id} | "
                f"Conflict: "
                f"{conflict_score:.4f} | "
                f"Forgetting: "
                f"{forgetting:.2f} pp"
            )

        # ----------------------------------------------------
        # Now actually learn the current task globally.
        #
        # This prepares the global model for the next
        # transition. The diagnostic local models above
        # are NOT aggregated.
        # ----------------------------------------------------

        print(
            f"Learning Task "
            f"{current_task + 1}"
        )

        train_federated_task(
            seed_server,
            current_task_loaders
        )


# ============================================================
# Extract all observations
# ============================================================

all_conflicts = [
    record["conflict"]
    for record in all_records
]

all_forgetting = [
    record["forgetting"]
    for record in all_records
]


# ============================================================
# Overall correlations
# ============================================================

overall_pearson = (
    pearson_correlation(
        all_conflicts,
        all_forgetting
    )
)

overall_spearman = (
    spearman_correlation(
        all_conflicts,
        all_forgetting
    )
)


# ============================================================
# Print overall summary
# ============================================================

print()
print("=" * 80)
print(
    "Overall Conflict-Forgetting Summary"
)
print("=" * 80)

print(
    f"Number of Observations: "
    f"{len(all_records)}"
)

print(
    f"Conflict Range: "
    f"{min(all_conflicts):.4f} "
    f"to "
    f"{max(all_conflicts):.4f}"
)

print(
    f"Forgetting Range: "
    f"{min(all_forgetting):.2f} "
    f"to "
    f"{max(all_forgetting):.2f} pp"
)

print(
    f"Pearson Correlation: "
    f"{overall_pearson:+.4f}"
)

print(
    f"Spearman Correlation: "
    f"{overall_spearman:+.4f}"
)


# ============================================================
# Correlation by seed
# ============================================================

print()
print("Correlation by Seed")
print("=" * 80)

for seed in SEEDS:

    seed_records = [
        record
        for record in all_records
        if record["seed"] == seed
    ]

    seed_conflicts = [
        record["conflict"]
        for record in seed_records
    ]

    seed_forgetting = [
        record["forgetting"]
        for record in seed_records
    ]

    seed_pearson = (
        pearson_correlation(
            seed_conflicts,
            seed_forgetting
        )
    )

    seed_spearman = (
        spearman_correlation(
            seed_conflicts,
            seed_forgetting
        )
    )

    print(
        f"Seed {seed} | "
        f"Pearson: "
        f"{seed_pearson:+.4f} | "
        f"Spearman: "
        f"{seed_spearman:+.4f}"
    )


# ============================================================
# Correlation by transition
# ============================================================

print()
print("Correlation by Task Transition")
print("=" * 80)

transition_names = [
    "1->2",
    "2->3",
    "3->4",
    "4->5"
]

for transition_name in (
    transition_names
):

    transition_records = [
        record
        for record in all_records
        if record["transition"]
        == transition_name
    ]

    transition_conflicts = [
        record["conflict"]
        for record
        in transition_records
    ]

    transition_forgetting = [
        record["forgetting"]
        for record
        in transition_records
    ]

    transition_pearson = (
        pearson_correlation(
            transition_conflicts,
            transition_forgetting
        )
    )

    transition_spearman = (
        spearman_correlation(
            transition_conflicts,
            transition_forgetting
        )
    )

    print(
        f"Transition "
        f"{transition_name} | "
        f"Pearson: "
        f"{transition_pearson:+.4f} | "
        f"Spearman: "
        f"{transition_spearman:+.4f}"
    )


# ============================================================
# Save all 60 observations
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    newline=""
) as output_file:

    writer = csv.DictWriter(
        output_file,
        fieldnames=[
            "seed",
            "transition",
            "client",
            "conflict",
            "before_accuracy",
            "after_accuracy",
            "forgetting"
        ]
    )

    writer.writeheader()

    writer.writerows(
        all_records
    )


print()
print(
    f"Diagnostic data saved to "
    f"{OUTPUT_FILE}"
)


# ============================================================
# Interpretation
# ============================================================

print()
print("Diagnostic Interpretation")
print("=" * 80)

absolute_pearson = abs(
    overall_pearson
)

absolute_spearman = abs(
    overall_spearman
)

if (
    overall_pearson >= 0.50
    and overall_spearman >= 0.50
):

    print(
        "RESULT: Update conflict shows a "
        "consistent positive relationship "
        "with forgetting."
    )

    print(
        "Conflict is a promising adaptive "
        "retention signal."
    )

elif (
    overall_pearson >= 0.30
    or overall_spearman >= 0.30
):

    print(
        "RESULT: Update conflict contains "
        "some information about forgetting, "
        "but is not sufficient alone."
    )

    print(
        "A multi-signal retention-risk "
        "estimator should be investigated."
    )

else:

    print(
        "RESULT: Update conflict is a weak "
        "predictor of forgetting in this setup."
    )

    print(
        "Conflict should not be used as the "
        "main adaptive retention signal."
    )