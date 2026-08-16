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
from src.algorithms.gradient_interference import (
    flatten_gradients,
    gradient_diagnostic
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
    "gradient_interference_forgetting_all.csv"
)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

criterion = nn.CrossEntropyLoss()

CLASS_GROUPS = [
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
    [8, 9]
]


# ============================================================
# Copy model state
# ============================================================

def copy_model_state(model):
    return {
        name: value.detach().cpu().clone()
        for name, value
        in model.state_dict().items()
    }


# ============================================================
# Local training
# ============================================================

def train_client_model(
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
# Accuracy
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
# Calculate gradient over a loader
# ============================================================

def calculate_loader_gradient(
    initial_state,
    data_loader
):
    gradient_model = SimpleCNN().to(
        device
    )

    gradient_model.load_state_dict(
        initial_state
    )

    gradient_model.train()

    gradient_model.zero_grad(
        set_to_none=True
    )

    total_samples = 0

    for images, labels in data_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = gradient_model(
            images
        )

        loss = criterion(
            outputs,
            labels
        )

        current_batch_size = int(
            labels.numel()
        )

        weighted_loss = (
            loss
            * current_batch_size
        )

        weighted_loss.backward()

        total_samples += (
            current_batch_size
        )

    if total_samples == 0:

        return torch.empty(
            0,
            dtype=torch.float32
        )

    for parameter in (
        gradient_model.parameters()
    ):

        if parameter.grad is not None:

            parameter.grad.div_(
                float(total_samples)
            )

    return flatten_gradients(
        gradient_model
    )


# ============================================================
# Train one federated task
# ============================================================

def train_federated_task(
    federated_server,
    client_loaders
):
    for _ in range(
        ROUNDS_PER_TASK
    ):

        current_state = (
            copy_model_state(
                federated_server.global_model
            )
        )

        round_states = []
        round_sizes = []

        for client_id in range(
            NUM_CLIENTS
        ):

            trained_client = (
                train_client_model(
                    current_state,
                    client_loaders[
                        client_id
                    ]
                )
            )

            trained_state = (
                copy_model_state(
                    trained_client
                )
            )

            round_states.append(
                trained_state
            )

            sample_count = len(
                client_loaders[
                    client_id
                ].dataset
            )

            round_sizes.append(
                sample_count
            )

        federated_server.aggregate(
            round_states,
            round_sizes
        )


# ============================================================
# Pearson correlation
# ============================================================

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

    return (
        numerator
        / denominator
    )


# ============================================================
# Rank values
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
    if len(values_x) != len(
        values_y
    ):
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
# Main experiment
# ============================================================

all_records = []

print()
print(
    "Multi-Seed Gradient-Interference Diagnostic"
)
print("=" * 85)

print(
    f"Device: {device}"
)


for seed in SEEDS:

    print()
    print("=" * 85)
    print(
        f"SEED {seed}"
    )
    print("=" * 85)

    set_seed(
        seed
    )

    seed_train_loader, _ = (
        get_cifar10_loaders(
            batch_size=BATCH_SIZE
        )
    )

    seed_train_dataset = (
        seed_train_loader.dataset
    )

    seed_tasks = (
        create_federated_continual_tasks(
            dataset=seed_train_dataset,
            num_clients=NUM_CLIENTS,
            class_groups=CLASS_GROUPS,
            batch_size=BATCH_SIZE,
            alpha=ALPHA,
            seed=seed
        )
    )

    seed_initial_model = (
        SimpleCNN()
    )

    seed_server = (
        FederatedServer(
            global_model=seed_initial_model,
            device=device
        )
    )

    # --------------------------------------------------------
    # Learn Task 1
    # --------------------------------------------------------

    print()
    print(
        "Learning Task 1"
    )

    train_federated_task(
        seed_server,
        seed_tasks[0]
    )

    # --------------------------------------------------------
    # Four transitions
    # --------------------------------------------------------

    for current_task_index in range(
        1,
        NUM_TASKS
    ):

        previous_task_index = (
            current_task_index - 1
        )

        transition_name = (
            f"{previous_task_index + 1}"
            f"->{current_task_index + 1}"
        )

        print()
        print(
            f"Transition "
            f"{transition_name}"
        )
        print("-" * 85)

        pre_transition_state = (
            copy_model_state(
                seed_server.global_model
            )
        )

        old_task_loaders = (
            seed_tasks[
                previous_task_index
            ]
        )

        new_task_loaders = (
            seed_tasks[
                current_task_index
            ]
        )

        # ----------------------------------------------------
        # Each client
        # ----------------------------------------------------

        for client_id in range(
            NUM_CLIENTS
        ):

            old_gradient = (
                calculate_loader_gradient(
                    pre_transition_state,
                    old_task_loaders[
                        client_id
                    ]
                )
            )

            new_gradient = (
                calculate_loader_gradient(
                    pre_transition_state,
                    new_task_loaders[
                        client_id
                    ]
                )
            )

            diagnostic = (
                gradient_diagnostic(
                    old_gradient,
                    new_gradient
                )
            )

            # ------------------------------------------------
            # Train a diagnostic copy on the new task
            # ------------------------------------------------

            diagnostic_model = (
                train_client_model(
                    pre_transition_state,
                    new_task_loaders[
                        client_id
                    ]
                )
            )

            # ------------------------------------------------
            # Accuracy before new-task training
            # ------------------------------------------------

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
                    old_task_loaders[
                        client_id
                    ]
                )
            )

            # ------------------------------------------------
            # Accuracy after new-task local training
            # ------------------------------------------------

            after_accuracy = (
                evaluate_accuracy(
                    diagnostic_model,
                    old_task_loaders[
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
                    transition_name
                ),
                "client": client_id,
                "cosine_similarity": float(
                    diagnostic[
                        "cosine_similarity"
                    ]
                ),
                "interference": float(
                    diagnostic[
                        "interference"
                    ]
                ),
                "old_gradient_norm": float(
                    diagnostic[
                        "old_gradient_norm"
                    ]
                ),
                "new_gradient_norm": float(
                    diagnostic[
                        "new_gradient_norm"
                    ]
                ),
                "magnitude_ratio": float(
                    diagnostic[
                        "magnitude_ratio"
                    ]
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
                f"Cos: "
                f"{diagnostic['cosine_similarity']:+.4f} | "
                f"Int: "
                f"{diagnostic['interference']:.4f} | "
                f"Ratio: "
                f"{diagnostic['magnitude_ratio']:.4f} | "
                f"Forget: "
                f"{forgetting:.2f} pp"
            )

        # ----------------------------------------------------
        # Actually learn the new task globally
        # ----------------------------------------------------

        print(
            f"Learning Task "
            f"{current_task_index + 1}"
        )

        train_federated_task(
            seed_server,
            new_task_loaders
        )


# ============================================================
# Save raw observations
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
            "cosine_similarity",
            "interference",
            "old_gradient_norm",
            "new_gradient_norm",
            "magnitude_ratio",
            "before_accuracy",
            "after_accuracy",
            "forgetting"
        ]
    )

    writer.writeheader()

    writer.writerows(
        all_records
    )


# ============================================================
# Correlation helper
# ============================================================

def calculate_signal_correlations(
    records,
    signal_name
):
    signal_values = [
        record[signal_name]
        for record in records
    ]

    forgetting_values = [
        record["forgetting"]
        for record in records
    ]

    pearson = (
        pearson_correlation(
            signal_values,
            forgetting_values
        )
    )

    spearman = (
        spearman_correlation(
            signal_values,
            forgetting_values
        )
    )

    return (
        pearson,
        spearman
    )


# ============================================================
# Overall results
# ============================================================

SIGNALS = [
    "cosine_similarity",
    "interference",
    "old_gradient_norm",
    "new_gradient_norm",
    "magnitude_ratio"
]


print()
print("=" * 85)
print(
    "Overall Gradient-Signal Correlations"
)
print("=" * 85)

print(
    f"Number of Observations: "
    f"{len(all_records)}"
)

overall_results = {}

for signal_name in SIGNALS:

    signal_pearson, signal_spearman = (
        calculate_signal_correlations(
            all_records,
            signal_name
        )
    )

    overall_results[
        signal_name
    ] = {
        "pearson": (
            signal_pearson
        ),
        "spearman": (
            signal_spearman
        )
    }

    print(
        f"{signal_name:<20} | "
        f"Pearson: "
        f"{signal_pearson:+.4f} | "
        f"Spearman: "
        f"{signal_spearman:+.4f}"
    )


# ============================================================
# Results by seed
# ============================================================

print()
print(
    "Gradient-Signal Correlations by Seed"
)
print("=" * 85)

for seed in SEEDS:

    seed_records = [
        record
        for record in all_records
        if record["seed"] == seed
    ]

    print()
    print(
        f"Seed {seed}"
    )

    for signal_name in SIGNALS:

        signal_pearson, signal_spearman = (
            calculate_signal_correlations(
                seed_records,
                signal_name
            )
        )

        print(
            f"  {signal_name:<20} | "
            f"Pearson: "
            f"{signal_pearson:+.4f} | "
            f"Spearman: "
            f"{signal_spearman:+.4f}"
        )


# ============================================================
# Results by transition
# ============================================================

print()
print(
    "Gradient-Signal Correlations by Transition"
)
print("=" * 85)

TRANSITIONS = [
    "1->2",
    "2->3",
    "3->4",
    "4->5"
]

for transition_name in TRANSITIONS:

    transition_records = [
        record
        for record in all_records
        if (
            record["transition"]
            == transition_name
        )
    ]

    print()
    print(
        f"Transition "
        f"{transition_name}"
    )

    for signal_name in SIGNALS:

        signal_pearson, signal_spearman = (
            calculate_signal_correlations(
                transition_records,
                signal_name
            )
        )

        print(
            f"  {signal_name:<20} | "
            f"Pearson: "
            f"{signal_pearson:+.4f} | "
            f"Spearman: "
            f"{signal_spearman:+.4f}"
        )


# ============================================================
# Find strongest signal
# ============================================================

best_signal_name = None
best_signal_score = -1.0

for signal_name, result in (
    overall_results.items()
):

    combined_strength = (
        abs(
            result["pearson"]
        )
        + abs(
            result["spearman"]
        )
    ) / 2.0

    if (
        combined_strength
        > best_signal_score
    ):

        best_signal_score = (
            combined_strength
        )

        best_signal_name = (
            signal_name
        )


best_result = (
    overall_results[
        best_signal_name
    ]
)


# ============================================================
# Summary
# ============================================================

print()
print(
    "Best Gradient Signal"
)
print("=" * 85)

print(
    f"Signal: "
    f"{best_signal_name}"
)

print(
    f"Pearson Correlation: "
    f"{best_result['pearson']:+.4f}"
)

print(
    f"Spearman Correlation: "
    f"{best_result['spearman']:+.4f}"
)

print(
    f"Mean Absolute Correlation: "
    f"{best_signal_score:.4f}"
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
print(
    "Gradient Diagnostic Interpretation"
)
print("=" * 85)

if best_signal_score >= 0.50:

    print(
        "RESULT: At least one gradient-based "
        "signal has a strong relationship "
        "with forgetting."
    )

    print(
        "This signal is a promising candidate "
        "for adaptive retention."
    )

elif best_signal_score >= 0.30:

    print(
        "RESULT: Gradient information contains "
        "useful information about forgetting, "
        "but prediction is only moderate."
    )

    print(
        "The strongest gradient signal should "
        "be tested together with other "
        "principled signals."
    )

else:

    print(
        "RESULT: Gradient-based signals are weak "
        "predictors of forgetting in this setup."
    )

    print(
        "They should not be used as the main "
        "adaptive retention mechanism."
    )