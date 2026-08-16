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

criterion = nn.CrossEntropyLoss()

print()
print("Gradient-Interference Diagnostic")
print("=" * 80)

print(
    f"Device: {device}"
)

print(
    f"Seed: {SEED}"
)


# ============================================================
# Load CIFAR-10
# ============================================================

train_loader, _ = get_cifar10_loaders(
    batch_size=BATCH_SIZE
)

train_dataset = train_loader.dataset


# ============================================================
# Continual tasks
# ============================================================

class_groups = [
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
        class_groups=class_groups,
        batch_size=BATCH_SIZE,
        alpha=ALPHA,
        seed=SEED
    )
)

task_1_loaders = federated_tasks[0]
task_2_loaders = federated_tasks[1]


# ============================================================
# Model and server
# ============================================================

initial_global_model = SimpleCNN()

server = FederatedServer(
    global_model=initial_global_model,
    device=device
)


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
# Train one local model
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
# Train Task 1 globally
# ============================================================

print()
print("Training Global Model on Task 1")
print("=" * 80)

for round_number in range(
    1,
    ROUNDS_TASK_1 + 1
):

    print(
        f"Federated Round "
        f"{round_number}/"
        f"{ROUNDS_TASK_1}"
    )

    current_state = (
        copy_model_state(
            server.global_model
        )
    )

    round_client_states = []
    round_client_sizes = []

    for client_id in range(
        NUM_CLIENTS
    ):

        trained_client = (
            train_client_model(
                current_state,
                task_1_loaders[
                    client_id
                ]
            )
        )

        trained_state = (
            copy_model_state(
                trained_client
            )
        )

        round_client_states.append(
            trained_state
        )

        client_sample_count = len(
            task_1_loaders[
                client_id
            ].dataset
        )

        round_client_sizes.append(
            client_sample_count
        )

    server.aggregate(
        round_client_states,
        round_client_sizes
    )


# ============================================================
# Global model after Task 1
# ============================================================

task_1_state = (
    copy_model_state(
        server.global_model
    )
)


# ============================================================
# Compute gradient from one loader
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

        batch_size = int(
            labels.numel()
        )

        weighted_loss = (
            loss * batch_size
        )

        weighted_loss.backward()

        total_samples += batch_size

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
# Calculate Task 1 vs Task 2 gradients
# ============================================================

print()
print(
    "Old-Task vs New-Task Gradient Interference"
)
print("=" * 80)

diagnostic_results = []

for client_id in range(
    NUM_CLIENTS
):

    old_gradient = (
        calculate_loader_gradient(
            task_1_state,
            task_1_loaders[
                client_id
            ]
        )
    )

    new_gradient = (
        calculate_loader_gradient(
            task_1_state,
            task_2_loaders[
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

    diagnostic_results.append(
        diagnostic
    )

    print(
        f"Client {client_id} | "
        f"Cosine: "
        f"{diagnostic['cosine_similarity']:+.4f} | "
        f"Interference: "
        f"{diagnostic['interference']:.4f} | "
        f"Old Norm: "
        f"{diagnostic['old_gradient_norm']:.4f} | "
        f"New Norm: "
        f"{diagnostic['new_gradient_norm']:.4f} | "
        f"Ratio: "
        f"{diagnostic['magnitude_ratio']:.4f}"
    )


# ============================================================
# Extract diagnostic values
# ============================================================

similarities = [
    result[
        "cosine_similarity"
    ]
    for result in diagnostic_results
]

interference_scores = [
    result[
        "interference"
    ]
    for result in diagnostic_results
]

old_norms = [
    result[
        "old_gradient_norm"
    ]
    for result in diagnostic_results
]

new_norms = [
    result[
        "new_gradient_norm"
    ]
    for result in diagnostic_results
]

magnitude_ratios = [
    result[
        "magnitude_ratio"
    ]
    for result in diagnostic_results
]


# ============================================================
# Summary
# ============================================================

print()
print(
    "Gradient-Interference Summary"
)
print("=" * 80)

print(
    f"Minimum Cosine Similarity: "
    f"{min(similarities):+.4f}"
)

print(
    f"Maximum Cosine Similarity: "
    f"{max(similarities):+.4f}"
)

print(
    f"Average Cosine Similarity: "
    f"{sum(similarities) / len(similarities):+.4f}"
)

print()

print(
    f"Minimum Interference: "
    f"{min(interference_scores):.4f}"
)

print(
    f"Maximum Interference: "
    f"{max(interference_scores):.4f}"
)

print(
    f"Average Interference: "
    f"{sum(interference_scores) / len(interference_scores):.4f}"
)

print(
    f"Interference Range: "
    f"{max(interference_scores) - min(interference_scores):.4f}"
)

print()

print(
    f"Minimum Old Gradient Norm: "
    f"{min(old_norms):.4f}"
)

print(
    f"Maximum Old Gradient Norm: "
    f"{max(old_norms):.4f}"
)

print(
    f"Minimum New Gradient Norm: "
    f"{min(new_norms):.4f}"
)

print(
    f"Maximum New Gradient Norm: "
    f"{max(new_norms):.4f}"
)

print()

print(
    f"Minimum Magnitude Ratio: "
    f"{min(magnitude_ratios):.4f}"
)

print(
    f"Maximum Magnitude Ratio: "
    f"{max(magnitude_ratios):.4f}"
)

print(
    f"Magnitude-Ratio Range: "
    f"{max(magnitude_ratios) - min(magnitude_ratios):.4f}"
)


# ============================================================
# Basic interpretation
# ============================================================

negative_count = sum(
    1
    for similarity in similarities
    if similarity < 0.0
)

positive_count = (
    NUM_CLIENTS
    - negative_count
)

nonzero_interference_count = sum(
    1
    for score in interference_scores
    if score > 0.0
)


print()
print(
    "Gradient-Interference Interpretation"
)
print("=" * 80)

print(
    f"Clients with negative cosine similarity: "
    f"{negative_count}/{NUM_CLIENTS}"
)

print(
    f"Clients with non-negative cosine similarity: "
    f"{positive_count}/{NUM_CLIENTS}"
)

print(
    f"Clients with destructive interference: "
    f"{nonzero_interference_count}/{NUM_CLIENTS}"
)


interference_range = (
    max(interference_scores)
    - min(interference_scores)
)

ratio_range = (
    max(magnitude_ratios)
    - min(magnitude_ratios)
)


if (
    nonzero_interference_count >= 2
    and interference_range >= 0.05
):

    print(
        "RESULT: Gradient interference shows "
        "meaningful client-specific variation."
    )

elif (
    nonzero_interference_count >= 1
    and interference_range > 0.0
):

    print(
        "RESULT: Destructive gradient interference "
        "is present, but variation is limited."
    )

elif ratio_range >= 0.10:

    print(
        "RESULT: Directional interference is weak, "
        "but gradient magnitudes vary across clients."
    )

else:

    print(
        "RESULT: This gradient diagnostic shows "
        "little client-specific variation."
    )