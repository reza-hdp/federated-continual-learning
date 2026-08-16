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
    calculate_conflict_score,
    cosine_similarity
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
    "cuda" if torch.cuda.is_available()
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
# Model and server
# --------------------------------

global_model = SimpleCNN()

server = FederatedServer(
    global_model=global_model,
    device=device
)

criterion = nn.CrossEntropyLoss()


# --------------------------------
# Local training helper
# --------------------------------

def train_local_model(
    initial_state,
    data_loader
):
    local_model = SimpleCNN().to(device)

    local_model.load_state_dict(
        initial_state
    )

    optimizer = optim.Adam(
        local_model.parameters(),
        lr=LEARNING_RATE
    )

    local_model.train()

    for images, labels in data_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = local_model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()
        optimizer.step()

    return {
        name: value.detach().cpu().clone()
        for name, value
        in local_model.state_dict().items()
    }


# --------------------------------
# Helper: copy model state
# --------------------------------

def copy_model_state(model):
    return {
        name: value.detach().cpu().clone()
        for name, value
        in model.state_dict().items()
    }


# --------------------------------
# Train Task 1 normally
# --------------------------------

print()
print("Training Task 1")
print("=" * 75)

task_1_loaders = federated_tasks[0]

for round_number in range(
    1,
    ROUNDS_TASK_1 + 1
):

    print(
        f"Federated Round "
        f"{round_number}/{ROUNDS_TASK_1}"
    )

    current_global_state = copy_model_state(
        server.global_model
    )

    client_states = []
    client_sizes = []

    for client_id in range(
        NUM_CLIENTS
    ):

        client_loader = (
            task_1_loaders[client_id]
        )

        local_state = train_local_model(
            current_global_state,
            client_loader
        )

        client_states.append(
            local_state
        )

        client_sizes.append(
            len(client_loader.dataset)
        )

    server.aggregate(
        client_states,
        client_sizes
    )


# --------------------------------
# Generate Task 2 client updates
# --------------------------------

print()
print("Generating Task 2 Client Updates")
print("=" * 75)

task_2_loaders = federated_tasks[1]

task_2_global_state = copy_model_state(
    server.global_model
)

task_2_sizes = []
task_2_updates = []

for client_id in range(
    NUM_CLIENTS
):

    client_loader = (
        task_2_loaders[client_id]
    )

    local_state = train_local_model(
        task_2_global_state,
        client_loader
    )

    update_vector = flatten_model_update(
        task_2_global_state,
        local_state
    )

    task_2_updates.append(
        update_vector
    )

    sample_count = len(
        client_loader.dataset
    )

    task_2_sizes.append(
        sample_count
    )

    update_norm = (
        torch.linalg.vector_norm(
            update_vector
        ).item()
    )

    print(
        f"Client {client_id} | "
        f"Samples: {sample_count} | "
        f"Update Norm: "
        f"{update_norm:.4f}"
    )


# --------------------------------
# FedAvg reference update
# --------------------------------

reference_update = (
    calculate_reference_update(
        task_2_updates,
        task_2_sizes
    )
)

reference_norm = (
    torch.linalg.vector_norm(
        reference_update
    ).item()
)

print()
print(
    "Reference Update Norm:",
    f"{reference_norm:.4f}"
)


# --------------------------------
# Client conflict scores
# --------------------------------

print()
print("Task 2 Update-Conflict Diagnostic")
print("=" * 75)

conflict_scores = []
similarities = []

for client_id, update_vector in enumerate(
    task_2_updates
):

    similarity = cosine_similarity(
        update_vector,
        reference_update
    )

    conflict = calculate_conflict_score(
        update_vector,
        reference_update
    )

    similarities.append(
        similarity
    )

    conflict_scores.append(
        conflict
    )

    print(
        f"Client {client_id} | "
        f"Cosine Similarity: "
        f"{similarity:+.4f} | "
        f"Conflict Score: "
        f"{conflict:.4f}"
    )


# --------------------------------
# Summary
# --------------------------------

minimum_conflict = min(
    conflict_scores
)

maximum_conflict = max(
    conflict_scores
)

average_conflict = (
    sum(conflict_scores)
    / len(conflict_scores)
)

conflict_range = (
    maximum_conflict
    - minimum_conflict
)


print()
print("Update-Conflict Summary")
print("=" * 75)

print(
    f"Minimum Conflict: "
    f"{minimum_conflict:.4f}"
)

print(
    f"Maximum Conflict: "
    f"{maximum_conflict:.4f}"
)

print(
    f"Average Conflict: "
    f"{average_conflict:.4f}"
)

print(
    f"Conflict Range:   "
    f"{conflict_range:.4f}"
)

print(
    f"Minimum Similarity: "
    f"{min(similarities):+.4f}"
)

print(
    f"Maximum Similarity: "
    f"{max(similarities):+.4f}"
)


# --------------------------------
# Leave-one-out diagnostic
# --------------------------------

print()
print("Leave-One-Out Conflict")
print("=" * 75)

loo_scores = []

for client_id in range(
    NUM_CLIENTS
):

    other_updates = [
        update
        for index, update
        in enumerate(task_2_updates)
        if index != client_id
    ]

    other_sizes = [
        size
        for index, size
        in enumerate(task_2_sizes)
        if index != client_id
    ]

    other_reference = (
        calculate_reference_update(
            other_updates,
            other_sizes
        )
    )

    loo_conflict = (
        calculate_conflict_score(
            task_2_updates[client_id],
            other_reference
        )
    )

    loo_scores.append(
        loo_conflict
    )

    print(
        f"Client {client_id} | "
        f"Leave-One-Out Conflict: "
        f"{loo_conflict:.4f}"
    )


loo_minimum = min(
    loo_scores
)

loo_maximum = max(
    loo_scores
)

loo_average = (
    sum(loo_scores)
    / len(loo_scores)
)

loo_range = (
    loo_maximum
    - loo_minimum
)


print()
print("Leave-One-Out Summary")
print("=" * 75)

print(
    f"Minimum LOO Conflict: "
    f"{loo_minimum:.4f}"
)

print(
    f"Maximum LOO Conflict: "
    f"{loo_maximum:.4f}"
)

print(
    f"Average LOO Conflict: "
    f"{loo_average:.4f}"
)

print(
    f"LOO Conflict Range:   "
    f"{loo_range:.4f}"
)


# --------------------------------
# Interpretation
# --------------------------------

print()

if loo_range >= 0.10:

    print(
        "RESULT: Update conflict shows strong "
        "client-specific variation."
    )

elif loo_range >= 0.05:

    print(
        "RESULT: Update conflict shows moderate "
        "client-specific variation."
    )

else:

    print(
        "RESULT: Update conflict shows weak "
        "client-specific variation."
    )