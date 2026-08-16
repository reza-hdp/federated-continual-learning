import torch
import torch.nn as nn
import torch.optim as optim

from src.models.cnn import SimpleCNN
from src.data.cifar10 import get_cifar10_loaders
from src.data.federated_continual import (
    create_federated_continual_tasks
)
from src.algorithms.prediction_shift import (
    calculate_prediction_shift
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
ROUNDS = 3
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
# Create global model
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
    global_state,
    data_loader
):
    local_model = SimpleCNN().to(device)

    local_model.load_state_dict(
        global_state
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
        name: parameter.detach().cpu().clone()
        for name, parameter
        in local_model.state_dict().items()
    }


# --------------------------------
# Train global model on Task 1
# --------------------------------

print()
print("Training global model on Task 1")
print("=" * 75)

task_1_loaders = federated_tasks[0]

for round_number in range(
    1,
    ROUNDS + 1
):

    print(
        f"Federated Round "
        f"{round_number}/{ROUNDS}"
    )

    client_states = []
    client_sizes = []

    global_state = (
        server.global_model.state_dict()
    )

    for client_id in range(
        NUM_CLIENTS
    ):

        client_loader = (
            task_1_loaders[client_id]
        )

        local_state = train_local_model(
            global_state,
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
# Measure prediction shift
# --------------------------------

print()
print("Prediction-Shift Diagnostic")
print("=" * 75)

known_scores = []
new_scores = []

task_2_loaders = federated_tasks[1]

for client_id in range(
    NUM_CLIENTS
):

    known_score = calculate_prediction_shift(
        model=server.global_model,
        data_loader=task_1_loaders[
            client_id
        ],
        device=device
    )

    new_score = calculate_prediction_shift(
        model=server.global_model,
        data_loader=task_2_loaders[
            client_id
        ],
        device=device
    )

    known_scores.append(
        known_score
    )

    new_scores.append(
        new_score
    )

    difference = (
        new_score - known_score
    )

    print()
    print(
        f"Client {client_id}"
    )

    print(
        f"Known Task 1 Entropy: "
        f"{known_score:.4f}"
    )

    print(
        f"New Task 2 Entropy:   "
        f"{new_score:.4f}"
    )

    print(
        f"Difference:           "
        f"{difference:+.4f}"
    )


# --------------------------------
# Summary
# --------------------------------

average_known = (
    sum(known_scores)
    / len(known_scores)
)

average_new = (
    sum(new_scores)
    / len(new_scores)
)

average_difference = (
    average_new
    - average_known
)


print()
print("Prediction-Shift Summary")
print("=" * 75)

print(
    f"Average Known-Task Entropy: "
    f"{average_known:.4f}"
)

print(
    f"Average New-Task Entropy:   "
    f"{average_new:.4f}"
)

print(
    f"Average Difference:         "
    f"{average_difference:+.4f}"
)

print(
    f"Minimum New-Task Score:     "
    f"{min(new_scores):.4f}"
)

print(
    f"Maximum New-Task Score:     "
    f"{max(new_scores):.4f}"
)

new_score_range = (
    max(new_scores)
    - min(new_scores)
)

print(
    f"New-Task Score Range:       "
    f"{new_score_range:.4f}"
)


# --------------------------------
# Diagnostic interpretation
# --------------------------------

higher_count = sum(
    1
    for known, new in zip(
        known_scores,
        new_scores
    )
    if new > known
)

print(
    f"Clients with higher entropy "
    f"on new task: "
    f"{higher_count}/{NUM_CLIENTS}"
)

print()

if (
    higher_count >= 4
    and average_difference > 0.05
    and new_score_range > 0.02
):
    print(
        "RESULT: Prediction entropy shows "
        "useful shift information."
    )

elif (
    higher_count >= 4
    and average_difference > 0.05
):
    print(
        "RESULT: New-task entropy is higher, "
        "but client-specific variation is weak."
    )

else:
    print(
        "RESULT: Prediction entropy is not "
        "a reliable shift signal in this setup."
    )