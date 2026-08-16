import csv
import os

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Configuration
# ============================================================

os.makedirs(
    "../../figures",
    exist_ok=True
)

SEEDS = [
    42,
    123,
    2026,
    777,
    1001
]

WEIGHT_FILES = [
    "results/"
    "fcl_gradient_balanced_weights_"
    "w0.5_1.5_r3_seed42.csv",

    "results/"
    "fcl_gradient_balanced_weights_"
    "w0.5_1.5_r3_seed123.csv",

    "results/"
    "fcl_gradient_balanced_weights_"
    "w0.5_1.5_r3_seed2026.csv",

    "results/"
    "fcl_gradient_balanced_weights_"
    "w0.5_1.5_r3_seed777.csv",

    "results/"
    "fcl_gradient_balanced_weights_"
    "w0.5_1.5_r3_seed1001.csv"
]


# ============================================================
# Load records
# ============================================================

records = []

for seed, file_path in zip(
    SEEDS,
    WEIGHT_FILES
):

    with open(
        file_path,
        "r"
    ) as input_file:

        reader = csv.DictReader(
            input_file
        )

        for row in reader:

            task = int(
                row["task"]
            )

            round_number = int(
                row["round"]
            )

            client = int(
                row["client"]
            )

            retention_weight = float(
                row["retention_weight"]
            )

            balance_score = float(
                row["balance_score"]
            )

            magnitude_ratio = float(
                row["magnitude_ratio"]
            )

            records.append({
                "seed": seed,
                "task": task,
                "round": round_number,
                "client": client,
                "retention_weight": (
                    retention_weight
                ),
                "balance_score": (
                    balance_score
                ),
                "magnitude_ratio": (
                    magnitude_ratio
                )
            })


# ============================================================
# Remove Task 1
# ============================================================
# Task 1 has no old memory, so its retention weight is 0.
# We analyze Tasks 2-5 only.
# ============================================================

adaptive_records = [
    record
    for record in records
    if record["task"] > 1
]


# ============================================================
# Helper
# ============================================================

def mean_and_std(
    values
):
    values_array = np.asarray(
        values,
        dtype=float
    )

    mean_value = float(
        np.mean(
            values_array
        )
    )

    if len(values_array) > 1:

        std_value = float(
            np.std(
                values_array,
                ddof=1
            )
        )

    else:

        std_value = 0.0

    return (
        mean_value,
        std_value
    )


# ============================================================
# Overall statistics
# ============================================================

all_weights = [
    record["retention_weight"]
    for record in adaptive_records
]

overall_mean, overall_std = (
    mean_and_std(
        all_weights
    )
)

overall_min = min(
    all_weights
)

overall_max = max(
    all_weights
)


print()
print(
    "Adaptive Retention Weight Summary"
)

print("=" * 75)

print(
    f"Number of adaptive observations: "
    f"{len(all_weights)}"
)

print(
    f"Minimum weight: "
    f"{overall_min:.4f}"
)

print(
    f"Maximum weight: "
    f"{overall_max:.4f}"
)

print(
    f"Mean weight: "
    f"{overall_mean:.4f}"
)

print(
    f"Standard deviation: "
    f"{overall_std:.4f}"
)


# ============================================================
# Weight by task
# ============================================================

tasks = [
    2,
    3,
    4,
    5
]

task_means = []
task_stds = []


print()
print(
    "Retention Weight by Task"
)

print("=" * 75)


for task in tasks:

    task_weights = [
        record[
            "retention_weight"
        ]
        for record in adaptive_records
        if record["task"] == task
    ]

    task_mean, task_std = (
        mean_and_std(
            task_weights
        )
    )

    task_means.append(
        task_mean
    )

    task_stds.append(
        task_std
    )

    print(
        f"Task {task} | "
        f"Mean: "
        f"{task_mean:.4f} | "
        f"Std: "
        f"{task_std:.4f} | "
        f"Min: "
        f"{min(task_weights):.4f} | "
        f"Max: "
        f"{max(task_weights):.4f}"
    )


# ============================================================
# Weight by round
# ============================================================

rounds = [
    1,
    2,
    3
]

round_means = []
round_stds = []


print()
print(
    "Retention Weight by Round"
)

print("=" * 75)


for round_number in rounds:

    round_weights = [
        record[
            "retention_weight"
        ]
        for record in adaptive_records
        if (
            record["round"]
            == round_number
        )
    ]

    round_mean, round_std = (
        mean_and_std(
            round_weights
        )
    )

    round_means.append(
        round_mean
    )

    round_stds.append(
        round_std
    )

    print(
        f"Round {round_number} | "
        f"Mean: "
        f"{round_mean:.4f} | "
        f"Std: "
        f"{round_std:.4f}"
    )


# ============================================================
# Weight by client
# ============================================================

clients = [
    0,
    1,
    2,
    3,
    4
]

client_means = []
client_stds = []


print()
print(
    "Retention Weight by Client"
)

print("=" * 75)


for client in clients:

    client_weights = [
        record[
            "retention_weight"
        ]
        for record in adaptive_records
        if record["client"] == client
    ]

    client_mean, client_std = (
        mean_and_std(
            client_weights
        )
    )

    client_means.append(
        client_mean
    )

    client_stds.append(
        client_std
    )

    print(
        f"Client {client} | "
        f"Mean: "
        f"{client_mean:.4f} | "
        f"Std: "
        f"{client_std:.4f}"
    )


# ============================================================
# Figure 1
# Mean adaptive weight by task
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 5)
)

ax.errorbar(
    tasks,
    task_means,
    yerr=task_stds,
    marker="o",
    linewidth=2,
    capsize=5
)

ax.axhline(
    y=1.0,
    linestyle="--",
    linewidth=1.5,
    label="Fixed Replay+LwF weight"
)

ax.set_xlabel(
    "Continual Task"
)

ax.set_ylabel(
    "Retention Weight"
)

ax.set_title(
    "Adaptive Retention Weight Across Tasks"
)

ax.set_xticks(
    tasks
)

ax.set_ylim(
    0.5,
    1.5
)

ax.grid(
    alpha=0.25
)

ax.legend()

fig.tight_layout()

fig.savefig(
    "figures/"
    "fcl_retention_weight_by_task.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    "figures/"
    "fcl_retention_weight_by_task.pdf",
    bbox_inches="tight"
)

plt.close(
    fig
)


# ============================================================
# Figure 2
# Mean adaptive weight by federated round
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 5)
)

ax.errorbar(
    rounds,
    round_means,
    yerr=round_stds,
    marker="o",
    linewidth=2,
    capsize=5
)

ax.axhline(
    y=1.0,
    linestyle="--",
    linewidth=1.5,
    label="Fixed Replay+LwF weight"
)

ax.set_xlabel(
    "Federated Round"
)

ax.set_ylabel(
    "Retention Weight"
)

ax.set_title(
    "Adaptive Retention Weight Across Federated Rounds"
)

ax.set_xticks(
    rounds
)

ax.set_ylim(
    0.5,
    1.5
)

ax.grid(
    alpha=0.25
)

ax.legend()

fig.tight_layout()

fig.savefig(
    "figures/"
    "fcl_retention_weight_by_round.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    "figures/"
    "fcl_retention_weight_by_round.pdf",
    bbox_inches="tight"
)

plt.close(
    fig
)


# ============================================================
# Figure 3
# Client-specific mean weights
# ============================================================

client_positions = np.arange(
    len(clients)
)

fig, ax = plt.subplots(
    figsize=(8, 5)
)

bars = ax.bar(
    client_positions,
    client_means,
    yerr=client_stds,
    capsize=5
)

ax.axhline(
    y=1.0,
    linestyle="--",
    linewidth=1.5,
    label="Fixed Replay+LwF weight"
)

ax.set_xticks(
    client_positions
)

ax.set_xticklabels([
    f"Client {client}"
    for client in clients
])

ax.set_ylabel(
    "Mean Retention Weight"
)

ax.set_title(
    "Client-Specific Adaptive Retention"
)

ax.set_ylim(
    0.5,
    1.5
)

ax.grid(
    axis="y",
    alpha=0.25
)

ax.legend()

for bar, mean_value in zip(
    bars,
    client_means
):

    ax.text(
        bar.get_x()
        + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{mean_value:.2f}",
        ha="center",
        va="bottom",
        fontsize=9
    )

fig.tight_layout()

fig.savefig(
    "figures/"
    "fcl_retention_weight_by_client.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    "figures/"
    "fcl_retention_weight_by_client.pdf",
    bbox_inches="tight"
)

plt.close(
    fig
)


# ============================================================
# Figure 4
# Task x round weight matrix
# ============================================================

weight_matrix = np.zeros(
    (
        len(tasks),
        len(rounds)
    )
)


for task_index, task in enumerate(
    tasks
):

    for round_index, round_number in enumerate(
        rounds
    ):

        values = [
            record[
                "retention_weight"
            ]
            for record in adaptive_records
            if (
                record["task"] == task
                and record["round"]
                == round_number
            )
        ]

        weight_matrix[
            task_index,
            round_index
        ] = np.mean(
            values
        )


fig, ax = plt.subplots(
    figsize=(7, 5)
)

image = ax.imshow(
    weight_matrix,
    aspect="auto"
)

ax.set_xticks(
    np.arange(
        len(rounds)
    )
)

ax.set_xticklabels([
    f"Round {round_number}"
    for round_number in rounds
])

ax.set_yticks(
    np.arange(
        len(tasks)
    )
)

ax.set_yticklabels([
    f"Task {task}"
    for task in tasks
])

ax.set_xlabel(
    "Federated Round"
)

ax.set_ylabel(
    "Continual Task"
)

ax.set_title(
    "Mean Adaptive Retention Weight"
)

for task_index in range(
    len(tasks)
):

    for round_index in range(
        len(rounds)
    ):

        ax.text(
            round_index,
            task_index,
            f"{weight_matrix[task_index, round_index]:.2f}",
            ha="center",
            va="center"
        )

fig.colorbar(
    image,
    ax=ax,
    label="Retention Weight"
)

fig.tight_layout()

fig.savefig(
    "figures/"
    "fcl_retention_weight_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    "figures/"
    "fcl_retention_weight_heatmap.pdf",
    bbox_inches="tight"
)

plt.close(
    fig
)


# ============================================================
# Figure 5
# Magnitude ratio vs retention weight
# ============================================================

magnitude_ratios = np.asarray([
    record["magnitude_ratio"]
    for record in adaptive_records
])

retention_weights = np.asarray([
    record["retention_weight"]
    for record in adaptive_records
])


fig, ax = plt.subplots(
    figsize=(7, 5)
)

ax.scatter(
    magnitude_ratios,
    retention_weights,
    alpha=0.65
)

ax.set_xlabel(
    "Old/New Gradient Magnitude Ratio"
)

ax.set_ylabel(
    "Adaptive Retention Weight"
)

ax.set_title(
    "Gradient Balance Controls Retention Strength"
)

ax.grid(
    alpha=0.25
)

fig.tight_layout()

fig.savefig(
    "figures/"
    "fcl_gradient_ratio_vs_weight.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    "figures/"
    "fcl_gradient_ratio_vs_weight.pdf",
    bbox_inches="tight"
)

plt.close(
    fig
)


# ============================================================
# Finished
# ============================================================

print()
print("=" * 75)

print(
    "Adaptive-weight figures created successfully."
)

print("=" * 75)

print(
    "figures/fcl_retention_weight_by_task.png"
)

print(
    "figures/fcl_retention_weight_by_round.png"
)

print(
    "figures/fcl_retention_weight_by_client.png"
)

print(
    "figures/fcl_retention_weight_heatmap.png"
)

print(
    "figures/fcl_gradient_ratio_vs_weight.png"
)

print()

print(
    "PDF versions were also created."
)