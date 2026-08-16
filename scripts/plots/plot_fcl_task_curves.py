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

NUM_TASKS = 5

SEEDS = [
    42,
    123,
    2026,
    777,
    1001
]


FIXED_FILES = [
    "results/fcl_replay_lwf_w1.0_r3.csv",
    "results/fcl_replay_lwf_w1.0_r3_seed123.csv",
    "results/fcl_replay_lwf_w1.0_r3_seed2026.csv",
    "results/fcl_replay_lwf_w1.0_r3_seed777.csv",
    "results/fcl_replay_lwf_w1.0_r3_seed1001.csv"
]


GRADIENT_BALANCED_FILES = [
    "results/fcl_gradient_balanced_w0.5_1.5_r3_seed42.csv",
    "results/fcl_gradient_balanced_w0.5_1.5_r3_seed123.csv",
    "results/fcl_gradient_balanced_w0.5_1.5_r3_seed2026.csv",
    "results/fcl_gradient_balanced_w0.5_1.5_r3_seed777.csv",
    "results/fcl_gradient_balanced_w0.5_1.5_r3_seed1001.csv"
]


# ============================================================
# Load one triangular accuracy matrix
# ============================================================

def load_accuracy_matrix(
    file_path
):
    matrix = []

    with open(
        file_path,
        "r"
    ) as input_file:

        reader = csv.DictReader(
            input_file
        )

        for row in reader:

            values = []

            for task_id in range(
                1,
                NUM_TASKS + 1
            ):

                value = row[
                    f"task_{task_id}"
                ]

                if value != "":
                    values.append(
                        float(value)
                    )

            matrix.append(
                values
            )

    return matrix


# ============================================================
# Load all seeds
# ============================================================

def load_all_seed_matrices(
    file_paths
):
    return [
        load_accuracy_matrix(
            file_path
        )
        for file_path in file_paths
    ]


fixed_matrices = (
    load_all_seed_matrices(
        FIXED_FILES
    )
)

adaptive_matrices = (
    load_all_seed_matrices(
        GRADIENT_BALANCED_FILES
    )
)


# ============================================================
# Mean and sample standard deviation helper
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
# Task-wise means across seeds
# ============================================================

def calculate_task_curves(
    matrices
):
    means = [
        []
        for _ in range(
            NUM_TASKS
        )
    ]

    stds = [
        []
        for _ in range(
            NUM_TASKS
        )
    ]

    for task_index in range(
        NUM_TASKS
    ):

        for stage_index in range(
            task_index,
            NUM_TASKS
        ):

            values = [
                matrix[
                    stage_index
                ][
                    task_index
                ]
                for matrix in matrices
            ]

            mean_value, std_value = (
                mean_and_std(
                    values
                )
            )

            means[
                task_index
            ].append(
                mean_value
            )

            stds[
                task_index
            ].append(
                std_value
            )

    return (
        means,
        stds
    )


(
    fixed_task_means,
    fixed_task_stds
) = calculate_task_curves(
    fixed_matrices
)

(
    adaptive_task_means,
    adaptive_task_stds
) = calculate_task_curves(
    adaptive_matrices
)


# ============================================================
# Figure 1
# Task-wise retention curves
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 6)
)


for task_index in range(
    NUM_TASKS
):

    x_values = np.arange(
        task_index + 1,
        NUM_TASKS + 1
    )

    ax.plot(
        x_values,
        fixed_task_means[
            task_index
        ],
        marker="o",
        linewidth=1.8,
        label=(
            f"Fixed — Task "
            f"{task_index + 1}"
        )
    )

    ax.plot(
        x_values,
        adaptive_task_means[
            task_index
        ],
        marker="s",
        linewidth=1.8,
        linestyle="--",
        label=(
            f"GB — Task "
            f"{task_index + 1}"
        )
    )


ax.set_xlabel(
    "After Learning Task"
)

ax.set_ylabel(
    "Accuracy (%)"
)

ax.set_title(
    "Task-Wise Retention Across Continual Learning"
)

ax.set_xticks(
    np.arange(
        1,
        NUM_TASKS + 1
    )
)

ax.set_ylim(
    0,
    100
)

ax.grid(
    alpha=0.25
)

ax.legend(
    ncol=2,
    fontsize=8
)

fig.tight_layout()

fig.savefig(
    "figures/"
    "fcl_task_retention_curves.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    "figures/"
    "fcl_task_retention_curves.pdf",
    bbox_inches="tight"
)

plt.close(
    fig
)


# ============================================================
# Figure 2
# Mean accuracy after each continual stage
# ============================================================

fixed_stage_means = []
fixed_stage_stds = []

adaptive_stage_means = []
adaptive_stage_stds = []


for stage_index in range(
    NUM_TASKS
):

    fixed_seed_stage_values = []

    adaptive_seed_stage_values = []

    for matrix in fixed_matrices:

        stage_values = matrix[
            stage_index
        ]

        fixed_seed_stage_values.append(
            sum(stage_values)
            / len(stage_values)
        )

    for matrix in adaptive_matrices:

        stage_values = matrix[
            stage_index
        ]

        adaptive_seed_stage_values.append(
            sum(stage_values)
            / len(stage_values)
        )

    fixed_mean, fixed_std = (
        mean_and_std(
            fixed_seed_stage_values
        )
    )

    adaptive_mean, adaptive_std = (
        mean_and_std(
            adaptive_seed_stage_values
        )
    )

    fixed_stage_means.append(
        fixed_mean
    )

    fixed_stage_stds.append(
        fixed_std
    )

    adaptive_stage_means.append(
        adaptive_mean
    )

    adaptive_stage_stds.append(
        adaptive_std
    )


stage_x_values = np.arange(
    1,
    NUM_TASKS + 1
)


fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.errorbar(
    stage_x_values,
    fixed_stage_means,
    yerr=fixed_stage_stds,
    marker="o",
    linewidth=2,
    capsize=5,
    label="Fixed Replay+LwF"
)

ax.errorbar(
    stage_x_values,
    adaptive_stage_means,
    yerr=adaptive_stage_stds,
    marker="s",
    linewidth=2,
    capsize=5,
    label="Gradient-Balanced"
)

ax.set_xlabel(
    "Number of Tasks Learned"
)

ax.set_ylabel(
    "Average Accuracy Over Learned Tasks (%)"
)

ax.set_title(
    "Continual Performance Across Five Seeds"
)

ax.set_xticks(
    stage_x_values
)

ax.set_ylim(
    0,
    100
)

ax.grid(
    alpha=0.25
)

ax.legend()

fig.tight_layout()

fig.savefig(
    "figures/"
    "fcl_stage_average_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    "figures/"
    "fcl_stage_average_accuracy.pdf",
    bbox_inches="tight"
)

plt.close(
    fig
)


# ============================================================
# Figure 3
# Final per-task retention
# ============================================================

fixed_final_means = []
fixed_final_stds = []

adaptive_final_means = []
adaptive_final_stds = []


for task_index in range(
    NUM_TASKS
):

    fixed_values = [
        matrix[-1][task_index]
        for matrix in fixed_matrices
    ]

    adaptive_values = [
        matrix[-1][task_index]
        for matrix in adaptive_matrices
    ]

    fixed_mean, fixed_std = (
        mean_and_std(
            fixed_values
        )
    )

    adaptive_mean, adaptive_std = (
        mean_and_std(
            adaptive_values
        )
    )

    fixed_final_means.append(
        fixed_mean
    )

    fixed_final_stds.append(
        fixed_std
    )

    adaptive_final_means.append(
        adaptive_mean
    )

    adaptive_final_stds.append(
        adaptive_std
    )


bar_width = 0.35

task_positions = np.arange(
    NUM_TASKS
)


fig, ax = plt.subplots(
    figsize=(9, 5)
)

ax.bar(
    task_positions
    - bar_width / 2,
    fixed_final_means,
    width=bar_width,
    yerr=fixed_final_stds,
    capsize=4,
    label="Fixed Replay+LwF"
)

ax.bar(
    task_positions
    + bar_width / 2,
    adaptive_final_means,
    width=bar_width,
    yerr=adaptive_final_stds,
    capsize=4,
    label="Gradient-Balanced"
)

ax.set_xticks(
    task_positions
)

ax.set_xticklabels([
    "Task 1",
    "Task 2",
    "Task 3",
    "Task 4",
    "Task 5"
])

ax.set_ylabel(
    "Final Task Accuracy (%)"
)

ax.set_title(
    "Final Retention of Individual Tasks"
)

ax.set_ylim(
    0,
    100
)

ax.grid(
    axis="y",
    alpha=0.25
)

ax.legend()

fig.tight_layout()

fig.savefig(
    "figures/"
    "fcl_final_task_retention.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    "figures/"
    "fcl_final_task_retention.pdf",
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
    "Task-retention figures created successfully "
    "without NaN warnings."
)

print("=" * 75)

print(
    "figures/fcl_task_retention_curves.png"
)

print(
    "figures/fcl_stage_average_accuracy.png"
)

print(
    "figures/fcl_final_task_retention.png"
)

print()

print(
    "PDF versions were also created."
)