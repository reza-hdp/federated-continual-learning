import csv
import statistics


NUM_TASKS = 5


def load_accuracy_matrix(csv_path):
    matrix = []

    with open(csv_path, "r") as file:
        reader = csv.DictReader(file)

        for row in reader:
            accuracies = []

            for task_id in range(1, NUM_TASKS + 1):
                value = row[f"task_{task_id}"]

                if value != "":
                    accuracies.append(
                        float(value)
                    )

            matrix.append(
                accuracies
            )

    return matrix


def calculate_metrics(matrix):
    final_accuracies = matrix[-1]

    final_average_accuracy = (
        sum(final_accuracies)
        / len(final_accuracies)
    )

    forgetting_scores = []

    for task_id in range(
        len(matrix) - 1
    ):
        best_previous_accuracy = max(
            row[task_id]
            for row in matrix
            if len(row) > task_id
        )

        final_accuracy = (
            matrix[-1][task_id]
        )

        forgetting = max(
            0.0,
            best_previous_accuracy
            - final_accuracy
        )

        forgetting_scores.append(
            forgetting
        )

    average_forgetting = (
        sum(forgetting_scores)
        / len(forgetting_scores)
    )

    return (
        final_average_accuracy,
        average_forgetting
    )


def evaluate_method(file_paths):
    accuracies = []
    forgetting_values = []

    for file_path in file_paths:
        matrix = load_accuracy_matrix(
            file_path
        )

        accuracy, forgetting = (
            calculate_metrics(
                matrix
            )
        )

        accuracies.append(
            accuracy
        )

        forgetting_values.append(
            forgetting
        )

    return {
        "accuracies": accuracies,
        "forgetting": forgetting_values,
        "accuracy_mean": statistics.mean(
            accuracies
        ),
        "accuracy_std": statistics.stdev(
            accuracies
        ),
        "forgetting_mean": statistics.mean(
            forgetting_values
        ),
        "forgetting_std": statistics.stdev(
            forgetting_values
        )
    }


# ============================================================
# Result files
# ============================================================

methods = {
    "FedAvg-FCL": [
        "results/fcl_fedavg_baseline_r3.csv",
        "results/fcl_fedavg_baseline_r3_seed123.csv",
        "results/fcl_fedavg_baseline_r3_seed2026.csv"
    ],

    "Replay": [
        "results/fcl_replay_r3_seed42.csv",
        "results/fcl_replay_r3_seed123.csv",
        "results/fcl_replay_r3_seed2026.csv"
    ],

    "LwF": [
        "results/fcl_lwf_r3_seed42.csv",
        "results/fcl_lwf_r3_seed123.csv",
        "results/fcl_lwf_r3_seed2026.csv"
    ],

    "Replay+LwF": [
        "results/fcl_replay_lwf_w1.0_r3.csv",
        "results/fcl_replay_lwf_w1.0_r3_seed123.csv",
        "results/fcl_replay_lwf_w1.0_r3_seed2026.csv"
    ],

    "Gradient-Balanced": [
        "results/fcl_gradient_balanced_w0.5_1.5_r3_seed42.csv",
        "results/fcl_gradient_balanced_w0.5_1.5_r3_seed123.csv",
        "results/fcl_gradient_balanced_w0.5_1.5_r3_seed2026.csv"
    ]
}


# ============================================================
# Calculate all results
# ============================================================

results = {}

for method_name, file_paths in methods.items():

    results[method_name] = (
        evaluate_method(
            file_paths
        )
    )


# ============================================================
# Individual seed results
# ============================================================

seeds = [
    42,
    123,
    2026
]

print()
print(
    "Individual Seed Results"
)

print("=" * 90)

for method_name, result in (
    results.items()
):

    print()
    print(
        method_name
    )

    for index, seed in enumerate(
        seeds
    ):

        print(
            f"Seed {seed} | "
            f"Accuracy: "
            f"{result['accuracies'][index]:.2f}% | "
            f"Forgetting: "
            f"{result['forgetting'][index]:.2f} pp"
        )


# ============================================================
# Final comparison
# ============================================================

print()
print(
    "Final FCL Comparison — Mean ± Standard Deviation"
)

print("=" * 90)

print(
    f"{'Method':<22}"
    f"{'Final Avg Accuracy':<30}"
    f"{'Average Forgetting'}"
)

print("-" * 90)

for method_name, result in (
    results.items()
):

    accuracy_text = (
        f"{result['accuracy_mean']:.2f} "
        f"± "
        f"{result['accuracy_std']:.2f}%"
    )

    forgetting_text = (
        f"{result['forgetting_mean']:.2f} "
        f"± "
        f"{result['forgetting_std']:.2f} pp"
    )

    print(
        f"{method_name:<22}"
        f"{accuracy_text:<30}"
        f"{forgetting_text}"
    )


# ============================================================
# Compare Gradient-Balanced with FedAvg
# ============================================================

fedavg = results[
    "FedAvg-FCL"
]

gradient_balanced = results[
    "Gradient-Balanced"
]

accuracy_gain_vs_fedavg = (
    gradient_balanced[
        "accuracy_mean"
    ]
    - fedavg[
        "accuracy_mean"
    ]
)

forgetting_reduction_vs_fedavg = (
    fedavg[
        "forgetting_mean"
    ]
    - gradient_balanced[
        "forgetting_mean"
    ]
)


# ============================================================
# Compare Gradient-Balanced with fixed Replay+LwF
# ============================================================

fixed_replay_lwf = results[
    "Replay+LwF"
]

accuracy_difference_vs_fixed = (
    gradient_balanced[
        "accuracy_mean"
    ]
    - fixed_replay_lwf[
        "accuracy_mean"
    ]
)

forgetting_difference_vs_fixed = (
    fixed_replay_lwf[
        "forgetting_mean"
    ]
    - gradient_balanced[
        "forgetting_mean"
    ]
)


# ============================================================
# Relative forgetting reduction
# ============================================================

if (
    fixed_replay_lwf[
        "forgetting_mean"
    ] > 0.0
):

    relative_forgetting_reduction = (
        forgetting_difference_vs_fixed
        / fixed_replay_lwf[
            "forgetting_mean"
        ]
        * 100.0
    )

else:

    relative_forgetting_reduction = (
        0.0
    )


# ============================================================
# Print improvements
# ============================================================

print()
print(
    "Gradient-Balanced vs FedAvg-FCL"
)

print("=" * 90)

print(
    f"Accuracy Improvement: "
    f"{accuracy_gain_vs_fedavg:+.2f} "
    f"percentage points"
)

print(
    f"Forgetting Reduction: "
    f"{forgetting_reduction_vs_fedavg:+.2f} "
    f"percentage points"
)


print()
print(
    "Gradient-Balanced vs Fixed Replay+LwF"
)

print("=" * 90)

print(
    f"Accuracy Difference: "
    f"{accuracy_difference_vs_fixed:+.2f} "
    f"percentage points"
)

print(
    f"Forgetting Reduction: "
    f"{forgetting_difference_vs_fixed:+.2f} "
    f"percentage points"
)

print(
    f"Relative Forgetting Reduction: "
    f"{relative_forgetting_reduction:.2f}%"
)


# ============================================================
# Save publication-ready summary CSV
# ============================================================

output_path = (
    "results/"
    "fcl_final_comparison.csv"
)

with open(
    output_path,
    "w",
    newline=""
) as output_file:

    fieldnames = [
        "method",
        "accuracy_mean",
        "accuracy_std",
        "forgetting_mean",
        "forgetting_std"
    ]

    writer = csv.DictWriter(
        output_file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for method_name, result in (
        results.items()
    ):

        writer.writerow({
            "method": method_name,
            "accuracy_mean": (
                result[
                    "accuracy_mean"
                ]
            ),
            "accuracy_std": (
                result[
                    "accuracy_std"
                ]
            ),
            "forgetting_mean": (
                result[
                    "forgetting_mean"
                ]
            ),
            "forgetting_std": (
                result[
                    "forgetting_std"
                ]
            )
        })


print()
print(
    f"Summary saved to "
    f"{output_path}"
)