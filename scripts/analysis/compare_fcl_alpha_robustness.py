import csv


NUM_TASKS = 5


# ============================================================
# Load accuracy matrix
# ============================================================

def load_accuracy_matrix(
    csv_path
):
    matrix = []

    with open(
        csv_path,
        "r"
    ) as input_file:

        reader = csv.DictReader(
            input_file
        )

        for row in reader:

            accuracies = []

            for task_id in range(
                1,
                NUM_TASKS + 1
            ):

                value = row[
                    f"task_{task_id}"
                ]

                if value != "":

                    accuracies.append(
                        float(value)
                    )

            matrix.append(
                accuracies
            )

    return matrix


# ============================================================
# Calculate metrics
# ============================================================

def calculate_metrics(
    matrix
):
    final_accuracies = (
        matrix[-1]
    )

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


# ============================================================
# Experiment files
# ============================================================

experiments = {
    0.1: {
        "Fixed Replay+LwF": (
            "results/"
            "fcl_replay_lwf_alpha0.1_w1.0_r3_seed42.csv"
        ),
        "Gradient-Balanced": (
            "results/"
            "fcl_gradient_balanced_"
            "alpha0.1_w0.5_1.5_r3_seed42.csv"
        )
    },

    0.5: {
        "Fixed Replay+LwF": (
            "results/"
            "fcl_replay_lwf_w1.0_r3.csv"
        ),
        "Gradient-Balanced": (
            "results/"
            "fcl_gradient_balanced_"
            "w0.5_1.5_r3_seed42.csv"
        )
    },

    1.0: {
        "Fixed Replay+LwF": (
            "results/"
            "fcl_replay_lwf_alpha1.0_w1.0_r3_seed42.csv"
        ),
        "Gradient-Balanced": (
            "results/"
            "fcl_gradient_balanced_"
            "alpha1.0_w0.5_1.5_r3_seed42.csv"
        )
    }
}


# ============================================================
# Calculate all results
# ============================================================

summary_rows = []


for alpha, methods in (
    experiments.items()
):

    fixed_matrix = (
        load_accuracy_matrix(
            methods[
                "Fixed Replay+LwF"
            ]
        )
    )

    adaptive_matrix = (
        load_accuracy_matrix(
            methods[
                "Gradient-Balanced"
            ]
        )
    )

    (
        fixed_accuracy,
        fixed_forgetting
    ) = calculate_metrics(
        fixed_matrix
    )

    (
        adaptive_accuracy,
        adaptive_forgetting
    ) = calculate_metrics(
        adaptive_matrix
    )

    accuracy_difference = (
        adaptive_accuracy
        - fixed_accuracy
    )

    forgetting_reduction = (
        fixed_forgetting
        - adaptive_forgetting
    )

    if fixed_forgetting > 0.0:

        relative_forgetting_reduction = (
            forgetting_reduction
            / fixed_forgetting
            * 100.0
        )

    else:

        relative_forgetting_reduction = (
            0.0
        )


    summary_rows.append({
        "alpha": alpha,
        "fixed_accuracy": (
            fixed_accuracy
        ),
        "adaptive_accuracy": (
            adaptive_accuracy
        ),
        "accuracy_difference": (
            accuracy_difference
        ),
        "fixed_forgetting": (
            fixed_forgetting
        ),
        "adaptive_forgetting": (
            adaptive_forgetting
        ),
        "forgetting_reduction": (
            forgetting_reduction
        ),
        "relative_forgetting_reduction": (
            relative_forgetting_reduction
        )
    })


# ============================================================
# Print robustness table
# ============================================================

print()
print(
    "FCL Heterogeneity Robustness — Seed 42"
)

print("=" * 100)

print(
    f"{'Alpha':<10}"
    f"{'Fixed Acc':<14}"
    f"{'GB Acc':<14}"
    f"{'Acc Diff':<14}"
    f"{'Fixed Forget':<16}"
    f"{'GB Forget':<14}"
    f"{'Forget Red.'}"
)

print("-" * 100)


for row in summary_rows:

    print(
        f"{row['alpha']:<10.1f}"
        f"{row['fixed_accuracy']:<14.2f}"
        f"{row['adaptive_accuracy']:<14.2f}"
        f"{row['accuracy_difference']:<+14.2f}"
        f"{row['fixed_forgetting']:<16.2f}"
        f"{row['adaptive_forgetting']:<14.2f}"
        f"{row['forgetting_reduction']:+.2f}"
    )


# ============================================================
# Relative forgetting reduction
# ============================================================

print()
print(
    "Relative Forgetting Reduction"
)

print("=" * 60)


for row in summary_rows:

    print(
        f"Alpha {row['alpha']:.1f} | "
        f"{row['relative_forgetting_reduction']:.2f}%"
    )


# ============================================================
# Average robustness effect
# ============================================================

average_accuracy_difference = (
    sum(
        row[
            "accuracy_difference"
        ]
        for row in summary_rows
    )
    / len(
        summary_rows
    )
)

average_forgetting_reduction = (
    sum(
        row[
            "forgetting_reduction"
        ]
        for row in summary_rows
    )
    / len(
        summary_rows
    )
)

average_relative_reduction = (
    sum(
        row[
            "relative_forgetting_reduction"
        ]
        for row in summary_rows
    )
    / len(
        summary_rows
    )
)


print()
print(
    "Average Effect Across Alpha Values"
)

print("=" * 60)

print(
    f"Average Accuracy Difference: "
    f"{average_accuracy_difference:+.2f} pp"
)

print(
    f"Average Forgetting Reduction: "
    f"{average_forgetting_reduction:+.2f} pp"
)

print(
    f"Average Relative Forgetting Reduction: "
    f"{average_relative_reduction:.2f}%"
)


# ============================================================
# Save summary CSV
# ============================================================

OUTPUT_FILE = (
    "results/"
    "fcl_alpha_robustness_summary.csv"
)


with open(
    OUTPUT_FILE,
    "w",
    newline=""
) as output_file:

    fieldnames = [
        "alpha",
        "fixed_accuracy",
        "adaptive_accuracy",
        "accuracy_difference",
        "fixed_forgetting",
        "adaptive_forgetting",
        "forgetting_reduction",
        "relative_forgetting_reduction"
    ]

    writer = csv.DictWriter(
        output_file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        summary_rows
    )


print()
print(
    f"Robustness summary saved to "
    f"{OUTPUT_FILE}"
)