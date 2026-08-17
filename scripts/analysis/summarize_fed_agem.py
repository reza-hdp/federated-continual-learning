from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

SEEDS = [
    42,
    123,
    2026,
    777,
    1001,
]

LEARNING_RATE = 0.001
ALPHA = 0.5
ROUNDS_PER_TASK = 3

RESULTS_DIR = Path("results")

OUTPUT_FILE = (
    RESULTS_DIR
    / "fcl_fed_agem_5seed_summary.csv"
)


# ============================================================
# Compute metrics from an accuracy matrix
# ============================================================

def compute_metrics(csv_path):
    dataframe = pd.read_csv(
        csv_path
    )

    task_columns = [
        "task_1",
        "task_2",
        "task_3",
        "task_4",
        "task_5",
    ]

    accuracy_matrix = []

    for _, row in dataframe.iterrows():

        task_accuracies = []

        for column in task_columns:

            value = row[column]

            if not pd.isna(value):
                task_accuracies.append(
                    float(value)
                )

        accuracy_matrix.append(
            task_accuracies
        )


    # --------------------------------------------------------
    # Final average accuracy
    # --------------------------------------------------------

    final_accuracies = (
        accuracy_matrix[-1]
    )

    final_average_accuracy = (
        np.mean(
            final_accuracies
        )
    )


    # --------------------------------------------------------
    # Average forgetting
    # --------------------------------------------------------

    forgetting_scores = []

    number_of_tasks = len(
        final_accuracies
    )


    # The final task has no subsequent task from which
    # forgetting can be measured.
    for task_id in range(
        number_of_tasks - 1
    ):

        previous_accuracies = []

        for row in accuracy_matrix:

            if len(row) > task_id:
                previous_accuracies.append(
                    row[task_id]
                )


        best_previous_accuracy = max(
            previous_accuracies
        )

        final_accuracy = (
            final_accuracies[
                task_id
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


    average_forgetting = (
        np.mean(
            forgetting_scores
        )
    )


    return (
        float(final_average_accuracy),
        float(average_forgetting),
    )


# ============================================================
# Read five seeds
# ============================================================

records = []


print()

print(
    "Fed-A-GEM Five-Seed Summary"
)

print("=" * 80)


for seed in SEEDS:

    filename = (
        "fcl_fed_agem_"
        f"lr0.001_"
        f"alpha0.5_"
        f"r3_"
        f"seed{seed}.csv"
    )

    csv_path = (
        RESULTS_DIR
        / filename
    )


    if not csv_path.exists():

        raise FileNotFoundError(
            f"Missing result file: "
            f"{csv_path}"
        )


    (
        final_accuracy,
        average_forgetting,
    ) = compute_metrics(
        csv_path
    )


    records.append({
        "seed": seed,
        "final_average_accuracy": (
            final_accuracy
        ),
        "average_forgetting": (
            average_forgetting
        ),
    })


    print(
        f"Seed {seed:<4} | "
        f"Final Accuracy: "
        f"{final_accuracy:6.2f}% | "
        f"Average Forgetting: "
        f"{average_forgetting:6.2f} pp"
    )


# ============================================================
# Descriptive statistics
# ============================================================

summary_dataframe = pd.DataFrame(
    records
)


accuracy_values = (
    summary_dataframe[
        "final_average_accuracy"
    ].to_numpy()
)

forgetting_values = (
    summary_dataframe[
        "average_forgetting"
    ].to_numpy()
)


accuracy_mean = np.mean(
    accuracy_values
)

accuracy_std = np.std(
    accuracy_values,
    ddof=1
)


forgetting_mean = np.mean(
    forgetting_values
)

forgetting_std = np.std(
    forgetting_values,
    ddof=1
)


print()

print(
    "Five-Seed Descriptive Statistics"
)

print("=" * 80)

print(
    "Final Average Accuracy: "
    f"{accuracy_mean:.2f} "
    f"± {accuracy_std:.2f}%"
)

print(
    "Average Forgetting: "
    f"{forgetting_mean:.2f} "
    f"± {forgetting_std:.2f} pp"
)


# ============================================================
# Add aggregate row
# ============================================================

aggregate_row = pd.DataFrame([
    {
        "seed": "mean",
        "final_average_accuracy": (
            accuracy_mean
        ),
        "average_forgetting": (
            forgetting_mean
        ),
        "accuracy_std": (
            accuracy_std
        ),
        "forgetting_std": (
            forgetting_std
        ),
    }
])


summary_dataframe[
    "accuracy_std"
] = np.nan

summary_dataframe[
    "forgetting_std"
] = np.nan


output_dataframe = pd.concat(
    [
        summary_dataframe,
        aggregate_row,
    ],
    ignore_index=True,
)


# ============================================================
# Save
# ============================================================

output_dataframe.to_csv(
    OUTPUT_FILE,
    index=False
)


print()

print(
    f"Summary saved to "
    f"{OUTPUT_FILE}"
)