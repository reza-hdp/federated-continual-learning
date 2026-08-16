import csv
import statistics


def load_accuracy_matrix(csv_path):
    matrix = []

    with open(csv_path, "r") as file:
        reader = csv.DictReader(file)

        for row in reader:
            accuracies = []

            for task_id in range(1, 6):
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


# --------------------------------
# FedAvg-FCL result files
# --------------------------------

fedavg_files = [
    "results/fcl_fedavg_baseline_r3.csv",
    "results/fcl_fedavg_baseline_r3_seed123.csv",
    "results/fcl_fedavg_baseline_r3_seed2026.csv"
]


# --------------------------------
# Replay + LwF result files
# --------------------------------

replay_lwf_files = [
    "results/fcl_replay_lwf_w1.0_r3.csv",
    "results/fcl_replay_lwf_w1.0_r3_seed123.csv",
    "results/fcl_replay_lwf_w1.0_r3_seed2026.csv"
]


# --------------------------------
# Collect FedAvg metrics
# --------------------------------

fedavg_accuracies = []
fedavg_forgetting = []

for file_path in fedavg_files:

    matrix = load_accuracy_matrix(
        file_path
    )

    accuracy, forgetting = (
        calculate_metrics(
            matrix
        )
    )

    fedavg_accuracies.append(
        accuracy
    )

    fedavg_forgetting.append(
        forgetting
    )


# --------------------------------
# Collect Replay + LwF metrics
# --------------------------------

replay_lwf_accuracies = []
replay_lwf_forgetting = []

for file_path in replay_lwf_files:

    matrix = load_accuracy_matrix(
        file_path
    )

    accuracy, forgetting = (
        calculate_metrics(
            matrix
        )
    )

    replay_lwf_accuracies.append(
        accuracy
    )

    replay_lwf_forgetting.append(
        forgetting
    )


# --------------------------------
# Calculate mean and std
# --------------------------------

fedavg_accuracy_mean = statistics.mean(
    fedavg_accuracies
)

fedavg_accuracy_std = statistics.stdev(
    fedavg_accuracies
)

fedavg_forgetting_mean = statistics.mean(
    fedavg_forgetting
)

fedavg_forgetting_std = statistics.stdev(
    fedavg_forgetting
)


replay_lwf_accuracy_mean = statistics.mean(
    replay_lwf_accuracies
)

replay_lwf_accuracy_std = statistics.stdev(
    replay_lwf_accuracies
)

replay_lwf_forgetting_mean = statistics.mean(
    replay_lwf_forgetting
)

replay_lwf_forgetting_std = statistics.stdev(
    replay_lwf_forgetting
)


# --------------------------------
# Print individual seeds
# --------------------------------

print()
print("Individual Seed Results")
print("=" * 60)

print()
print("FedAvg-FCL")

for index, accuracy in enumerate(
    fedavg_accuracies
):
    print(
        f"Run {index + 1} | "
        f"Accuracy: {accuracy:.2f}% | "
        f"Forgetting: "
        f"{fedavg_forgetting[index]:.2f} pp"
    )


print()
print("Replay + LwF FCL")

for index, accuracy in enumerate(
    replay_lwf_accuracies
):
    print(
        f"Run {index + 1} | "
        f"Accuracy: {accuracy:.2f}% | "
        f"Forgetting: "
        f"{replay_lwf_forgetting[index]:.2f} pp"
    )


# --------------------------------
# Final statistical comparison
# --------------------------------

print()
print("Three-Seed Statistical Comparison")
print("=" * 60)

print(
    f"FedAvg-FCL"
    f" | Accuracy: "
    f"{fedavg_accuracy_mean:.2f} "
    f"± {fedavg_accuracy_std:.2f}%"
    f" | Forgetting: "
    f"{fedavg_forgetting_mean:.2f} "
    f"± {fedavg_forgetting_std:.2f} pp"
)

print(
    f"Replay+LwF FCL"
    f" | Accuracy: "
    f"{replay_lwf_accuracy_mean:.2f} "
    f"± {replay_lwf_accuracy_std:.2f}%"
    f" | Forgetting: "
    f"{replay_lwf_forgetting_mean:.2f} "
    f"± {replay_lwf_forgetting_std:.2f} pp"
)