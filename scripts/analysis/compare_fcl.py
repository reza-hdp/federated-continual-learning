import csv


def load_accuracy_matrix(csv_path):
    matrix = []

    with open(csv_path, "r") as file:
        reader = csv.DictReader(file)

        for row in reader:
            accuracies = []

            for task_id in range(1, 6):
                value = row[f"task_{task_id}"]

                if value != "":
                    accuracies.append(float(value))

            matrix.append(accuracies)

    return matrix


def calculate_metrics(matrix):
    final_accuracies = matrix[-1]

    final_average_accuracy = (
        sum(final_accuracies)
        / len(final_accuracies)
    )

    forgetting_scores = []

    for task_id in range(len(matrix) - 1):
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


fedavg_matrix = load_accuracy_matrix(
    "../../results/fcl_fedavg_baseline_r3.csv"
)

replay_lwf_matrix = load_accuracy_matrix(
    "../../results/fcl_replay_lwf_w1.0_r3.csv"
)


fedavg_accuracy, fedavg_forgetting = (
    calculate_metrics(
        fedavg_matrix
    )
)

replay_lwf_accuracy, replay_lwf_forgetting = (
    calculate_metrics(
        replay_lwf_matrix
    )
)


print()
print("FCL Method Comparison")
print("---------------------------------------------")

print(
    f"FedAvg-FCL     | "
    f"Final Avg Accuracy: "
    f"{fedavg_accuracy:.2f}% | "
    f"Average Forgetting: "
    f"{fedavg_forgetting:.2f} pp"
)

print(
    f"Replay+LwF FCL | "
    f"Final Avg Accuracy: "
    f"{replay_lwf_accuracy:.2f}% | "
    f"Average Forgetting: "
    f"{replay_lwf_forgetting:.2f} pp"
)