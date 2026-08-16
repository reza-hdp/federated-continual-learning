from src.utils.plotting import plot_continual_comparison
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
        sum(final_accuracies) / len(final_accuracies)
    )

    forgetting_scores = []

    for task_id in range(len(matrix) - 1):
        best_previous_accuracy = max(
            row[task_id]
            for row in matrix
            if len(row) > task_id
        )

        final_accuracy = matrix[-1][task_id]

        forgetting_scores.append(
            best_previous_accuracy - final_accuracy
        )

    average_forgetting = (
        sum(forgetting_scores)
        / len(forgetting_scores)
    )

    return final_average_accuracy, average_forgetting


# Load EWC results
ewc_matrix = load_accuracy_matrix(
    "../../results/continual_ewc_accuracy.csv"
)


# Load Replay results
replay_matrix = load_accuracy_matrix(
    "../../results/continual_replay_accuracy.csv"
)


# Load LwF results
lwf_matrix = load_accuracy_matrix(
    "../../results/continual_lwf_accuracy.csv"
)

replay_lwf_matrix = load_accuracy_matrix(
    "../../results/continual_replay_lwf_accuracy.csv"
)

# Display EWC results
print("EWC results:")

for row in ewc_matrix:
    print(row)


print()


# Display Replay results
print("Replay results:")

for row in replay_matrix:
    print(row)


print()


# Display LwF results
print("LwF results:")

for row in lwf_matrix:
    print(row)


# Calculate Replay metrics
replay_accuracy, replay_forgetting = calculate_metrics(
    replay_matrix
)


# Calculate EWC metrics
ewc_accuracy, ewc_forgetting = calculate_metrics(
    ewc_matrix
)


# Calculate LwF metrics
lwf_accuracy, lwf_forgetting = calculate_metrics(
    lwf_matrix
)

replay_lwf_accuracy, replay_lwf_forgetting = calculate_metrics(
    replay_lwf_matrix
)

# Print comparison
print()
print("Method Comparison")
print("-----------------------------------------------")

print(
    f"Replay | Final Average Accuracy: "
    f"{replay_accuracy:.2f}% | "
    f"Average Forgetting: {replay_forgetting:.2f} pp"
)

print(
    f"EWC    | Final Average Accuracy: "
    f"{ewc_accuracy:.2f}% | "
    f"Average Forgetting: {ewc_forgetting:.2f} pp"
)

print(
    f"LwF    | Final Average Accuracy: "
    f"{lwf_accuracy:.2f}% | "
    f"Average Forgetting: {lwf_forgetting:.2f} pp"
)

print(
    f"Replay+LwF | Final Average Accuracy: "
    f"{replay_lwf_accuracy:.2f}% | "
    f"Average Forgetting: {replay_lwf_forgetting:.2f} pp"
)

# Keep the existing Replay vs EWC figure
plot_continual_comparison(
    replay_matrix,
    ewc_matrix,
    "../../figures/continual_replay_vs_ewc.png"
)

print()
print(
    "Comparison figure saved to "
    "figures/continual_replay_vs_ewc.png"
)