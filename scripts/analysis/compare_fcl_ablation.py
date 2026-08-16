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

        final_accuracy = matrix[-1][task_id]

        forgetting = max(
            0.0,
            best_previous_accuracy - final_accuracy
        )

        forgetting_scores.append(forgetting)

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
            calculate_metrics(matrix)
        )

        accuracies.append(accuracy)
        forgetting_values.append(forgetting)

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


# --------------------------------
# Experiment files
# --------------------------------

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
    ]
}


# --------------------------------
# Calculate results
# --------------------------------

results = {}

for method_name, file_paths in methods.items():

    results[method_name] = evaluate_method(
        file_paths
    )


# --------------------------------
# Individual seed results
# --------------------------------

print()
print("Individual Seed Results")
print("=" * 75)

seeds = [42, 123, 2026]

for method_name, result in results.items():

    print()
    print(method_name)

    for index, seed in enumerate(seeds):

        print(
            f"Seed {seed} | "
            f"Final Avg Accuracy: "
            f"{result['accuracies'][index]:.2f}% | "
            f"Average Forgetting: "
            f"{result['forgetting'][index]:.2f} pp"
        )


# --------------------------------
# Ablation table
# --------------------------------

print()
print("FCL Ablation Study — Mean ± Standard Deviation")
print("=" * 75)

print(
    f"{'Method':<16}"
    f"{'Final Avg Accuracy':<28}"
    f"{'Average Forgetting'}"
)

print("-" * 75)

for method_name, result in results.items():

    accuracy_text = (
        f"{result['accuracy_mean']:.2f} "
        f"± {result['accuracy_std']:.2f}%"
    )

    forgetting_text = (
        f"{result['forgetting_mean']:.2f} "
        f"± {result['forgetting_std']:.2f} pp"
    )

    print(
        f"{method_name:<16}"
        f"{accuracy_text:<28}"
        f"{forgetting_text}"
    )


# --------------------------------
# Improvement over FedAvg
# --------------------------------

baseline = results["FedAvg-FCL"]
combined = results["Replay+LwF"]

accuracy_improvement = (
    combined["accuracy_mean"]
    - baseline["accuracy_mean"]
)

forgetting_reduction = (
    baseline["forgetting_mean"]
    - combined["forgetting_mean"]
)


print()
print("Replay+LwF Improvement over FedAvg-FCL")
print("=" * 75)

print(
    f"Final Average Accuracy Improvement: "
    f"{accuracy_improvement:.2f} percentage points"
)

print(
    f"Average Forgetting Reduction: "
    f"{forgetting_reduction:.2f} percentage points"
)