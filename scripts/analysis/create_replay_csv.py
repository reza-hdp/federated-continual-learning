import csv


accuracy_matrix = [
    [91.50],
    [19.90, 77.45],
    [17.50, 0.10, 76.15],
    [26.90, 0.05, 0.00, 85.00],
    [0.80, 2.25, 7.95, 29.80, 85.65]
]


with open(
    "results/continual_replay_accuracy.csv",
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "after_task",
        "task_1",
        "task_2",
        "task_3",
        "task_4",
        "task_5"
    ])

    for task_id, accuracies in enumerate(
        accuracy_matrix,
        start=1
    ):
        row = [task_id] + accuracies

        while len(row) < 6:
            row.append("")

        writer.writerow(row)


print(
    "Created results/continual_replay_accuracy.csv"
)