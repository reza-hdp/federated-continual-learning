import csv
import matplotlib.pyplot as plt


def plot_training_results(csv_path, output_path):
    epochs = []
    train_accuracy = []
    test_accuracy = []

    with open(csv_path, "r") as file:
        reader = csv.DictReader(file)

        for row in reader:
            epochs.append(int(row["epoch"]))
            train_accuracy.append(float(row["train_accuracy"]))
            test_accuracy.append(float(row["test_accuracy"]))

    plt.figure()

    plt.plot(epochs, train_accuracy, marker="o", label="Train Accuracy")
    plt.plot(epochs, test_accuracy, marker="o", label="Test Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Centralized CIFAR-10 Training")
    plt.legend()
    plt.grid(True)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_federated_results(csv_path, output_path):
    rounds = []
    test_accuracy = []

    with open(csv_path, "r") as file:
        reader = csv.DictReader(file)

        for row in reader:
            rounds.append(int(row["round"]))
            test_accuracy.append(float(row["test_accuracy"]))

    plt.figure()

    plt.plot(rounds, test_accuracy, marker="o")

    plt.xlabel("Federated Round")
    plt.ylabel("Test Accuracy (%)")
    plt.title("FedAvg on CIFAR-10")
    plt.grid(True)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_noniid_results(csv_path, output_path):
    rounds = []
    test_accuracy = []

    with open(csv_path, "r") as file:
        reader = csv.DictReader(file)

        for row in reader:
            rounds.append(int(row["round"]))
            test_accuracy.append(float(row["test_accuracy"]))

    plt.figure()

    plt.plot(rounds, test_accuracy, marker="o")

    plt.xlabel("Federated Round")
    plt.ylabel("Test Accuracy (%)")
    plt.title("FedAvg on Non-IID CIFAR-10 (Alpha = 0.5)")
    plt.grid(True)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_iid_vs_noniid(iid_csv, noniid_csv, output_path):
    iid_rounds = []
    iid_accuracy = []

    noniid_rounds = []
    noniid_accuracy = []

    with open(iid_csv, "r") as file:
        reader = csv.DictReader(file)

        for row in reader:
            iid_rounds.append(int(row["round"]))
            iid_accuracy.append(float(row["test_accuracy"]))

    with open(noniid_csv, "r") as file:
        reader = csv.DictReader(file)

        for row in reader:
            noniid_rounds.append(int(row["round"]))
            noniid_accuracy.append(float(row["test_accuracy"]))

    plt.figure()

    plt.plot(
        iid_rounds,
        iid_accuracy,
        marker="o",
        label="IID FedAvg"
    )

    plt.plot(
        noniid_rounds,
        noniid_accuracy,
        marker="o",
        label="Non-IID FedAvg (α = 0.5)"
    )

    plt.xlabel("Federated Round")
    plt.ylabel("Test Accuracy (%)")
    plt.title("FedAvg: IID vs Non-IID CIFAR-10")
    plt.legend()
    plt.grid(True)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_continual_comparison(
    replay_matrix,
    ewc_matrix,
    output_path
):
    methods = ["Replay", "EWC"]

    replay_final = (
        sum(replay_matrix[-1])
        / len(replay_matrix[-1])
    )

    ewc_final = (
        sum(ewc_matrix[-1])
        / len(ewc_matrix[-1])
    )

    final_accuracies = [
        replay_final,
        ewc_final
    ]

    plt.figure()

    plt.bar(
        methods,
        final_accuracies
    )

    plt.ylabel("Final Average Accuracy (%)")
    plt.title("Continual Learning on CIFAR-10")

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()