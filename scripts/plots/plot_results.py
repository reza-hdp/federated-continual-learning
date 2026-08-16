from src.utils.plotting import (
    plot_training_results,
    plot_federated_results,
    plot_noniid_results,
    plot_iid_vs_noniid
)


plot_training_results(
    "../../results/centralized_cifar10.csv",
    "figures/centralized_cifar10_accuracy.png"
)

plot_federated_results(
    "../../results/fedavg_cifar10.csv",
    "figures/fedavg_cifar10_accuracy.png"
)

plot_noniid_results(
    "../../results/fedavg_noniid_alpha_0.5.csv",
    "figures/fedavg_noniid_alpha_0.5_accuracy.png"
)

plot_iid_vs_noniid(
    "results/fedavg_cifar10.csv",
    "results/fedavg_noniid_alpha_0.5.csv",
    "../../figures/fedavg_iid_vs_noniid.png"
)

print("All figures generated successfully.")