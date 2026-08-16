import csv
import os


# ============================================================
# Setup
# ============================================================

os.makedirs(
    "../../results",
    exist_ok=True
)


# ============================================================
# Experiment records
# ============================================================

experiments = [
    {
        "experiment_id": "main_fedavg_seed42",
        "method": "FedAvg-FCL",
        "dataset": "CIFAR-10",
        "num_clients": 5,
        "num_tasks": 5,
        "classes_per_task": 2,
        "alpha": 0.5,
        "seed": 42,
        "rounds_per_task": 3,
        "local_epochs": 1,
        "batch_size": 64,
        "learning_rate": 0.001,
        "memory_size": 0,
        "temperature": "",
        "distillation_weight": "",
        "min_weight": "",
        "max_weight": "",
        "results_file": (
            "results/"
            "fcl_fedavg_baseline_r3.csv"
        ),
        "model_file": (
            "results/"
            "fcl_fedavg_baseline_r3_model.pth"
        )
    },
    {
        "experiment_id": "main_fedavg_seed123",
        "method": "FedAvg-FCL",
        "dataset": "CIFAR-10",
        "num_clients": 5,
        "num_tasks": 5,
        "classes_per_task": 2,
        "alpha": 0.5,
        "seed": 123,
        "rounds_per_task": 3,
        "local_epochs": 1,
        "batch_size": 64,
        "learning_rate": 0.001,
        "memory_size": 0,
        "temperature": "",
        "distillation_weight": "",
        "min_weight": "",
        "max_weight": "",
        "results_file": (
            "results/"
            "fcl_fedavg_baseline_r3_seed123.csv"
        ),
        "model_file": (
            "results/"
            "fcl_fedavg_baseline_r3_seed123_model.pth"
        )
    },
    {
        "experiment_id": "main_fedavg_seed2026",
        "method": "FedAvg-FCL",
        "dataset": "CIFAR-10",
        "num_clients": 5,
        "num_tasks": 5,
        "classes_per_task": 2,
        "alpha": 0.5,
        "seed": 2026,
        "rounds_per_task": 3,
        "local_epochs": 1,
        "batch_size": 64,
        "learning_rate": 0.001,
        "memory_size": 0,
        "temperature": "",
        "distillation_weight": "",
        "min_weight": "",
        "max_weight": "",
        "results_file": (
            "results/"
            "fcl_fedavg_baseline_r3_seed2026.csv"
        ),
        "model_file": (
            "results/"
            "fcl_fedavg_baseline_r3_seed2026_model.pth"
        )
    },

    # ========================================================
    # Replay ablation
    # ========================================================

    {
        "experiment_id": "replay_seed42",
        "method": "Replay",
        "dataset": "CIFAR-10",
        "num_clients": 5,
        "num_tasks": 5,
        "classes_per_task": 2,
        "alpha": 0.5,
        "seed": 42,
        "rounds_per_task": 3,
        "local_epochs": 1,
        "batch_size": 64,
        "learning_rate": 0.001,
        "memory_size": 500,
        "temperature": "",
        "distillation_weight": "",
        "min_weight": "",
        "max_weight": "",
        "results_file": (
            "results/"
            "fcl_replay_r3_seed42.csv"
        ),
        "model_file": (
            "results/"
            "fcl_replay_r3_seed42_model.pth"
        )
    },
    {
        "experiment_id": "replay_seed123",
        "method": "Replay",
        "dataset": "CIFAR-10",
        "num_clients": 5,
        "num_tasks": 5,
        "classes_per_task": 2,
        "alpha": 0.5,
        "seed": 123,
        "rounds_per_task": 3,
        "local_epochs": 1,
        "batch_size": 64,
        "learning_rate": 0.001,
        "memory_size": 500,
        "temperature": "",
        "distillation_weight": "",
        "min_weight": "",
        "max_weight": "",
        "results_file": (
            "results/"
            "fcl_replay_r3_seed123.csv"
        ),
        "model_file": (
            "results/"
            "fcl_replay_r3_seed123_model.pth"
        )
    },
    {
        "experiment_id": "replay_seed2026",
        "method": "Replay",
        "dataset": "CIFAR-10",
        "num_clients": 5,
        "num_tasks": 5,
        "classes_per_task": 2,
        "alpha": 0.5,
        "seed": 2026,
        "rounds_per_task": 3,
        "local_epochs": 1,
        "batch_size": 64,
        "learning_rate": 0.001,
        "memory_size": 500,
        "temperature": "",
        "distillation_weight": "",
        "min_weight": "",
        "max_weight": "",
        "results_file": (
            "results/"
            "fcl_replay_r3_seed2026.csv"
        ),
        "model_file": (
            "results/"
            "fcl_replay_r3_seed2026_model.pth"
        )
    },

    # ========================================================
    # LwF ablation
    # ========================================================

    {
        "experiment_id": "lwf_seed42",
        "method": "LwF",
        "dataset": "CIFAR-10",
        "num_clients": 5,
        "num_tasks": 5,
        "classes_per_task": 2,
        "alpha": 0.5,
        "seed": 42,
        "rounds_per_task": 3,
        "local_epochs": 1,
        "batch_size": 64,
        "learning_rate": 0.001,
        "memory_size": 0,
        "temperature": 2.0,
        "distillation_weight": 1.0,
        "min_weight": "",
        "max_weight": "",
        "results_file": (
            "results/"
            "fcl_lwf_r3_seed42.csv"
        ),
        "model_file": (
            "results/"
            "fcl_lwf_r3_seed42_model.pth"
        )
    },
    {
        "experiment_id": "lwf_seed123",
        "method": "LwF",
        "dataset": "CIFAR-10",
        "num_clients": 5,
        "num_tasks": 5,
        "classes_per_task": 2,
        "alpha": 0.5,
        "seed": 123,
        "rounds_per_task": 3,
        "local_epochs": 1,
        "batch_size": 64,
        "learning_rate": 0.001,
        "memory_size": 0,
        "temperature": 2.0,
        "distillation_weight": 1.0,
        "min_weight": "",
        "max_weight": "",
        "results_file": (
            "results/"
            "fcl_lwf_r3_seed123.csv"
        ),
        "model_file": (
            "results/"
            "fcl_lwf_r3_seed123_model.pth"
        )
    },
    {
        "experiment_id": "lwf_seed2026",
        "method": "LwF",
        "dataset": "CIFAR-10",
        "num_clients": 5,
        "num_tasks": 5,
        "classes_per_task": 2,
        "alpha": 0.5,
        "seed": 2026,
        "rounds_per_task": 3,
        "local_epochs": 1,
        "batch_size": 64,
        "learning_rate": 0.001,
        "memory_size": 0,
        "temperature": 2.0,
        "distillation_weight": 1.0,
        "min_weight": "",
        "max_weight": "",
        "results_file": (
            "results/"
            "fcl_lwf_r3_seed2026.csv"
        ),
        "model_file": (
            "results/"
            "fcl_lwf_r3_seed2026_model.pth"
        )
    },

    # ========================================================
    # Fixed Replay + LwF, five seeds
    # ========================================================

    {
        "experiment_id": "fixed_replay_lwf_seed42",
        "method": "Replay+LwF",
        "dataset": "CIFAR-10",
        "num_clients": 5,
        "num_tasks": 5,
        "classes_per_task": 2,
        "alpha": 0.5,
        "seed": 42,
        "rounds_per_task": 3,
        "local_epochs": 1,
        "batch_size": 64,
        "learning_rate": 0.001,
        "memory_size": 500,
        "temperature": 2.0,
        "distillation_weight": 1.0,
        "min_weight": "",
        "max_weight": "",
        "results_file": (
            "results/"
            "fcl_replay_lwf_w1.0_r3.csv"
        ),
        "model_file": (
            "results/"
            "fcl_replay_lwf_w1.0_r3_model.pth"
        )
    },
    {
        "experiment_id": "fixed_replay_lwf_seed123",
        "method": "Replay+LwF",
        "dataset": "CIFAR-10",
        "num_clients": 5,
        "num_tasks": 5,
        "classes_per_task": 2,
        "alpha": 0.5,
        "seed": 123,
        "rounds_per_task": 3,
        "local_epochs": 1,
        "batch_size": 64,
        "learning_rate": 0.001,
        "memory_size": 500,
        "temperature": 2.0,
        "distillation_weight": 1.0,
        "min_weight": "",
        "max_weight": "",
        "results_file": (
            "results/"
            "fcl_replay_lwf_w1.0_r3_seed123.csv"
        ),
        "model_file": (
            "results/"
            "fcl_replay_lwf_w1.0_r3_seed123_model.pth"
        )
    },
    {
        "experiment_id": "fixed_replay_lwf_seed2026",
        "method": "Replay+LwF",
        "dataset": "CIFAR-10",
        "num_clients": 5,
        "num_tasks": 5,
        "classes_per_task": 2,
        "alpha": 0.5,
        "seed": 2026,
        "rounds_per_task": 3,
        "local_epochs": 1,
        "batch_size": 64,
        "learning_rate": 0.001,
        "memory_size": 500,
        "temperature": 2.0,
        "distillation_weight": 1.0,
        "min_weight": "",
        "max_weight": "",
        "results_file": (
            "results/"
            "fcl_replay_lwf_w1.0_r3_seed2026.csv"
        ),
        "model_file": (
            "results/"
            "fcl_replay_lwf_w1.0_r3_seed2026_model.pth"
        )
    },
    {
        "experiment_id": "fixed_replay_lwf_seed777",
        "method": "Replay+LwF",
        "dataset": "CIFAR-10",
        "num_clients": 5,
        "num_tasks": 5,
        "classes_per_task": 2,
        "alpha": 0.5,
        "seed": 777,
        "rounds_per_task": 3,
        "local_epochs": 1,
        "batch_size": 64,
        "learning_rate": 0.001,
        "memory_size": 500,
        "temperature": 2.0,
        "distillation_weight": 1.0,
        "min_weight": "",
        "max_weight": "",
        "results_file": (
            "results/"
            "fcl_replay_lwf_w1.0_r3_seed777.csv"
        ),
        "model_file": (
            "results/"
            "fcl_replay_lwf_w1.0_r3_seed777_model.pth"
        )
    },
    {
        "experiment_id": "fixed_replay_lwf_seed1001",
        "method": "Replay+LwF",
        "dataset": "CIFAR-10",
        "num_clients": 5,
        "num_tasks": 5,
        "classes_per_task": 2,
        "alpha": 0.5,
        "seed": 1001,
        "rounds_per_task": 3,
        "local_epochs": 1,
        "batch_size": 64,
        "learning_rate": 0.001,
        "memory_size": 500,
        "temperature": 2.0,
        "distillation_weight": 1.0,
        "min_weight": "",
        "max_weight": "",
        "results_file": (
            "results/"
            "fcl_replay_lwf_w1.0_r3_seed1001.csv"
        ),
        "model_file": (
            "results/"
            "fcl_replay_lwf_w1.0_r3_seed1001_model.pth"
        )
    },

    # ========================================================
    # Gradient-Balanced, five seeds
    # ========================================================

    {
        "experiment_id": "gradient_balanced_seed42",
        "method": "Gradient-Balanced Replay+LwF",
        "dataset": "CIFAR-10",
        "num_clients": 5,
        "num_tasks": 5,
        "classes_per_task": 2,
        "alpha": 0.5,
        "seed": 42,
        "rounds_per_task": 3,
        "local_epochs": 1,
        "batch_size": 64,
        "learning_rate": 0.001,
        "memory_size": 500,
        "temperature": 2.0,
        "distillation_weight": "adaptive",
        "min_weight": 0.5,
        "max_weight": 1.5,
        "results_file": (
            "results/"
            "fcl_gradient_balanced_w0.5_1.5_r3_seed42.csv"
        ),
        "model_file": (
            "results/"
            "fcl_gradient_balanced_w0.5_1.5_r3_seed42_model.pth"
        )
    },
    {
        "experiment_id": "gradient_balanced_seed123",
        "method": "Gradient-Balanced Replay+LwF",
        "dataset": "CIFAR-10",
        "num_clients": 5,
        "num_tasks": 5,
        "classes_per_task": 2,
        "alpha": 0.5,
        "seed": 123,
        "rounds_per_task": 3,
        "local_epochs": 1,
        "batch_size": 64,
        "learning_rate": 0.001,
        "memory_size": 500,
        "temperature": 2.0,
        "distillation_weight": "adaptive",
        "min_weight": 0.5,
        "max_weight": 1.5,
        "results_file": (
            "results/"
            "fcl_gradient_balanced_w0.5_1.5_r3_seed123.csv"
        ),
        "model_file": (
            "results/"
            "fcl_gradient_balanced_w0.5_1.5_r3_seed123_model.pth"
        )
    },
    {
        "experiment_id": "gradient_balanced_seed2026",
        "method": "Gradient-Balanced Replay+LwF",
        "dataset": "CIFAR-10",
        "num_clients": 5,
        "num_tasks": 5,
        "classes_per_task": 2,
        "alpha": 0.5,
        "seed": 2026,
        "rounds_per_task": 3,
        "local_epochs": 1,
        "batch_size": 64,
        "learning_rate": 0.001,
        "memory_size": 500,
        "temperature": 2.0,
        "distillation_weight": "adaptive",
        "min_weight": 0.5,
        "max_weight": 1.5,
        "results_file": (
            "results/"
            "fcl_gradient_balanced_w0.5_1.5_r3_seed2026.csv"
        ),
        "model_file": (
            "results/"
            "fcl_gradient_balanced_w0.5_1.5_r3_seed2026_model.pth"
        )
    },
    {
        "experiment_id": "gradient_balanced_seed777",
        "method": "Gradient-Balanced Replay+LwF",
        "dataset": "CIFAR-10",
        "num_clients": 5,
        "num_tasks": 5,
        "classes_per_task": 2,
        "alpha": 0.5,
        "seed": 777,
        "rounds_per_task": 3,
        "local_epochs": 1,
        "batch_size": 64,
        "learning_rate": 0.001,
        "memory_size": 500,
        "temperature": 2.0,
        "distillation_weight": "adaptive",
        "min_weight": 0.5,
        "max_weight": 1.5,
        "results_file": (
            "results/"
            "fcl_gradient_balanced_w0.5_1.5_r3_seed777.csv"
        ),
        "model_file": (
            "results/"
            "fcl_gradient_balanced_w0.5_1.5_r3_seed777_model.pth"
        )
    },
    {
        "experiment_id": "gradient_balanced_seed1001",
        "method": "Gradient-Balanced Replay+LwF",
        "dataset": "CIFAR-10",
        "num_clients": 5,
        "num_tasks": 5,
        "classes_per_task": 2,
        "alpha": 0.5,
        "seed": 1001,
        "rounds_per_task": 3,
        "local_epochs": 1,
        "batch_size": 64,
        "learning_rate": 0.001,
        "memory_size": 500,
        "temperature": 2.0,
        "distillation_weight": "adaptive",
        "min_weight": 0.5,
        "max_weight": 1.5,
        "results_file": (
            "results/"
            "fcl_gradient_balanced_w0.5_1.5_r3_seed1001.csv"
        ),
        "model_file": (
            "results/"
            "fcl_gradient_balanced_w0.5_1.5_r3_seed1001_model.pth"
        )
    },

    # ========================================================
    # Heterogeneity robustness
    # ========================================================

    {
        "experiment_id": "robust_fixed_alpha0.1_seed42",
        "method": "Replay+LwF",
        "dataset": "CIFAR-10",
        "num_clients": 5,
        "num_tasks": 5,
        "classes_per_task": 2,
        "alpha": 0.1,
        "seed": 42,
        "rounds_per_task": 3,
        "local_epochs": 1,
        "batch_size": 64,
        "learning_rate": 0.001,
        "memory_size": 500,
        "temperature": 2.0,
        "distillation_weight": 1.0,
        "min_weight": "",
        "max_weight": "",
        "results_file": (
            "results/"
            "fcl_replay_lwf_alpha0.1_w1.0_r3_seed42.csv"
        ),
        "model_file": (
            "results/"
            "fcl_replay_lwf_alpha0.1_w1.0_r3_seed42_model.pth"
        )
    },
    {
        "experiment_id": "robust_gb_alpha0.1_seed42",
        "method": "Gradient-Balanced Replay+LwF",
        "dataset": "CIFAR-10",
        "num_clients": 5,
        "num_tasks": 5,
        "classes_per_task": 2,
        "alpha": 0.1,
        "seed": 42,
        "rounds_per_task": 3,
        "local_epochs": 1,
        "batch_size": 64,
        "learning_rate": 0.001,
        "memory_size": 500,
        "temperature": 2.0,
        "distillation_weight": "adaptive",
        "min_weight": 0.5,
        "max_weight": 1.5,
        "results_file": (
            "results/"
            "fcl_gradient_balanced_"
            "alpha0.1_w0.5_1.5_r3_seed42.csv"
        ),
        "model_file": (
            "results/"
            "fcl_gradient_balanced_"
            "alpha0.1_w0.5_1.5_r3_seed42_model.pth"
        )
    },
    {
        "experiment_id": "robust_fixed_alpha1.0_seed42",
        "method": "Replay+LwF",
        "dataset": "CIFAR-10",
        "num_clients": 5,
        "num_tasks": 5,
        "classes_per_task": 2,
        "alpha": 1.0,
        "seed": 42,
        "rounds_per_task": 3,
        "local_epochs": 1,
        "batch_size": 64,
        "learning_rate": 0.001,
        "memory_size": 500,
        "temperature": 2.0,
        "distillation_weight": 1.0,
        "min_weight": "",
        "max_weight": "",
        "results_file": (
            "results/"
            "fcl_replay_lwf_alpha1.0_w1.0_r3_seed42.csv"
        ),
        "model_file": (
            "results/"
            "fcl_replay_lwf_alpha1.0_w1.0_r3_seed42_model.pth"
        )
    },
    {
        "experiment_id": "robust_gb_alpha1.0_seed42",
        "method": "Gradient-Balanced Replay+LwF",
        "dataset": "CIFAR-10",
        "num_clients": 5,
        "num_tasks": 5,
        "classes_per_task": 2,
        "alpha": 1.0,
        "seed": 42,
        "rounds_per_task": 3,
        "local_epochs": 1,
        "batch_size": 64,
        "learning_rate": 0.001,
        "memory_size": 500,
        "temperature": 2.0,
        "distillation_weight": "adaptive",
        "min_weight": 0.5,
        "max_weight": 1.5,
        "results_file": (
            "results/"
            "fcl_gradient_balanced_"
            "alpha1.0_w0.5_1.5_r3_seed42.csv"
        ),
        "model_file": (
            "results/"
            "fcl_gradient_balanced_"
            "alpha1.0_w0.5_1.5_r3_seed42_model.pth"
        )
    }
]


# ============================================================
# Save manifest
# ============================================================

OUTPUT_FILE = (
    "results/"
    "experiment_manifest.csv"
)


with open(
    OUTPUT_FILE,
    "w",
    newline=""
) as output_file:

    fieldnames = list(
        experiments[0].keys()
    )

    writer = csv.DictWriter(
        output_file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        experiments
    )


# ============================================================
# Verify referenced files
# ============================================================

missing_results = []
missing_models = []


for experiment in experiments:

    results_file = (
        experiment[
            "results_file"
        ]
    )

    model_file = (
        experiment[
            "model_file"
        ]
    )

    if not os.path.exists(
        results_file
    ):
        missing_results.append(
            results_file
        )

    if not os.path.exists(
        model_file
    ):
        missing_models.append(
            model_file
        )


# ============================================================
# Print summary
# ============================================================

print()
print("=" * 85)

print(
    "FCL Experiment Manifest"
)

print("=" * 85)

print(
    f"Experiments recorded: "
    f"{len(experiments)}"
)

print(
    f"Missing result files: "
    f"{len(missing_results)}"
)

print(
    f"Missing model files: "
    f"{len(missing_models)}"
)


if missing_results:

    print()
    print(
        "Missing result files:"
    )

    for file_path in (
        missing_results
    ):

        print(
            file_path
        )


if missing_models:

    print()
    print(
        "Missing model files:"
    )

    for file_path in (
        missing_models
    ):

        print(
            file_path
        )


print()
print(
    f"Manifest saved to "
    f"{OUTPUT_FILE}"
)


if (
    not missing_results
    and not missing_models
):

    print()

    print(
        "RESULT: All recorded experimental "
        "artifacts are present."
    )

else:

    print()

    print(
        "RESULT: Some experimental artifacts "
        "are missing. Review the paths above."
    )