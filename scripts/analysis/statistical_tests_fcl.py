import csv
import itertools
import math
import statistics


NUM_TASKS = 5

SEEDS = [
    42,
    123,
    2026,
    777,
    1001
]


# ============================================================
# Files
# ============================================================

FIXED_FILES = [
    "results/fcl_replay_lwf_w1.0_r3.csv",
    "results/fcl_replay_lwf_w1.0_r3_seed123.csv",
    "results/fcl_replay_lwf_w1.0_r3_seed2026.csv",
    "results/fcl_replay_lwf_w1.0_r3_seed777.csv",
    "results/fcl_replay_lwf_w1.0_r3_seed1001.csv"
]

GRADIENT_BALANCED_FILES = [
    "results/fcl_gradient_balanced_w0.5_1.5_r3_seed42.csv",
    "results/fcl_gradient_balanced_w0.5_1.5_r3_seed123.csv",
    "results/fcl_gradient_balanced_w0.5_1.5_r3_seed2026.csv",
    "results/fcl_gradient_balanced_w0.5_1.5_r3_seed777.csv",
    "results/fcl_gradient_balanced_w0.5_1.5_r3_seed1001.csv"
]


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
# Load metrics
# ============================================================

def load_method_metrics(
    file_paths
):
    accuracies = []
    forgetting_values = []

    for file_path in file_paths:

        matrix = (
            load_accuracy_matrix(
                file_path
            )
        )

        accuracy, forgetting = (
            calculate_metrics(
                matrix
            )
        )

        accuracies.append(
            accuracy
        )

        forgetting_values.append(
            forgetting
        )

    return (
        accuracies,
        forgetting_values
    )


(
    fixed_accuracy,
    fixed_forgetting
) = load_method_metrics(
    FIXED_FILES
)


(
    adaptive_accuracy,
    adaptive_forgetting
) = load_method_metrics(
    GRADIENT_BALANCED_FILES
)


# ============================================================
# Paired differences
# ============================================================

accuracy_differences = [
    adaptive - fixed
    for adaptive, fixed in zip(
        adaptive_accuracy,
        fixed_accuracy
    )
]


forgetting_reductions = [
    fixed - adaptive
    for fixed, adaptive in zip(
        fixed_forgetting,
        adaptive_forgetting
    )
]


# ============================================================
# 95% confidence interval
# ============================================================

def mean_confidence_interval_95(
    values
):
    n = len(values)

    mean_value = statistics.mean(
        values
    )

    if n < 2:
        return (
            mean_value,
            mean_value,
            mean_value
        )

    sample_std = statistics.stdev(
        values
    )

    standard_error = (
        sample_std
        / math.sqrt(n)
    )

    t_critical_values = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228
    }

    degrees_of_freedom = (
        n - 1
    )

    t_critical = (
        t_critical_values[
            degrees_of_freedom
        ]
    )

    margin = (
        t_critical
        * standard_error
    )

    lower_bound = (
        mean_value - margin
    )

    upper_bound = (
        mean_value + margin
    )

    return (
        mean_value,
        lower_bound,
        upper_bound
    )


# ============================================================
# Cohen's dz
# ============================================================

def cohens_dz(
    differences
):
    if len(differences) < 2:
        return 0.0

    mean_difference = statistics.mean(
        differences
    )

    standard_deviation = statistics.stdev(
        differences
    )

    if standard_deviation == 0.0:

        if mean_difference > 0.0:
            return float("inf")

        if mean_difference < 0.0:
            return float("-inf")

        return 0.0

    return (
        mean_difference
        / standard_deviation
    )


# ============================================================
# Exact paired sign-flip test
# ============================================================

def exact_sign_flip_test(
    differences
):
    observed_statistic = abs(
        statistics.mean(
            differences
        )
    )

    absolute_values = [
        abs(value)
        for value in differences
    ]

    permutation_statistics = []

    for signs in itertools.product(
        [-1.0, 1.0],
        repeat=len(differences)
    ):

        permuted_values = [
            sign * value
            for sign, value
            in zip(
                signs,
                absolute_values
            )
        ]

        statistic = abs(
            statistics.mean(
                permuted_values
            )
        )

        permutation_statistics.append(
            statistic
        )

    extreme_count = sum(
        1
        for statistic
        in permutation_statistics
        if (
            statistic
            >= observed_statistic
            - 1e-12
        )
    )

    p_value = (
        extreme_count
        / len(
            permutation_statistics
        )
    )

    return (
        observed_statistic,
        p_value
    )


# ============================================================
# Statistics
# ============================================================

(
    accuracy_mean_difference,
    accuracy_ci_lower,
    accuracy_ci_upper
) = mean_confidence_interval_95(
    accuracy_differences
)


(
    forgetting_mean_reduction,
    forgetting_ci_lower,
    forgetting_ci_upper
) = mean_confidence_interval_95(
    forgetting_reductions
)


accuracy_effect_size = cohens_dz(
    accuracy_differences
)

forgetting_effect_size = cohens_dz(
    forgetting_reductions
)


(
    accuracy_test_statistic,
    accuracy_p_value
) = exact_sign_flip_test(
    accuracy_differences
)


(
    forgetting_test_statistic,
    forgetting_p_value
) = exact_sign_flip_test(
    forgetting_reductions
)


# ============================================================
# Descriptive means
# ============================================================

fixed_accuracy_mean = statistics.mean(
    fixed_accuracy
)

fixed_accuracy_std = statistics.stdev(
    fixed_accuracy
)

adaptive_accuracy_mean = statistics.mean(
    adaptive_accuracy
)

adaptive_accuracy_std = statistics.stdev(
    adaptive_accuracy
)

fixed_forgetting_mean = statistics.mean(
    fixed_forgetting
)

fixed_forgetting_std = statistics.stdev(
    fixed_forgetting
)

adaptive_forgetting_mean = statistics.mean(
    adaptive_forgetting
)

adaptive_forgetting_std = statistics.stdev(
    adaptive_forgetting
)


relative_forgetting_reduction = (
    (
        fixed_forgetting_mean
        - adaptive_forgetting_mean
    )
    / fixed_forgetting_mean
    * 100.0
)


# ============================================================
# Seed-level results
# ============================================================

print()
print(
    "Five-Seed Paired Results"
)

print("=" * 90)

for index, seed in enumerate(
    SEEDS
):

    print()
    print(
        f"Seed {seed}"
    )

    print(
        f"Fixed Replay+LwF Accuracy: "
        f"{fixed_accuracy[index]:.2f}%"
    )

    print(
        f"Gradient-Balanced Accuracy: "
        f"{adaptive_accuracy[index]:.2f}%"
    )

    print(
        f"Accuracy Difference: "
        f"{accuracy_differences[index]:+.2f} pp"
    )

    print(
        f"Fixed Replay+LwF Forgetting: "
        f"{fixed_forgetting[index]:.2f} pp"
    )

    print(
        f"Gradient-Balanced Forgetting: "
        f"{adaptive_forgetting[index]:.2f} pp"
    )

    print(
        f"Forgetting Reduction: "
        f"{forgetting_reductions[index]:+.2f} pp"
    )


# ============================================================
# Descriptive summary
# ============================================================

print()
print(
    "Five-Seed Descriptive Summary"
)

print("=" * 90)

print(
    f"Fixed Replay+LwF Accuracy: "
    f"{fixed_accuracy_mean:.2f} "
    f"± {fixed_accuracy_std:.2f}%"
)

print(
    f"Gradient-Balanced Accuracy: "
    f"{adaptive_accuracy_mean:.2f} "
    f"± {adaptive_accuracy_std:.2f}%"
)

print(
    f"Fixed Replay+LwF Forgetting: "
    f"{fixed_forgetting_mean:.2f} "
    f"± {fixed_forgetting_std:.2f} pp"
)

print(
    f"Gradient-Balanced Forgetting: "
    f"{adaptive_forgetting_mean:.2f} "
    f"± {adaptive_forgetting_std:.2f} pp"
)

print(
    f"Relative Forgetting Reduction: "
    f"{relative_forgetting_reduction:.2f}%"
)


# ============================================================
# Accuracy analysis
# ============================================================

print()
print(
    "Paired Accuracy Analysis"
)

print("=" * 90)

print(
    f"Mean Difference "
    f"(Gradient-Balanced - Fixed): "
    f"{accuracy_mean_difference:+.2f} pp"
)

print(
    f"95% CI: "
    f"[{accuracy_ci_lower:+.2f}, "
    f"{accuracy_ci_upper:+.2f}] pp"
)

print(
    f"Cohen's dz: "
    f"{accuracy_effect_size:+.3f}"
)

print(
    f"Exact Sign-Flip Test Statistic: "
    f"{accuracy_test_statistic:.4f}"
)

print(
    f"Exact Two-Sided p-value: "
    f"{accuracy_p_value:.4f}"
)


# ============================================================
# Forgetting analysis
# ============================================================

print()
print(
    "Paired Forgetting Analysis"
)

print("=" * 90)

print(
    f"Mean Forgetting Reduction "
    f"(Fixed - Gradient-Balanced): "
    f"{forgetting_mean_reduction:+.2f} pp"
)

print(
    f"95% CI: "
    f"[{forgetting_ci_lower:+.2f}, "
    f"{forgetting_ci_upper:+.2f}] pp"
)

print(
    f"Cohen's dz: "
    f"{forgetting_effect_size:+.3f}"
)

print(
    f"Exact Sign-Flip Test Statistic: "
    f"{forgetting_test_statistic:.4f}"
)

print(
    f"Exact Two-Sided p-value: "
    f"{forgetting_p_value:.4f}"
)


# ============================================================
# Save results
# ============================================================

OUTPUT_FILE = (
    "results/"
    "fcl_statistical_tests_5seeds.csv"
)


with open(
    OUTPUT_FILE,
    "w",
    newline=""
) as output_file:

    fieldnames = [
        "metric",
        "mean_difference",
        "ci_95_lower",
        "ci_95_upper",
        "cohens_dz",
        "exact_p_value"
    ]

    writer = csv.DictWriter(
        output_file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerow({
        "metric": (
            "final_average_accuracy"
        ),
        "mean_difference": (
            accuracy_mean_difference
        ),
        "ci_95_lower": (
            accuracy_ci_lower
        ),
        "ci_95_upper": (
            accuracy_ci_upper
        ),
        "cohens_dz": (
            accuracy_effect_size
        ),
        "exact_p_value": (
            accuracy_p_value
        )
    })

    writer.writerow({
        "metric": (
            "average_forgetting_reduction"
        ),
        "mean_difference": (
            forgetting_mean_reduction
        ),
        "ci_95_lower": (
            forgetting_ci_lower
        ),
        "ci_95_upper": (
            forgetting_ci_upper
        ),
        "cohens_dz": (
            forgetting_effect_size
        ),
        "exact_p_value": (
            forgetting_p_value
        )
    })


print()
print(
    f"Statistical results saved to "
    f"{OUTPUT_FILE}"
)


# ============================================================
# Interpretation
# ============================================================

print()
print(
    "Statistical Interpretation"
)

print("=" * 90)

if forgetting_p_value < 0.05:

    print(
        "The five-seed paired exact test "
        "supports a statistically significant "
        "reduction in forgetting."
    )

else:

    print(
        "The forgetting reduction does not "
        "reach the conventional 0.05 "
        "significance threshold."
    )

if accuracy_mean_difference < 0.0:

    print(
        "Gradient-Balanced continues to trade "
        "some final average accuracy for "
        "improved retention."
    )

print(
    "The paired effect size, confidence interval, "
    "and consistency across seeds should be "
    "reported together with the p-value."
)