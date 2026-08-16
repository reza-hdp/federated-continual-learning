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
# Table 1
# Main ablation study
# ============================================================

ablation_results = [
    {
        "method": "FedAvg-FCL",
        "accuracy_mean": 16.42,
        "accuracy_std": 1.04,
        "forgetting_mean": 68.75,
        "forgetting_std": 2.84
    },
    {
        "method": "Replay",
        "accuracy_mean": 19.33,
        "accuracy_std": 4.72,
        "forgetting_mean": 65.55,
        "forgetting_std": 2.79
    },
    {
        "method": "LwF",
        "accuracy_mean": 19.89,
        "accuracy_std": 1.84,
        "forgetting_mean": 43.66,
        "forgetting_std": 5.93
    },
    {
        "method": "Replay+LwF",
        "accuracy_mean": 46.33,
        "accuracy_std": 4.08,
        "forgetting_mean": 5.23,
        "forgetting_std": 0.56
    }
]


# ============================================================
# Table 2
# Five-seed main comparison
# ============================================================

five_seed_results = [
    {
        "method": "Fixed Replay+LwF",
        "accuracy_mean": 46.94,
        "accuracy_std": 3.40,
        "forgetting_mean": 4.76,
        "forgetting_std": 0.76
    },
    {
        "method": "Gradient-Balanced",
        "accuracy_mean": 44.76,
        "accuracy_std": 3.50,
        "forgetting_mean": 1.77,
        "forgetting_std": 0.34
    }
]


# ============================================================
# Table 3
# Heterogeneity robustness
# ============================================================

heterogeneity_results = [
    {
        "alpha": 0.1,
        "fixed_accuracy": 32.19,
        "gb_accuracy": 28.28,
        "accuracy_difference": -3.91,
        "fixed_forgetting": 7.56,
        "gb_forgetting": 3.99,
        "forgetting_reduction": 3.58,
        "relative_reduction": 47.27
    },
    {
        "alpha": 0.5,
        "fixed_accuracy": 46.57,
        "gb_accuracy": 45.77,
        "accuracy_difference": -0.80,
        "fixed_forgetting": 4.80,
        "gb_forgetting": 2.01,
        "forgetting_reduction": 2.79,
        "relative_reduction": 58.07
    },
    {
        "alpha": 1.0,
        "fixed_accuracy": 48.29,
        "gb_accuracy": 44.46,
        "accuracy_difference": -3.83,
        "fixed_forgetting": 5.15,
        "gb_forgetting": 1.47,
        "forgetting_reduction": 3.68,
        "relative_reduction": 71.36
    }
]


# ============================================================
# Table 4
# Adaptive-weight analysis
# ============================================================

weight_results = [
    {
        "group": "Overall",
        "mean": 1.2963,
        "std": 0.0951,
        "minimum": 1.0619,
        "maximum": 1.4903
    },
    {
        "group": "Task 2",
        "mean": 1.3892,
        "std": 0.0701,
        "minimum": 1.2384,
        "maximum": 1.4903
    },
    {
        "group": "Task 3",
        "mean": 1.2129,
        "std": 0.0807,
        "minimum": 1.0619,
        "maximum": 1.4456
    },
    {
        "group": "Task 4",
        "mean": 1.2528,
        "std": 0.0571,
        "minimum": 1.1240,
        "maximum": 1.3461
    },
    {
        "group": "Task 5",
        "mean": 1.3302,
        "std": 0.0548,
        "minimum": 1.2166,
        "maximum": 1.4155
    },
    {
        "group": "Round 1",
        "mean": 1.3594,
        "std": 0.0729,
        "minimum": "",
        "maximum": ""
    },
    {
        "group": "Round 2",
        "mean": 1.2735,
        "std": 0.0863,
        "minimum": "",
        "maximum": ""
    },
    {
        "group": "Round 3",
        "mean": 1.2560,
        "std": 0.0916,
        "minimum": "",
        "maximum": ""
    }
]


# ============================================================
# Table 5
# Statistical comparison
# ============================================================

statistical_results = [
    {
        "metric": "Final Average Accuracy",
        "fixed_mean": 46.94,
        "gradient_balanced_mean": 44.76,
        "paired_difference": -2.18,
        "ci_lower": -3.78,
        "ci_upper": -0.58,
        "cohens_dz": -1.692,
        "exact_p": 0.0625
    },
    {
        "metric": "Average Forgetting",
        "fixed_mean": 4.76,
        "gradient_balanced_mean": 1.77,
        "paired_difference": -3.00,
        "ci_lower": -4.05,
        "ci_upper": -1.94,
        "cohens_dz": -3.538,
        "exact_p": 0.0625
    }
]


# ============================================================
# Save helper
# ============================================================

def save_table(
    file_path,
    rows
):
    if not rows:
        return

    with open(
        file_path,
        "w",
        newline=""
    ) as output_file:

        writer = csv.DictWriter(
            output_file,
            fieldnames=list(
                rows[0].keys()
            )
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# Save all tables
# ============================================================

save_table(
    "../../results/paper_table_ablation.csv",
    ablation_results
)

save_table(
    "../../results/paper_table_main_5seeds.csv",
    five_seed_results
)

save_table(
    "../../results/paper_table_heterogeneity.csv",
    heterogeneity_results
)

save_table(
    "../../results/paper_table_adaptive_weights.csv",
    weight_results
)

save_table(
    "../../results/paper_table_statistics.csv",
    statistical_results
)


# ============================================================
# Print paper-ready tables
# ============================================================

print()
print("=" * 90)

print(
    "TABLE 1 — FCL Ablation Study"
)

print("=" * 90)

print(
    f"{'Method':<20}"
    f"{'Final Avg Accuracy':<25}"
    f"{'Average Forgetting'}"
)

print("-" * 90)

for row in ablation_results:

    accuracy = (
        f"{row['accuracy_mean']:.2f} "
        f"± {row['accuracy_std']:.2f}%"
    )

    forgetting = (
        f"{row['forgetting_mean']:.2f} "
        f"± {row['forgetting_std']:.2f} pp"
    )

    print(
        f"{row['method']:<20}"
        f"{accuracy:<25}"
        f"{forgetting}"
    )


print()
print("=" * 90)

print(
    "TABLE 2 — Five-Seed Main Comparison"
)

print("=" * 90)

print(
    f"{'Method':<25}"
    f"{'Final Avg Accuracy':<25}"
    f"{'Average Forgetting'}"
)

print("-" * 90)

for row in five_seed_results:

    accuracy = (
        f"{row['accuracy_mean']:.2f} "
        f"± {row['accuracy_std']:.2f}%"
    )

    forgetting = (
        f"{row['forgetting_mean']:.2f} "
        f"± {row['forgetting_std']:.2f} pp"
    )

    print(
        f"{row['method']:<25}"
        f"{accuracy:<25}"
        f"{forgetting}"
    )


print()
print("=" * 110)

print(
    "TABLE 3 — Heterogeneity Robustness (Seed 42)"
)

print("=" * 110)

print(
    f"{'Alpha':<10}"
    f"{'Fixed Acc':<13}"
    f"{'GB Acc':<13}"
    f"{'Acc Diff':<13}"
    f"{'Fixed F':<13}"
    f"{'GB F':<13}"
    f"{'F Reduction':<15}"
    f"{'Relative'}"
)

print("-" * 110)

for row in heterogeneity_results:

    print(
        f"{row['alpha']:<10.1f}"
        f"{row['fixed_accuracy']:<13.2f}"
        f"{row['gb_accuracy']:<13.2f}"
        f"{row['accuracy_difference']:<+13.2f}"
        f"{row['fixed_forgetting']:<13.2f}"
        f"{row['gb_forgetting']:<13.2f}"
        f"{row['forgetting_reduction']:<+15.2f}"
        f"{row['relative_reduction']:.2f}%"
    )


print()
print("=" * 90)

print(
    "TABLE 4 — Adaptive Retention Weights"
)

print("=" * 90)

print(
    f"{'Group':<20}"
    f"{'Mean':<15}"
    f"{'Std':<15}"
    f"{'Minimum':<15}"
    f"{'Maximum'}"
)

print("-" * 90)

for row in weight_results:

    minimum = (
        f"{row['minimum']:.4f}"
        if row["minimum"] != ""
        else "-"
    )

    maximum = (
        f"{row['maximum']:.4f}"
        if row["maximum"] != ""
        else "-"
    )

    print(
        f"{row['group']:<20}"
        f"{row['mean']:<15.4f}"
        f"{row['std']:<15.4f}"
        f"{minimum:<15}"
        f"{maximum}"
    )


print()
print("=" * 100)

print(
    "TABLE 5 — Paired Five-Seed Statistical Analysis"
)

print("=" * 100)

for row in statistical_results:

    print()

    print(
        row["metric"]
    )

    print(
        f"Fixed: "
        f"{row['fixed_mean']:.2f}"
    )

    print(
        f"Gradient-Balanced: "
        f"{row['gradient_balanced_mean']:.2f}"
    )

    print(
        f"Paired Difference: "
        f"{row['paired_difference']:+.2f} pp"
    )

    print(
        f"95% CI: "
        f"[{row['ci_lower']:+.2f}, "
        f"{row['ci_upper']:+.2f}] pp"
    )

    print(
        f"Cohen's dz: "
        f"{row['cohens_dz']:+.3f}"
    )

    print(
        f"Exact two-sided p-value: "
        f"{row['exact_p']:.4f}"
    )


# ============================================================
# Key findings
# ============================================================

print()
print("=" * 90)

print(
    "KEY EXPERIMENTAL FINDINGS"
)

print("=" * 90)

print(
    "1. Replay and LwF individually provide limited "
    "protection against catastrophic forgetting."
)

print(
    "2. Combining Replay and LwF produces a major "
    "improvement over the FedAvg-FCL baseline."
)

print(
    "3. Gradient-Balanced FCL further reduces average "
    "forgetting from 4.76 pp to 1.77 pp across five seeds."
)

print(
    "4. This corresponds to a 62.89% relative reduction "
    "in forgetting."
)

print(
    "5. The retention improvement is accompanied by a "
    "2.18 pp reduction in final average accuracy."
)

print(
    "6. Gradient-Balanced reduces forgetting for all five "
    "tested seeds."
)

print(
    "7. Seed-42 heterogeneity experiments show lower "
    "forgetting at alpha = 0.1, 0.5, and 1.0."
)

print(
    "8. Adaptive weights range from 1.0619 to 1.4903, "
    "showing that the mechanism does not collapse to the "
    "fixed weight of 1.0."
)

print()
print(
    "Paper tables saved successfully."
)

print()

print(
    "results/paper_table_ablation.csv"
)

print(
    "results/paper_table_main_5seeds.csv"
)

print(
    "results/paper_table_heterogeneity.csv"
)

print(
    "results/paper_table_adaptive_weights.csv"
)

print(
    "results/paper_table_statistics.csv"
)