import os

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Output directory
# ============================================================

os.makedirs(
    "../../figures",
    exist_ok=True
)


# ============================================================
# Main five-seed results
# ============================================================

methods = [
    "Fixed Replay+LwF",
    "Gradient-Balanced"
]

accuracy_means = [
    46.94,
    44.76
]

accuracy_stds = [
    3.40,
    3.50
]

forgetting_means = [
    4.76,
    1.77
]

forgetting_stds = [
    0.76,
    0.34
]


# ============================================================
# Figure 1
# Final average accuracy
# ============================================================

x = np.arange(
    len(methods)
)

fig, ax = plt.subplots(
    figsize=(7, 5)
)

bars = ax.bar(
    x,
    accuracy_means,
    yerr=accuracy_stds,
    capsize=7
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    methods
)

ax.set_ylabel(
    "Final Average Accuracy (%)"
)

ax.set_title(
    "Final Average Accuracy Across Five Seeds"
)

ax.set_ylim(
    0,
    55
)

ax.grid(
    axis="y",
    alpha=0.25
)

for bar, mean_value in zip(
    bars,
    accuracy_means
):

    ax.text(
        bar.get_x()
        + bar.get_width() / 2,
        bar.get_height() + 1.0,
        f"{mean_value:.2f}%",
        ha="center",
        va="bottom",
        fontsize=10
    )


fig.tight_layout()

fig.savefig(
    "figures/"
    "fcl_five_seed_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    "figures/"
    "fcl_five_seed_accuracy.pdf",
    bbox_inches="tight"
)

plt.close(
    fig
)


# ============================================================
# Figure 2
# Average forgetting
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 5)
)

bars = ax.bar(
    x,
    forgetting_means,
    yerr=forgetting_stds,
    capsize=7
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    methods
)

ax.set_ylabel(
    "Average Forgetting (percentage points)"
)

ax.set_title(
    "Average Forgetting Across Five Seeds"
)

ax.set_ylim(
    0,
    6
)

ax.grid(
    axis="y",
    alpha=0.25
)

for bar, mean_value in zip(
    bars,
    forgetting_means
):

    ax.text(
        bar.get_x()
        + bar.get_width() / 2,
        bar.get_height() + 0.20,
        f"{mean_value:.2f}",
        ha="center",
        va="bottom",
        fontsize=10
    )


fig.tight_layout()

fig.savefig(
    "figures/"
    "fcl_five_seed_forgetting.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    "figures/"
    "fcl_five_seed_forgetting.pdf",
    bbox_inches="tight"
)

plt.close(
    fig
)


# ============================================================
# Heterogeneity robustness data
# ============================================================

alpha_values = np.array([
    0.1,
    0.5,
    1.0
])

fixed_accuracy = np.array([
    32.19,
    46.57,
    48.29
])

adaptive_accuracy = np.array([
    28.28,
    45.77,
    44.46
])

fixed_forgetting = np.array([
    7.56,
    4.80,
    5.15
])

adaptive_forgetting = np.array([
    3.99,
    2.01,
    1.47
])


# ============================================================
# Figure 3
# Heterogeneity vs accuracy
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 5)
)

ax.plot(
    alpha_values,
    fixed_accuracy,
    marker="o",
    linewidth=2,
    label="Fixed Replay+LwF"
)

ax.plot(
    alpha_values,
    adaptive_accuracy,
    marker="s",
    linewidth=2,
    label="Gradient-Balanced"
)

ax.set_xlabel(
    "Dirichlet α"
)

ax.set_ylabel(
    "Final Average Accuracy (%)"
)

ax.set_title(
    "Accuracy Under Different Client Heterogeneity Levels"
)

ax.set_xticks(
    alpha_values
)

ax.grid(
    alpha=0.25
)

ax.legend()

fig.tight_layout()

fig.savefig(
    "figures/"
    "fcl_alpha_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    "figures/"
    "fcl_alpha_accuracy.pdf",
    bbox_inches="tight"
)

plt.close(
    fig
)


# ============================================================
# Figure 4
# Heterogeneity vs forgetting
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 5)
)

ax.plot(
    alpha_values,
    fixed_forgetting,
    marker="o",
    linewidth=2,
    label="Fixed Replay+LwF"
)

ax.plot(
    alpha_values,
    adaptive_forgetting,
    marker="s",
    linewidth=2,
    label="Gradient-Balanced"
)

ax.set_xlabel(
    "Dirichlet α"
)

ax.set_ylabel(
    "Average Forgetting (percentage points)"
)

ax.set_title(
    "Forgetting Under Different Client Heterogeneity Levels"
)

ax.set_xticks(
    alpha_values
)

ax.grid(
    alpha=0.25
)

ax.legend()

fig.tight_layout()

fig.savefig(
    "figures/"
    "fcl_alpha_forgetting.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    "figures/"
    "fcl_alpha_forgetting.pdf",
    bbox_inches="tight"
)

plt.close(
    fig
)


# ============================================================
# Figure 5
# Relative forgetting reduction
# ============================================================

relative_reduction = np.array([
    47.27,
    58.07,
    71.36
])

fig, ax = plt.subplots(
    figsize=(7, 5)
)

ax.plot(
    alpha_values,
    relative_reduction,
    marker="o",
    linewidth=2
)

ax.set_xlabel(
    "Dirichlet α"
)

ax.set_ylabel(
    "Relative Forgetting Reduction (%)"
)

ax.set_title(
    "Retention Benefit of Gradient-Balanced FCL"
)

ax.set_xticks(
    alpha_values
)

ax.set_ylim(
    0,
    80
)

ax.grid(
    alpha=0.25
)

for alpha_value, reduction in zip(
    alpha_values,
    relative_reduction
):

    ax.annotate(
        f"{reduction:.1f}%",
        (
            alpha_value,
            reduction
        ),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center"
    )


fig.tight_layout()

fig.savefig(
    "figures/"
    "fcl_alpha_relative_forgetting_reduction.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    "figures/"
    "fcl_alpha_relative_forgetting_reduction.pdf",
    bbox_inches="tight"
)

plt.close(
    fig
)


# ============================================================
# Finished
# ============================================================

print()
print("=" * 70)

print(
    "FCL publication figures created successfully."
)

print("=" * 70)

print(
    "figures/fcl_five_seed_accuracy.png"
)

print(
    "figures/fcl_five_seed_forgetting.png"
)

print(
    "figures/fcl_alpha_accuracy.png"
)

print(
    "figures/fcl_alpha_forgetting.png"
)

print(
    "figures/fcl_alpha_relative_forgetting_reduction.png"
)

print()

print(
    "PDF versions were also created "
    "for publication."
)