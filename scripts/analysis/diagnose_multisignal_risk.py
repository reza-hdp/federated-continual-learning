import csv
import math


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = (
    "results/"
    "conflict_forgetting_diagnostic_all.csv"
)

ALPHAS = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0
]


# ============================================================
# Pearson correlation
# ============================================================

def pearson_correlation(
    values_x,
    values_y
):
    if len(values_x) != len(values_y):
        raise ValueError(
            "Input lists must have "
            "the same length."
        )

    if len(values_x) < 2:
        return 0.0

    mean_x = (
        sum(values_x)
        / len(values_x)
    )

    mean_y = (
        sum(values_y)
        / len(values_y)
    )

    differences_x = [
        value - mean_x
        for value in values_x
    ]

    differences_y = [
        value - mean_y
        for value in values_y
    ]

    numerator = sum(
        difference_x * difference_y
        for difference_x, difference_y
        in zip(
            differences_x,
            differences_y
        )
    )

    denominator_x = sum(
        difference ** 2
        for difference in differences_x
    )

    denominator_y = sum(
        difference ** 2
        for difference in differences_y
    )

    denominator = math.sqrt(
        denominator_x
        * denominator_y
    )

    if denominator == 0.0:
        return 0.0

    return (
        numerator / denominator
    )


# ============================================================
# Rank values
# ============================================================

def rank_values(values):
    indexed_values = list(
        enumerate(values)
    )

    sorted_values = sorted(
        indexed_values,
        key=lambda item: item[1]
    )

    ranks = [
        0.0
        for _ in values
    ]

    position = 0

    while position < len(
        sorted_values
    ):

        end_position = position

        current_value = (
            sorted_values[
                position
            ][1]
        )

        while (
            end_position + 1
            < len(sorted_values)
            and sorted_values[
                end_position + 1
            ][1]
            == current_value
        ):
            end_position += 1

        average_rank = (
            position
            + end_position
        ) / 2.0 + 1.0

        for rank_position in range(
            position,
            end_position + 1
        ):

            original_index = (
                sorted_values[
                    rank_position
                ][0]
            )

            ranks[
                original_index
            ] = average_rank

        position = (
            end_position + 1
        )

    return ranks


# ============================================================
# Spearman correlation
# ============================================================

def spearman_correlation(
    values_x,
    values_y
):
    ranks_x = rank_values(
        values_x
    )

    ranks_y = rank_values(
        values_y
    )

    return pearson_correlation(
        ranks_x,
        ranks_y
    )


# ============================================================
# Load diagnostic data
# ============================================================

records = []

with open(
    INPUT_FILE,
    "r",
    newline=""
) as input_file:

    reader = csv.DictReader(
        input_file
    )

    for row in reader:

        record = {
            "seed": int(
                row["seed"]
            ),
            "transition": (
                row["transition"]
            ),
            "client": int(
                row["client"]
            ),
            "conflict": float(
                row["conflict"]
            ),
            "before_accuracy": float(
                row["before_accuracy"]
            ),
            "forgetting": float(
                row["forgetting"]
            )
        }

        records.append(
            record
        )


# ============================================================
# Construct signals
# ============================================================

conflicts = [
    record["conflict"]
    for record in records
]

vulnerabilities = [
    max(
        0.0,
        min(
            1.0,
            1.0
            - record[
                "before_accuracy"
            ] / 100.0
        )
    )
    for record in records
]

forgetting = [
    record["forgetting"]
    for record in records
]


# ============================================================
# Individual signal diagnostics
# ============================================================

conflict_pearson = (
    pearson_correlation(
        conflicts,
        forgetting
    )
)

conflict_spearman = (
    spearman_correlation(
        conflicts,
        forgetting
    )
)

vulnerability_pearson = (
    pearson_correlation(
        vulnerabilities,
        forgetting
    )
)

vulnerability_spearman = (
    spearman_correlation(
        vulnerabilities,
        forgetting
    )
)


print()
print("Individual Signal Correlations")
print("=" * 80)

print(
    f"Conflict      | "
    f"Pearson: "
    f"{conflict_pearson:+.4f} | "
    f"Spearman: "
    f"{conflict_spearman:+.4f}"
)

print(
    f"Vulnerability | "
    f"Pearson: "
    f"{vulnerability_pearson:+.4f} | "
    f"Spearman: "
    f"{vulnerability_spearman:+.4f}"
)


# ============================================================
# Test combined risk
# ============================================================

print()
print("Combined Risk Search")
print("=" * 80)

results = []

for alpha in ALPHAS:

    combined_risk = [
        (
            alpha * conflict
            + (1.0 - alpha)
            * vulnerability
        )
        for conflict, vulnerability
        in zip(
            conflicts,
            vulnerabilities
        )
    ]

    pearson = (
        pearson_correlation(
            combined_risk,
            forgetting
        )
    )

    spearman = (
        spearman_correlation(
            combined_risk,
            forgetting
        )
    )

    average_correlation = (
        pearson + spearman
    ) / 2.0

    results.append({
        "alpha": alpha,
        "pearson": pearson,
        "spearman": spearman,
        "average": (
            average_correlation
        )
    })

    print(
        f"Alpha {alpha:.1f} | "
        f"Pearson: "
        f"{pearson:+.4f} | "
        f"Spearman: "
        f"{spearman:+.4f} | "
        f"Mean: "
        f"{average_correlation:+.4f}"
    )


# ============================================================
# Best combination
# ============================================================

best_result = max(
    results,
    key=lambda result: (
        result["average"]
    )
)


print()
print("Best Combined Risk")
print("=" * 80)

print(
    f"Best Alpha: "
    f"{best_result['alpha']:.1f}"
)

print(
    f"Conflict Weight: "
    f"{best_result['alpha']:.1f}"
)

print(
    f"Vulnerability Weight: "
    f"{1.0 - best_result['alpha']:.1f}"
)

print(
    f"Pearson Correlation: "
    f"{best_result['pearson']:+.4f}"
)

print(
    f"Spearman Correlation: "
    f"{best_result['spearman']:+.4f}"
)

print(
    f"Mean Correlation: "
    f"{best_result['average']:+.4f}"
)


# ============================================================
# Compare with conflict alone
# ============================================================

pearson_gain = (
    best_result["pearson"]
    - conflict_pearson
)

spearman_gain = (
    best_result["spearman"]
    - conflict_spearman
)


print()
print("Improvement over Conflict Alone")
print("=" * 80)

print(
    f"Pearson Gain: "
    f"{pearson_gain:+.4f}"
)

print(
    f"Spearman Gain: "
    f"{spearman_gain:+.4f}"
)


# ============================================================
# Interpretation
# ============================================================

print()
print("Multi-Signal Interpretation")
print("=" * 80)

if (
    pearson_gain >= 0.10
    and spearman_gain >= 0.10
):

    print(
        "RESULT: Combining conflict and "
        "retention vulnerability substantially "
        "improves forgetting-risk prediction."
    )

elif (
    pearson_gain > 0.0
    and spearman_gain > 0.0
):

    print(
        "RESULT: The combined signal improves "
        "prediction, but the gain is modest."
    )

else:

    print(
        "RESULT: This vulnerability signal does "
        "not improve conflict-based prediction."
    )
