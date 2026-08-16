import math
from typing import Sequence


def normalize_distribution(
    counts: Sequence[int]
) -> list[float]:
    """
    Convert class counts into probabilities.
    """

    total = sum(counts)

    if total == 0:
        return [
            0.0
            for _ in counts
        ]

    return [
        count / total
        for count in counts
    ]


def jensen_shannon_divergence(
    distribution_a: Sequence[float],
    distribution_b: Sequence[float]
) -> float:
    """
    Calculate normalized Jensen-Shannon divergence.

    Output is approximately between 0 and 1.

    0:
        distributions are identical

    1:
        distributions are maximally different
    """

    if len(distribution_a) != len(distribution_b):
        raise ValueError(
            "Distributions must have the same length."
        )

    if len(distribution_a) == 0:
        return 0.0

    epsilon = 1e-12

    p = [
        max(float(value), 0.0)
        for value in distribution_a
    ]

    q = [
        max(float(value), 0.0)
        for value in distribution_b
    ]

    p_sum = sum(p)
    q_sum = sum(q)

    if p_sum == 0.0 or q_sum == 0.0:
        return 0.0

    p = [
        value / p_sum
        for value in p
    ]

    q = [
        value / q_sum
        for value in q
    ]

    midpoint = [
        0.5 * (p_value + q_value)
        for p_value, q_value in zip(p, q)
    ]

    def kl_divergence(
        first: Sequence[float],
        second: Sequence[float]
    ) -> float:

        result = 0.0

        for first_value, second_value in zip(
            first,
            second
        ):
            if first_value <= 0.0:
                continue

            safe_second = max(
                second_value,
                epsilon
            )

            result += (
                first_value
                * math.log(
                    first_value
                    / safe_second
                )
            )

        return result

    js_divergence = 0.5 * (
        kl_divergence(p, midpoint)
        + kl_divergence(q, midpoint)
    )

    # Jensen-Shannon divergence using
    # natural logarithms has maximum log(2).
    normalized_js = (
        js_divergence
        / math.log(2.0)
    )

    return max(
        0.0,
        min(1.0, normalized_js)
    )


def calculate_shift_score(
    previous_counts: Sequence[int],
    current_counts: Sequence[int]
) -> float:
    """
    Calculate distribution shift from
    previous and current class counts.
    """

    if len(previous_counts) != len(current_counts):
        raise ValueError(
            "Class-count vectors must have "
            "the same length."
        )

    previous_distribution = (
        normalize_distribution(
            previous_counts
        )
    )

    current_distribution = (
        normalize_distribution(
            current_counts
        )
    )

    return jensen_shannon_divergence(
        previous_distribution,
        current_distribution
    )


def calculate_forgetting_score(
    best_previous_accuracy: float,
    current_accuracy: float
) -> float:
    """
    Convert accuracy loss into a normalized
    forgetting-risk score between 0 and 1.

    Accuracies are expected as percentages,
    e.g. 82.5 rather than 0.825.
    """

    forgetting = max(
        0.0,
        best_previous_accuracy
        - current_accuracy
    )

    normalized_forgetting = (
        forgetting / 100.0
    )

    return max(
        0.0,
        min(
            1.0,
            normalized_forgetting
        )
    )


def calculate_average_forgetting_risk(
    best_accuracies: Sequence[float],
    current_accuracies: Sequence[float]
) -> float:
    """
    Calculate mean normalized forgetting risk
    across previously learned tasks.
    """

    if len(best_accuracies) != len(
        current_accuracies
    ):
        raise ValueError(
            "Accuracy lists must have "
            "the same length."
        )

    if len(best_accuracies) == 0:
        return 0.0

    risks = [
        calculate_forgetting_score(
            best_accuracy,
            current_accuracy
        )
        for best_accuracy, current_accuracy
        in zip(
            best_accuracies,
            current_accuracies
        )
    ]

    return (
        sum(risks)
        / len(risks)
    )