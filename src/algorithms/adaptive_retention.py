import math


class AdaptiveRetention:
    def __init__(
        self,
        min_weight=0.2,
        max_weight=2.0
    ):
        if min_weight < 0:
            raise ValueError(
                "min_weight must be non-negative"
            )

        if max_weight <= min_weight:
            raise ValueError(
                "max_weight must be greater than min_weight"
            )

        self.min_weight = min_weight
        self.max_weight = max_weight

    def compute_weight(
        self,
        shift_score,
        forgetting_score
    ):
        shift_score = max(
            0.0,
            min(1.0, shift_score)
        )

        forgetting_score = max(
            0.0,
            min(1.0, forgetting_score)
        )

        combined_risk = (
            shift_score
            * forgetting_score
        )

        weight = (
            self.min_weight
            + (
                self.max_weight
                - self.min_weight
            )
            * combined_risk
        )

        return weight


def normalized_entropy(
    class_probabilities
):
    epsilon = 1e-12

    probabilities = [
        max(epsilon, probability)
        for probability in class_probabilities
    ]

    entropy = -sum(
        probability
        * math.log(probability)
        for probability in probabilities
    )

    max_entropy = math.log(
        len(probabilities)
    )

    if max_entropy == 0:
        return 0.0

    return entropy / max_entropy