import math


class GradientBalancedRetention:
    def __init__(
        self,
        min_weight=0.2,
        max_weight=2.0,
        epsilon=1e-12
    ):
        if min_weight < 0.0:
            raise ValueError(
                "min_weight must be non-negative"
            )

        if max_weight <= min_weight:
            raise ValueError(
                "max_weight must be greater "
                "than min_weight"
            )

        if epsilon <= 0.0:
            raise ValueError(
                "epsilon must be positive"
            )

        self.min_weight = float(
            min_weight
        )

        self.max_weight = float(
            max_weight
        )

        self.epsilon = float(
            epsilon
        )


    def calculate_balance_score(
        self,
        old_gradient_norm,
        new_gradient_norm
    ):
        old_gradient_norm = max(
            0.0,
            float(old_gradient_norm)
        )

        new_gradient_norm = max(
            0.0,
            float(new_gradient_norm)
        )

        denominator = (
            old_gradient_norm
            + new_gradient_norm
            + self.epsilon
        )

        balance_score = (
            new_gradient_norm
            / denominator
        )

        return max(
            0.0,
            min(
                1.0,
                balance_score
            )
        )


    def calculate_weight(
        self,
        old_gradient_norm,
        new_gradient_norm
    ):
        balance_score = (
            self.calculate_balance_score(
                old_gradient_norm,
                new_gradient_norm
            )
        )

        weight = (
            self.min_weight
            + (
                self.max_weight
                - self.min_weight
            )
            * balance_score
        )

        return float(
            weight
        )


    def calculate_diagnostics(
        self,
        old_gradient_norm,
        new_gradient_norm
    ):
        balance_score = (
            self.calculate_balance_score(
                old_gradient_norm,
                new_gradient_norm
            )
        )

        retention_weight = (
            self.calculate_weight(
                old_gradient_norm,
                new_gradient_norm
            )
        )

        old_gradient_norm = float(
            old_gradient_norm
        )

        new_gradient_norm = float(
            new_gradient_norm
        )

        magnitude_ratio = (
            old_gradient_norm
            / (
                new_gradient_norm
                + self.epsilon
            )
        )

        return {
            "old_gradient_norm": (
                old_gradient_norm
            ),
            "new_gradient_norm": (
                new_gradient_norm
            ),
            "magnitude_ratio": (
                magnitude_ratio
            ),
            "balance_score": (
                balance_score
            ),
            "retention_weight": (
                retention_weight
            )
        }