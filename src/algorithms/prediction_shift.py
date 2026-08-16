import math

import torch
import torch.nn.functional as F


def normalized_prediction_entropy(
    logits: torch.Tensor
) -> torch.Tensor:
    """
    Calculate normalized predictive entropy
    for a batch of model outputs.

    Returns one entropy value per sample.

    Output range:
        approximately 0.0 to 1.0

    Low entropy:
        model is confident

    High entropy:
        model is uncertain
    """

    if logits.ndim != 2:
        raise ValueError(
            "logits must have shape "
            "[batch_size, num_classes]"
        )

    num_classes = logits.size(1)

    if num_classes <= 1:
        raise ValueError(
            "The model must have at least "
            "two output classes."
        )

    probabilities = F.softmax(
        logits,
        dim=1
    )

    log_probabilities = F.log_softmax(
        logits,
        dim=1
    )

    entropy = -torch.sum(
        probabilities * log_probabilities,
        dim=1
    )

    maximum_entropy = math.log(
        num_classes
    )

    normalized_entropy = (
        entropy / maximum_entropy
    )

    return torch.clamp(
        normalized_entropy,
        min=0.0,
        max=1.0
    )


def calculate_prediction_shift(
    model,
    data_loader,
    device
) -> float:
    """
    Estimate how unfamiliar a client's
    current data is to the existing model.

    The score is the mean normalized
    predictive entropy over the client's
    samples.

    Output:
        0.0 -> highly confident predictions
        1.0 -> highly uncertain predictions
    """

    model.eval()

    total_entropy = 0.0
    total_samples = 0

    with torch.no_grad():

        for images, _ in data_loader:

            images = images.to(device)

            logits = model(images)

            entropy = (
                normalized_prediction_entropy(
                    logits
                )
            )

            batch_size = images.size(0)

            total_entropy += (
                entropy.sum().item()
            )

            total_samples += batch_size

    if total_samples == 0:
        return 0.0

    average_entropy = (
        total_entropy
        / total_samples
    )

    return max(
        0.0,
        min(
            1.0,
            average_entropy
        )
    )