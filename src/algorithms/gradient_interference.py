import torch


def flatten_gradients(
    model
) -> torch.Tensor:
    """
    Collect all available parameter gradients from
    a model and concatenate them into one vector.

    Parameters whose gradients are None are represented
    by zeros so that gradient vectors always have the
    same dimensionality.
    """

    gradient_parts = []

    for parameter in model.parameters():

        if not parameter.requires_grad:
            continue

        if parameter.grad is None:

            gradient_part = torch.zeros_like(
                parameter,
                dtype=torch.float32
            )

        else:

            gradient_part = (
                parameter.grad
                .detach()
                .float()
                .clone()
            )

        gradient_parts.append(
            gradient_part.reshape(-1)
        )

    if not gradient_parts:

        return torch.empty(
            0,
            dtype=torch.float32
        )

    return torch.cat(
        gradient_parts
    )


def gradient_cosine_similarity(
    gradient_a: torch.Tensor,
    gradient_b: torch.Tensor
) -> float:
    """
    Calculate cosine similarity between two
    flattened gradient vectors.

    Interpretation:

        +1.0 -> strongly aligned
         0.0 -> approximately orthogonal
        -1.0 -> strongly conflicting
    """

    if gradient_a.numel() != gradient_b.numel():

        raise ValueError(
            "Gradient vectors must contain "
            "the same number of elements."
        )

    if gradient_a.numel() == 0:
        return 0.0

    vector_a = (
        gradient_a
        .detach()
        .float()
        .reshape(-1)
    )

    vector_b = (
        gradient_b
        .detach()
        .float()
        .reshape(-1)
    )

    norm_a = torch.linalg.vector_norm(
        vector_a
    )

    norm_b = torch.linalg.vector_norm(
        vector_b
    )

    if (
        norm_a.item() == 0.0
        or norm_b.item() == 0.0
    ):
        return 0.0

    similarity = torch.dot(
        vector_a,
        vector_b
    ) / (
        norm_a * norm_b
    )

    similarity = torch.clamp(
        similarity,
        min=-1.0,
        max=1.0
    )

    return float(
        similarity.item()
    )


def calculate_interference_score(
    gradient_old: torch.Tensor,
    gradient_new: torch.Tensor
) -> float:
    """
    Measure destructive gradient interference.

    The score is:

        max(0, -cosine_similarity)

    Therefore:

        similarity >= 0
            -> interference = 0

        similarity < 0
            -> interference > 0

    Range:

        0.0 -> no destructive interference
        1.0 -> maximally opposing gradients
    """

    similarity = (
        gradient_cosine_similarity(
            gradient_old,
            gradient_new
        )
    )

    interference = max(
        0.0,
        -similarity
    )

    return float(
        interference
    )


def calculate_gradient_norm(
    gradient: torch.Tensor
) -> float:
    """
    Return the L2 norm of a flattened
    gradient vector.
    """

    if gradient.numel() == 0:
        return 0.0

    gradient_vector = (
        gradient
        .detach()
        .float()
        .reshape(-1)
    )

    norm = torch.linalg.vector_norm(
        gradient_vector
    )

    return float(
        norm.item()
    )


def calculate_gradient_ratio(
    gradient_old: torch.Tensor,
    gradient_new: torch.Tensor,
    epsilon: float = 1e-12
) -> float:
    """
    Compare the magnitude of the old-task gradient
    with the magnitude of the new-task gradient.

    ratio = ||g_old|| / (||g_new|| + epsilon)

    This is kept separate from the interference
    score because direction and magnitude represent
    different phenomena.
    """

    old_norm = calculate_gradient_norm(
        gradient_old
    )

    new_norm = calculate_gradient_norm(
        gradient_new
    )

    ratio = (
        old_norm
        / (
            new_norm
            + epsilon
        )
    )

    return float(
        ratio
    )


def gradient_diagnostic(
    gradient_old: torch.Tensor,
    gradient_new: torch.Tensor
) -> dict[str, float]:
    """
    Return the complete gradient-interference
    diagnostic for one old/new gradient pair.
    """

    similarity = (
        gradient_cosine_similarity(
            gradient_old,
            gradient_new
        )
    )

    interference = max(
        0.0,
        -similarity
    )

    old_norm = calculate_gradient_norm(
        gradient_old
    )

    new_norm = calculate_gradient_norm(
        gradient_new
    )

    magnitude_ratio = (
        calculate_gradient_ratio(
            gradient_old,
            gradient_new
        )
    )

    return {
        "cosine_similarity": similarity,
        "interference": interference,
        "old_gradient_norm": old_norm,
        "new_gradient_norm": new_norm,
        "magnitude_ratio": magnitude_ratio
    }