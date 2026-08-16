from collections.abc import Mapping

import torch


def flatten_model_update(
    global_state: Mapping[str, torch.Tensor],
    local_state: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    """
    Convert the difference between a local model
    and the global model into one flat vector.

    Update = local_parameters - global_parameters
    """

    update_parts = []

    for name, global_parameter in global_state.items():

        if name not in local_state:
            raise KeyError(
                f"Parameter '{name}' is missing "
                f"from the local state."
            )

        local_parameter = local_state[name]

        # Ignore non-floating tensors such as
        # integer bookkeeping buffers.
        if not torch.is_floating_point(
            global_parameter
        ):
            continue

        difference = (
            local_parameter.detach().float().cpu()
            - global_parameter.detach().float().cpu()
        )

        update_parts.append(
            difference.reshape(-1)
        )

    if not update_parts:
        return torch.empty(
            0,
            dtype=torch.float32
        )

    return torch.cat(
        update_parts
    )


def cosine_similarity(
    vector_a: torch.Tensor,
    vector_b: torch.Tensor
) -> float:
    """
    Calculate cosine similarity between two
    parameter-update vectors.

    Range:
        -1 -> opposite directions
         0 -> orthogonal
        +1 -> same direction
    """

    if vector_a.numel() != vector_b.numel():
        raise ValueError(
            "Update vectors must have "
            "the same number of elements."
        )

    if vector_a.numel() == 0:
        return 0.0

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

    return float(
        torch.clamp(
            similarity,
            min=-1.0,
            max=1.0
        ).item()
    )


def calculate_reference_update(
    update_vectors: list[torch.Tensor],
    sample_counts: list[int] | None = None
) -> torch.Tensor:
    """
    Calculate a reference update.

    If sample counts are provided, use a
    sample-weighted average similar to FedAvg.
    Otherwise, use an unweighted average.
    """

    if len(update_vectors) == 0:
        raise ValueError(
            "At least one update vector "
            "is required."
        )

    vector_size = (
        update_vectors[0].numel()
    )

    for vector in update_vectors:
        if vector.numel() != vector_size:
            raise ValueError(
                "All update vectors must "
                "have the same size."
            )

    if sample_counts is None:

        stacked = torch.stack(
            update_vectors
        )

        return torch.mean(
            stacked,
            dim=0
        )

    if len(sample_counts) != len(
        update_vectors
    ):
        raise ValueError(
            "sample_counts must contain one "
            "value per update vector."
        )

    total_samples = sum(
        sample_counts
    )

    if total_samples <= 0:
        raise ValueError(
            "Total sample count must "
            "be positive."
        )

    reference = torch.zeros_like(
        update_vectors[0]
    )

    for vector, count in zip(
        update_vectors,
        sample_counts
    ):
        weight = (
            count / total_samples
        )

        reference += (
            weight * vector
        )

    return reference


def calculate_conflict_score(
    client_update: torch.Tensor,
    reference_update: torch.Tensor
) -> float:
    """
    Convert cosine similarity into a normalized
    conflict score between 0 and 1.

    similarity = +1 -> conflict = 0
    similarity =  0 -> conflict = 0.5
    similarity = -1 -> conflict = 1
    """

    similarity = cosine_similarity(
        client_update,
        reference_update
    )

    conflict = (
        1.0 - similarity
    ) / 2.0

    return max(
        0.0,
        min(
            1.0,
            conflict
        )
    )