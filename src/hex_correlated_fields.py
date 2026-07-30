"""Utilities for thresholded correlated-field experiments on Hex boards."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def boolean_boards_to_bitboards(boards: torch.Tensor) -> torch.Tensor:
    if boards.ndim != 3 or boards.shape[-2] != boards.shape[-1]:
        raise ValueError("boards must have shape [batch, n, n]")
    size = boards.shape[-1]
    if size * size > 62:
        raise ValueError("signed int64 bitboards support at most 62 cells")
    states = torch.zeros(
        boards.shape[0],
        dtype=torch.int64,
        device=boards.device,
    )
    flat = boards.reshape(boards.shape[0], -1)
    for index in range(size * size):
        states |= flat[:, index].to(torch.int64) << index
    return states


def axial_neighbor_agreement(boards: torch.Tensor) -> torch.Tensor:
    """Return per-board same-color fractions over undirected Hex edges."""

    if boards.ndim != 3 or boards.shape[-2] != boards.shape[-1]:
        raise ValueError("boards must have shape [batch, n, n]")
    comparisons = (
        boards[:, 1:, :] == boards[:, :-1, :],
        boards[:, :, 1:] == boards[:, :, :-1],
        boards[:, 1:, :-1] == boards[:, :-1, 1:],
    )
    agreements = torch.stack(
        [comparison.sum(dim=(1, 2)) for comparison in comparisons],
        dim=1,
    ).sum(dim=1)
    edge_count = sum(
        comparison.shape[1] * comparison.shape[2]
        for comparison in comparisons
    )
    return agreements.to(torch.float64) / edge_count


def interpolate_probability_level(
    probabilities: Sequence[float],
    values: Sequence[float],
    target: float,
) -> float:
    if len(probabilities) != len(values) or len(values) < 2:
        raise ValueError("probabilities and values must have equal length >= 2")
    if not 0.0 < target < 1.0:
        raise ValueError("target must lie in (0, 1)")
    ordered = sorted(zip(probabilities, values, strict=True))
    monotone_values = []
    running = 0.0
    for probability, value in ordered:
        running = max(running, float(value))
        monotone_values.append((float(probability), running))
    for (left_p, left_v), (right_p, right_v) in zip(
        monotone_values,
        monotone_values[1:],
        strict=True,
    ):
        if left_v <= target <= right_v:
            if right_v == left_v:
                return 0.5 * (left_p + right_p)
            fraction = (target - left_v) / (right_v - left_v)
            return left_p + fraction * (right_p - left_p)
    raise ValueError("target is outside the sampled curve")
