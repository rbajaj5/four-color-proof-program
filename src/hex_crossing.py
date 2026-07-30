"""Bitboard primitives and reliability statistics for rhombic Hex boards."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


def cell_count(size: int) -> int:
    if size < 1:
        raise ValueError("size must be positive")
    return size * size


def full_mask(size: int) -> int:
    count = cell_count(size)
    if count > 62:
        raise ValueError("signed int64 bitboards support at most 62 cells")
    return (1 << count) - 1


def side_mask(size: int, side: str) -> int:
    predicates = {
        "q0": lambda q, _r: q == 0,
        "q1": lambda q, _r: q == size - 1,
        "r0": lambda _q, r: r == 0,
        "r1": lambda _q, r: r == size - 1,
    }
    try:
        predicate = predicates[side]
    except KeyError as error:
        raise ValueError(f"unknown side: {side}") from error
    return sum(
        1 << (q * size + r)
        for q in range(size)
        for r in range(size)
        if predicate(q, r)
    )


def neighbor_shift_specs(size: int) -> tuple[tuple[int, int], ...]:
    """Return source masks and bit shifts for the six axial neighbors."""

    full_mask(size)
    directions = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1))
    specs = []
    for dq, dr in directions:
        source_mask = 0
        for q in range(size):
            for r in range(size):
                if 0 <= q + dq < size and 0 <= r + dr < size:
                    source_mask |= 1 << (q * size + r)
        specs.append((source_mask, dq * size + dr))
    return tuple(specs)


def expand_neighbors(
    reach: torch.Tensor,
    specs: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    expanded = reach
    for source_mask, shift in specs:
        source = reach & source_mask
        expanded = (
            expanded | (source << shift)
            if shift >= 0
            else expanded | (source >> -shift)
        )
    return expanded


def flood(
    active: torch.Tensor,
    seed_mask: int,
    *,
    size: int,
    specs: tuple[tuple[int, int], ...] | None = None,
) -> torch.Tensor:
    if specs is None:
        specs = neighbor_shift_specs(size)
    reach = active & seed_mask
    for _ in range(cell_count(size)):
        reach = expand_neighbors(reach, specs) & active
    return reach


def crossing_outcomes(
    blue: torch.Tensor,
    size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return horizontal-blue and vertical-yellow crossing indicators."""

    specs = neighbor_shift_specs(size)
    yellow = blue ^ full_mask(size)
    blue_reach = flood(blue, side_mask(size, "q0"), size=size, specs=specs)
    yellow_reach = flood(yellow, side_mask(size, "r0"), size=size, specs=specs)
    return (
        (blue_reach & side_mask(size, "q1")).ne(0),
        (yellow_reach & side_mask(size, "r1")).ne(0),
    )


def popcount(states: torch.Tensor, count: int) -> torch.Tensor:
    weights = torch.zeros_like(states)
    for index in range(count):
        weights += (states >> index) & 1
    return weights


def reliability_probability(
    blue_winner_counts: Sequence[int],
    probability: float,
) -> float:
    """Evaluate the blue-crossing reliability polynomial."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    count = len(blue_winner_counts) - 1
    return math.fsum(
        coefficient
        * probability**blue_cells
        * (1.0 - probability) ** (count - blue_cells)
        for blue_cells, coefficient in enumerate(blue_winner_counts)
    )


def reliability_derivative(
    blue_winner_counts: Sequence[int],
    probability: float,
) -> float:
    """Differentiate the reliability polynomial term by term."""

    if not 0.0 < probability < 1.0:
        raise ValueError("derivative evaluator requires probability in (0, 1)")
    count = len(blue_winner_counts) - 1
    return math.fsum(
        coefficient
        * probability**blue_cells
        * (1.0 - probability) ** (count - blue_cells)
        * (
            blue_cells / probability
            - (count - blue_cells) / (1.0 - probability)
        )
        for blue_cells, coefficient in enumerate(blue_winner_counts)
    )


def inverse_reliability(
    blue_winner_counts: Sequence[int],
    target: float,
    *,
    iterations: int = 80,
) -> float:
    if not 0.0 < target < 1.0:
        raise ValueError("target must lie in (0, 1)")
    low, high = 0.0, 1.0
    for _ in range(iterations):
        midpoint = (low + high) / 2.0
        if reliability_probability(blue_winner_counts, midpoint) < target:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0
