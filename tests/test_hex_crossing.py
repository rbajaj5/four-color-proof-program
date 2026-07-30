import math

import pytest
import torch

from src.hex_crossing import (
    crossing_outcomes,
    inverse_reliability,
    popcount,
    reliability_derivative,
    reliability_probability,
)


def exact_coefficients(size: int) -> list[int]:
    count = size * size
    states = torch.arange(1 << count, dtype=torch.int64)
    blue, yellow = crossing_outcomes(states, size)
    assert bool(torch.all(blue ^ yellow))
    weights = popcount(states, count)
    return torch.bincount(weights[blue], minlength=count + 1).tolist()


@pytest.mark.parametrize("size", (1, 2, 3))
def test_exactly_one_hex_winner(size: int) -> None:
    count = size * size
    states = torch.arange(1 << count, dtype=torch.int64)
    blue, yellow = crossing_outcomes(states, size)
    assert not bool(torch.any(blue & yellow))
    assert not bool(torch.any(~(blue | yellow)))


@pytest.mark.parametrize("size", (1, 2, 3))
def test_reliability_duality_and_half_probability(size: int) -> None:
    coefficients = exact_coefficients(size)
    count = size * size
    for blue_cells, coefficient in enumerate(coefficients):
        assert coefficient + coefficients[count - blue_cells] == math.comb(
            count,
            blue_cells,
        )
    assert reliability_probability(coefficients, 0.5) == pytest.approx(0.5)


def test_reliability_inverse_and_derivative() -> None:
    coefficients = exact_coefficients(3)
    low = inverse_reliability(coefficients, 0.25)
    high = inverse_reliability(coefficients, 0.75)
    assert low < 0.5 < high
    assert low == pytest.approx(1.0 - high)
    assert reliability_derivative(coefficients, 0.5) > 1.0
