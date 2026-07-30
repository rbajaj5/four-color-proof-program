import pytest
import torch

from src.hex_correlated_fields import (
    axial_neighbor_agreement,
    boolean_boards_to_bitboards,
    interpolate_probability_level,
)


def test_boolean_board_conversion() -> None:
    boards = torch.tensor(
        [
            [[True, False], [False, True]],
            [[False, True], [True, False]],
        ]
    )
    assert boolean_boards_to_bitboards(boards).tolist() == [9, 6]


def test_axial_neighbor_agreement() -> None:
    constant = torch.ones((1, 3, 3), dtype=torch.bool)
    assert axial_neighbor_agreement(constant).item() == 1.0
    boards = torch.stack((constant[0], ~constant[0]))
    assert torch.all(axial_neighbor_agreement(boards) == 1.0)


def test_interpolate_probability_level() -> None:
    probabilities = (0.3, 0.4, 0.5, 0.6, 0.7)
    values = (0.1, 0.2, 0.5, 0.8, 0.9)
    assert interpolate_probability_level(probabilities, values, 0.25) == pytest.approx(
        0.4166666666666667
    )
    assert interpolate_probability_level(probabilities, values, 0.75) == pytest.approx(
        0.5833333333333334
    )
