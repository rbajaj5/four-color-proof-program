import pytest

from src.y_majority_coarse_grain import (
    compact_mask_to_coloring,
    coloring_to_compact_mask,
    majority_reduce,
    reduce_to_single_color,
    triangular_coordinates,
    triangular_neighbors,
    y_winners,
)


def test_coordinate_and_mask_round_trip() -> None:
    size = 4
    for mask in (0, 1, 19, (1 << len(triangular_coordinates(size))) - 1):
        coloring = compact_mask_to_coloring(size, mask)
        assert coloring_to_compact_mask(size, coloring) == mask


def test_triangular_neighbors_are_symmetric() -> None:
    size = 5
    for coordinate in triangular_coordinates(size):
        for neighbor in triangular_neighbors(size, coordinate):
            assert coordinate in triangular_neighbors(size, neighbor)


@pytest.mark.parametrize("size", (1, 2, 3, 4))
def test_exhaustive_y_uniqueness_and_reduction(size: int) -> None:
    cell_count = len(triangular_coordinates(size))
    for mask in range(1 << cell_count):
        coloring = compact_mask_to_coloring(size, mask)
        blue, yellow = y_winners(size, coloring)
        assert blue ^ yellow
        assert reduce_to_single_color(size, coloring) == blue


def test_majority_reduction_dimension() -> None:
    size = 5
    coloring = compact_mask_to_coloring(size, 0b101101001011010)
    reduced = majority_reduce(size, coloring)
    assert set(reduced) == set(triangular_coordinates(size - 1))
