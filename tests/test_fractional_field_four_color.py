from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from src.fractional_field_four_color import (
    coloring_is_proper,
    compactified_triangulation_adjacency,
    compactified_triangulation_degrees,
    curvature_triangulation_statistics,
    diagonal_majority_transport_mismatch,
    dsatur_coloring,
    estimate_hurst_from_structure_function,
    fractional_gaussian_surfaces,
    mixed_curvature_diagonals,
    periodic_closed_window,
    subsample_closed_window,
)


def test_fractional_surface_is_seeded_centered_and_normalized() -> None:
    first = fractional_gaussian_surfaces(
        4,
        32,
        0.5,
        seed=17,
        dtype=torch.float64,
    )
    second = fractional_gaussian_surfaces(
        4,
        32,
        0.5,
        seed=17,
        dtype=torch.float64,
    )
    assert torch.equal(first, second)
    assert torch.allclose(
        first.mean(dim=(1, 2)),
        torch.zeros(4, dtype=torch.float64),
        atol=1e-14,
    )
    assert torch.allclose(
        first.std(dim=(1, 2)),
        torch.ones(4, dtype=torch.float64),
        atol=1e-14,
    )


def test_periodic_window_and_nested_subsampling() -> None:
    surfaces = torch.arange(64, dtype=torch.float64).reshape(1, 8, 8)
    window = periodic_closed_window(surfaces)
    assert window.shape == (1, 9, 9)
    assert torch.equal(window[:, -1, :-1], surfaces[:, 0, :])
    assert torch.equal(window[:, :-1, -1], surfaces[:, :, 0])
    assert subsample_closed_window(window, 2).shape == (1, 5, 5)


def test_compactification_is_a_sphere_triangulation() -> None:
    diagonals = torch.tensor(
        [
            [
                [True, False, True],
                [False, True, False],
                [True, True, False],
            ]
        ]
    )
    degrees = compactified_triangulation_degrees(diagonals)
    vertex_count = 4 * 4 + 1
    edge_count = 3 * vertex_count - 6
    assert degrees.shape == (1, vertex_count)
    assert int(degrees.sum()) == 2 * edge_count


def test_checkerboard_is_three_colorable_and_one_flip_requires_four() -> None:
    row = torch.tensor((True, False, True, False))
    checkerboard = torch.stack((row, ~row, row, ~row))
    three_degrees = compactified_triangulation_degrees(
        checkerboard.unsqueeze(0)
    )
    assert torch.all(three_degrees % 2 == 0)
    four_pattern = checkerboard.clone()
    four_pattern[0, 0] = ~four_pattern[0, 0]
    four_degrees = compactified_triangulation_degrees(
        four_pattern.unsqueeze(0)
    )
    assert torch.any(four_degrees % 2 == 1)


def test_nested_diagonal_transport_detects_mismatch() -> None:
    coarse = torch.tensor([[[True, False], [False, True]]])
    fine = coarse.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
    assert float(diagonal_majority_transport_mismatch(coarse, fine)[0]) == 0
    fine[:, :2, :2] = False
    assert float(diagonal_majority_transport_mismatch(coarse, fine)[0]) == 0.25


def test_chromatic_classification_and_explicit_four_coloring() -> None:
    window = torch.tensor(
        [
            [
                [0.0, 1.0, -1.0, 0.0],
                [1.0, -2.0, 2.0, 1.0],
                [-1.0, 2.0, -2.0, -1.0],
                [0.0, 1.0, -1.0, 0.0],
            ]
        ],
        dtype=torch.float64,
    )
    statistics = curvature_triangulation_statistics(window)
    assert int(statistics["chromatic_number"][0]) in (3, 4)
    diagonals = mixed_curvature_diagonals(window)[0]
    adjacency = compactified_triangulation_adjacency(diagonals)
    colors = dsatur_coloring(adjacency, 4)
    assert max(colors) <= 3
    assert coloring_is_proper(adjacency, colors)


def test_estimated_hurst_increases_with_input_hurst() -> None:
    rough = fractional_gaussian_surfaces(
        64,
        64,
        0.2,
        seed=91,
        dtype=torch.float64,
    )
    smooth = fractional_gaussian_surfaces(
        64,
        64,
        0.8,
        seed=92,
        dtype=torch.float64,
    )
    rough_estimate = estimate_hurst_from_structure_function(
        rough,
        (1, 2, 4, 8),
    ).mean()
    smooth_estimate = estimate_hurst_from_structure_function(
        smooth,
        (1, 2, 4, 8),
    ).mean()
    assert rough_estimate < smooth_estimate
