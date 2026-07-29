from __future__ import annotations

import math

import pytest


torch = pytest.importorskip("torch")

from src.quasiperiodic_polyhedral_rg import (
    PHI,
    bounded_defect_relation_residual,
    curve_point_curvatures,
    defect_library,
    fibonacci_axis,
    fibonacci_word,
    itinerary_metrics,
    polyhedral_force_susceptibility,
    projected_winding_numbers,
    quasiperiodic_tetrahedral_itinerary,
    space_form_coupling_force,
    specular_reflection,
    tangent_holonomy,
    tetrahedron_vertex_defects,
    tetrahedron_vertices,
)


def circle_curve(samples: int = 512) -> torch.Tensor:
    theta = torch.linspace(
        0.0,
        2.0 * torch.pi,
        samples + 1,
        dtype=torch.float64,
    )
    return torch.stack(
        (torch.cos(theta), torch.sin(theta), torch.zeros_like(theta)),
        dim=-1,
    )


def test_fibonacci_axis_has_two_golden_tile_lengths() -> None:
    word = fibonacci_word(34)
    assert word.startswith("LSLLS")
    coordinates, axis_word = fibonacci_axis(34, scale=0.5, phase=3)
    lengths = coordinates[1:] - coordinates[:-1]
    assert len(axis_word) == 34
    assert set(axis_word) == {"L", "S"}
    assert sorted(set(round(float(value), 12) for value in lengths)) == [
        0.5,
        round(0.5 * PHI, 12),
    ]


def test_tetrahedron_defects_sum_to_four_pi() -> None:
    vertices = tetrahedron_vertices((1.0, PHI, 1.0), 4)
    defects = tetrahedron_vertex_defects(vertices)
    assert defects.shape == (4,)
    assert float(defects.sum()) == pytest.approx(4.0 * math.pi, rel=1e-12)
    assert bounded_defect_relation_residual(defects) >= 0
    assert len(defect_library()) == 48


def test_inflation_reduces_cell_itinerary_complexity() -> None:
    curve = circle_curve()
    curvatures = curve_point_curvatures(curve)
    fine = quasiperiodic_tetrahedral_itinerary(curve, inflation_level=-1)
    coarse = quasiperiodic_tetrahedral_itinerary(curve, inflation_level=2)
    fine_metrics = itinerary_metrics(fine, curvatures)
    coarse_metrics = itinerary_metrics(coarse, curvatures)
    assert fine_metrics["occupied_tetrahedra"] > coarse_metrics[
        "occupied_tetrahedra"
    ]
    assert fine_metrics["itinerary_transitions"] > coarse_metrics[
        "itinerary_transitions"
    ]
    fine_force = polyhedral_force_susceptibility(fine)
    coarse_force = polyhedral_force_susceptibility(coarse)
    assert fine_force[
        "positive_space_form_force_mean_per_mv2"
    ] > coarse_force["positive_space_form_force_mean_per_mv2"]


def test_space_form_force_has_correct_sign_and_linear_limit() -> None:
    positive = space_form_coupling_force(1.0, 0.4)
    negative = space_form_coupling_force(-1.0, 0.4)
    flat = space_form_coupling_force(0.0, 0.4)
    assert float(positive) > 0.0
    assert float(negative) < 0.0
    assert float(flat) == 0.0

    curvature = torch.tensor((0.01, -0.01), dtype=torch.float64)
    separation = 0.02
    force = space_form_coupling_force(curvature, separation)
    linear = curvature * separation
    assert torch.allclose(force, linear, rtol=5e-7, atol=1e-14)


def test_specular_reflection_is_isometric_and_involutive() -> None:
    velocity = torch.tensor((1.0, -2.0, -3.0), dtype=torch.float64)
    normal = torch.tensor((0.0, 0.0, 2.0), dtype=torch.float64)
    reflected = specular_reflection(velocity, normal)
    assert reflected.tolist() == [1.0, -2.0, 3.0]
    assert torch.linalg.vector_norm(reflected) == pytest.approx(
        torch.linalg.vector_norm(velocity)
    )
    assert torch.allclose(specular_reflection(reflected, normal), velocity)


def test_circle_tangent_holonomy_and_projected_winding() -> None:
    curve = circle_curve()
    holonomy = tangent_holonomy(curve)
    assert holonomy["total_polygonal_turning"] == pytest.approx(
        2.0 * math.pi,
        rel=1e-4,
    )
    assert abs(holonomy["tangent_holonomy_angle"]) < 1e-8
    centers = torch.tensor(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
        dtype=torch.float64,
    )
    winding = projected_winding_numbers(curve, centers)
    assert winding.tolist() == [1, 0]
