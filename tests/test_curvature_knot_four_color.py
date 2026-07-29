from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from src.curvature_knot_four_color import (
    compactified_triangulation_faces,
    curvature_potential_surface,
    klein_flux_certificate,
    projected_curve_and_curvature_weights,
)
from src.curvature_rg_flow import periodic_heat_coarse_grain
from src.fractional_field_four_color import (
    coloring_is_proper,
    compactified_triangulation_adjacency,
    dsatur_coloring,
    mixed_curvature_diagonals,
)
from src.maxwell_knot_fields import magnetic_core_curve


def test_curvature_channels_are_finite_and_projection_is_bounded() -> None:
    curve = magnetic_core_curve(2, 3, 256, dtype=torch.float64)
    channels = projected_curve_and_curvature_weights(curve)
    assert channels["projected"].shape == (256, 2)
    assert float(channels["projected"].abs().max()) <= 0.7800001
    assert torch.isfinite(channels["curvature_density"]).all()
    assert torch.isfinite(channels["signed_turning"]).all()
    assert torch.all(channels["curvature_density"] >= 0)


@pytest.mark.parametrize("channel", ("curvature_density", "signed_turning"))
def test_curvature_potential_is_normalized(channel: str) -> None:
    curve = magnetic_core_curve(2, 3, 128, dtype=torch.float64)
    surface = curvature_potential_surface(curve, 17, 0.12, channel)
    assert surface.shape == (17, 17)
    assert torch.isfinite(surface).all()
    assert float(surface.mean().abs()) < 1e-12
    assert float((surface.std() - 1.0).abs()) < 1e-12


def test_klein_flux_certificate_is_nonzero_and_conserved() -> None:
    curve = magnetic_core_curve(2, 3, 128, dtype=torch.float64)
    surface = curvature_potential_surface(
        periodic_heat_coarse_grain(curve, 0.02, normalize=True),
        9,
        0.16,
        "curvature_density",
    )
    diagonals = mixed_curvature_diagonals(surface.unsqueeze(0))[0]
    adjacency = compactified_triangulation_adjacency(diagonals)
    colors = dsatur_coloring(adjacency, 4)
    assert coloring_is_proper(adjacency, colors)
    faces = compactified_triangulation_faces(diagonals)
    vertex_count = len(adjacency)
    assert len(faces) == 2 * vertex_count - 4
    certificate = klein_flux_certificate(faces, colors)
    assert certificate["all_edge_fluxes_nonzero"]
    assert certificate["all_dual_vertices_conserved"]
    assert sum(
        certificate[f"flux_{label}_fraction"] for label in (1, 2, 3)
    ) == pytest.approx(1.0)
