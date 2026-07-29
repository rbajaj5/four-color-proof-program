from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from src.curvature_rg_flow import (
    effective_mode_count,
    first_nonzero_mode,
    leading_mode_rank,
    periodic_heat_coarse_grain,
    spectral_energies,
)
from src.maxwell_knot_fields import magnetic_core_curve


def test_heat_flow_preserves_closure() -> None:
    curve = magnetic_core_curve(2, 3, 256, dtype=torch.float64)
    smoothed = periodic_heat_coarse_grain(curve, 0.2)
    assert smoothed.shape == curve.shape
    assert torch.linalg.vector_norm(smoothed[-1] - smoothed[0]) < 1e-12


def test_heat_semigroup_without_normalization() -> None:
    curve = magnetic_core_curve(3, 4, 256, dtype=torch.float64)
    first = periodic_heat_coarse_grain(curve, 0.07)
    composed = periodic_heat_coarse_grain(first, 0.11)
    direct = periodic_heat_coarse_grain(curve, 0.18)
    assert torch.max(torch.abs(composed - direct)) < 1e-11


def test_spectral_energies_and_mode_count_decrease() -> None:
    curve = magnetic_core_curve(2, 3, 256, dtype=torch.float64)
    smoothed = periodic_heat_coarse_grain(curve, 0.1)
    initial = spectral_energies(curve)
    later = spectral_energies(smoothed)
    assert later["spectral_dirichlet_energy"] < initial[
        "spectral_dirichlet_energy"
    ]
    assert later["spectral_bending_energy"] < initial[
        "spectral_bending_energy"
    ]
    assert effective_mode_count(smoothed) < effective_mode_count(curve)


def test_leading_modes_distinguish_ellipse_and_multiple_cover_limits() -> None:
    trefoil = magnetic_core_curve(2, 3, 512, dtype=torch.float64)
    cinquefoil = magnetic_core_curve(2, 5, 512, dtype=torch.float64)
    assert first_nonzero_mode(trefoil) == 1
    assert first_nonzero_mode(cinquefoil) == 2
    assert leading_mode_rank(trefoil) == 2
    assert leading_mode_rank(cinquefoil) == 2
