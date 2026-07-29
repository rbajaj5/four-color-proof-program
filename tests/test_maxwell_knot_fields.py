from __future__ import annotations

from math import gcd

import pytest


torch = pytest.importorskip("torch")

from src.maxwell_knot_fields import (
    all_magnetic_core_curves,
    bateman_alpha_beta_gradients,
    core_tangent_alignment,
    expected_core_component_count,
    magnetic_core_curve,
    maxwell_residuals,
)


@pytest.mark.parametrize("p,q", [(1, 1), (2, 3), (2, 5), (2, 2)])
def test_core_curve_closes_and_has_expected_count(p: int, q: int) -> None:
    curves = all_magnetic_core_curves(
        p,
        q,
        256,
        dtype=torch.float64,
    )
    assert len(curves) == 2 * gcd(p, q)
    assert len(curves) == expected_core_component_count(p, q)
    for curve in curves:
        assert curve.shape == (257, 3)
        assert torch.isfinite(curve).all()
        assert torch.linalg.vector_norm(curve[-1] - curve[0]) < 1e-10


def test_bateman_coordinates_lie_on_three_sphere_at_time_zero() -> None:
    generator = torch.Generator().manual_seed(20260729)
    points = torch.randn(128, 3, generator=generator, dtype=torch.float64)
    alpha, beta, _, _ = bateman_alpha_beta_gradients(points)
    sphere_norm = alpha.abs().square() + beta.abs().square()
    assert torch.max(torch.abs(sphere_norm - 1.0)) < 1e-12


def test_trefoil_core_is_tangent_to_magnetic_field() -> None:
    curve = magnetic_core_curve(2, 3, 512, dtype=torch.float64)
    alignment = core_tangent_alignment(curve, 2, 3)
    assert float(alignment.mean()) > 0.999
    assert float(alignment.min()) > 0.995


def test_non_coprime_core_components_are_distinct_and_tangent() -> None:
    curves = all_magnetic_core_curves(2, 2, 256, dtype=torch.float64)
    assert len(curves) == 4
    for curve in curves:
        alignment = core_tangent_alignment(curve, 2, 2)
        assert float(alignment.min()) > 0.995
    for left in range(len(curves)):
        for right in range(left + 1, len(curves)):
            pairwise = torch.cdist(curves[left][:-1], curves[right][:-1])
            assert float(pairwise.min()) > 0.1


def test_maxwell_residuals_are_small_on_cpu_fixture() -> None:
    generator = torch.Generator().manual_seed(20260729)
    points = (
        2.0
        * torch.rand(64, 3, generator=generator, dtype=torch.float64)
        - 1.0
    )
    residuals = maxwell_residuals(points, 2, 3, step=2e-5)
    assert residuals["relative_divergence_magnetic_mean"] < 1e-7
    assert residuals["relative_faraday_mean"] < 1e-7
    assert residuals["relative_ampere_mean"] < 1e-7
    assert residuals["null_dot_mean_abs"] < 1e-11
    assert residuals["null_norm_mean_abs"] < 1e-11
