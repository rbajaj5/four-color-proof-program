import math

import pytest


torch = pytest.importorskip("torch")

from src.hopf_fibration_gate import (
    chern_number_midpoint,
    gauss_linking_number,
    hopf_fiber,
    hopf_map,
    phase_rotate,
    projective_projector,
    projector_distance_residual,
    stereographic_s3,
)


def test_projector_is_phase_invariant_and_idempotent() -> None:
    generator = torch.Generator().manual_seed(20260801)
    vectors = torch.complex(
        torch.randn(32, 5, generator=generator, dtype=torch.float64),
        torch.randn(32, 5, generator=generator, dtype=torch.float64),
    )
    phases = torch.rand(32, generator=generator, dtype=torch.float64)
    original = projective_projector(vectors)
    rotated = projective_projector(phase_rotate(vectors, phases))
    assert torch.max(torch.abs(original - rotated)) < 1e-12
    assert torch.max(torch.abs(original @ original - original)) < 1e-12
    assert torch.max(torch.abs(original.diagonal(dim1=-2, dim2=-1).sum(-1) - 1)) < 1e-12


def test_hopf_map_lands_on_sphere_and_is_phase_invariant() -> None:
    generator = torch.Generator().manual_seed(20260801)
    vectors = torch.complex(
        torch.randn(64, 2, generator=generator, dtype=torch.float64),
        torch.randn(64, 2, generator=generator, dtype=torch.float64),
    )
    phases = torch.rand(64, generator=generator, dtype=torch.float64)
    image = hopf_map(vectors)
    rotated = hopf_map(phase_rotate(vectors, phases))
    assert torch.max(torch.abs(torch.linalg.vector_norm(image, dim=-1) - 1)) < 1e-12
    assert torch.max(torch.abs(image - rotated)) < 1e-12


def test_projector_distance_identity() -> None:
    left = torch.tensor([[1.0 + 0j, 1.0j]], dtype=torch.complex128)
    right = torch.tensor([[1.0 - 1.0j, 2.0 + 0j]], dtype=torch.complex128)
    assert torch.max(torch.abs(projector_distance_residual(left, right))) < 1e-12


def test_first_chern_number_converges_to_one() -> None:
    coarse = chern_number_midpoint(32, device=torch.device("cpu"))
    fine = chern_number_midpoint(256, device=torch.device("cpu"))
    assert abs(fine - 1.0) < abs(coarse - 1.0)
    assert abs(fine - 1.0) < 1e-5


def test_distinct_hopf_fibers_link_once() -> None:
    first = stereographic_s3(
        hopf_fiber(math.pi / 2.0, 0.0, 512, device=torch.device("cpu"))
    )
    second = stereographic_s3(
        hopf_fiber(math.pi / 2.0, math.pi / 2.0, 512, device=torch.device("cpu"))
    )
    linking = gauss_linking_number(first, second)
    assert abs(abs(linking) - 1.0) < 5e-4
