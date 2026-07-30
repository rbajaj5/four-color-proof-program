import pytest


torch = pytest.importorskip("torch")

from src.hopf_hex_slices import (
    centered_hex_lattice,
    random_plane_frames,
    sample_hopf_hex_window,
)


def test_centered_hex_lattice_has_declared_diameter() -> None:
    lattice = centered_hex_lattice(
        7,
        3.0,
        device=torch.device("cpu"),
    )
    assert lattice.shape == (7, 7, 2)
    assert torch.max(torch.abs(lattice.mean(dim=(0, 1)))) < 1e-12
    assert torch.linalg.vector_norm(lattice, dim=-1).max() == pytest.approx(1.5)


def test_random_plane_frames_are_orthonormal() -> None:
    generator = torch.Generator().manual_seed(20260802)
    normal, first, second = random_plane_frames(
        128,
        device=torch.device("cpu"),
        generator=generator,
    )
    for vectors in (normal, first, second):
        assert torch.max(
            torch.abs(torch.linalg.vector_norm(vectors, dim=-1) - 1.0)
        ) < 1e-12
    assert torch.max(torch.abs((normal * first).sum(-1))) < 1e-12
    assert torch.max(torch.abs((normal * second).sum(-1))) < 1e-12
    assert torch.max(torch.abs((first * second).sum(-1))) < 1e-12


def test_hopf_window_is_phase_invariant() -> None:
    generator = torch.Generator().manual_seed(20260802)
    image, rotated, phase_error = sample_hopf_hex_window(
        6,
        2.0,
        8,
        center_extent=1.0,
        device=torch.device("cpu"),
        generator=generator,
    )
    assert image.shape == (8, 6, 6, 3)
    assert torch.max(
        torch.abs(torch.linalg.vector_norm(image, dim=-1) - 1.0)
    ) < 1e-12
    assert phase_error < 2e-12
    assert torch.max(torch.abs(image - rotated)) < 2e-12
