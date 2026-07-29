from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from src.magnetic_braid_mhd import (
    braided_field_and_current,
    field_direction_spherical_path_length,
    finite_difference_divergence,
    integrate_line_tied_field_lines,
    integrated_parallel_current,
    mapping_stretch_diagnostics,
    neighbor_pairs,
    pairwise_winding,
    seed_grid,
    spherical_path_length,
    squashing_factor_from_mapping,
)


def test_braided_field_is_numerically_divergence_free() -> None:
    points = torch.tensor(
        [
            [
                (0.2, 0.1, -3.0),
                (-0.8, 0.4, 0.0),
                (1.2, -0.3, 3.0),
            ]
        ],
        dtype=torch.float64,
    )
    divergence = finite_difference_divergence(
        points,
        cycles=1,
        twist=1.0,
        guide_field=1.0,
        center_offset=1.0,
        diffusion_time=0.3,
        step=1e-5,
    )
    assert float(divergence.abs().max()) < 1e-9


def test_free_diffusion_reduces_ring_energy_and_current() -> None:
    points = torch.tensor(
        [
            [
                (1.5, 0.4, -4.0),
                (-1.5, 0.4, 4.0),
                (0.5, -0.6, -3.0),
            ],
            [
                (1.5, 0.4, -4.0),
                (-1.5, 0.4, 4.0),
                (0.5, -0.6, -3.0),
            ],
        ],
        dtype=torch.float64,
    )
    magnetic, current = braided_field_and_current(
        points,
        cycles=1,
        twist=torch.tensor((1.0, 1.0), dtype=torch.float64),
        guide_field=1.0,
        center_offset=1.0,
        diffusion_time=torch.tensor((0.0, 1.0), dtype=torch.float64),
    )
    transverse_energy = magnetic[..., :2].square().sum(dim=-1).mean(dim=-1)
    current_energy = current.square().sum(dim=-1).mean(dim=-1)
    assert transverse_energy[1] < transverse_energy[0]
    assert current_energy[1] < current_energy[0]


def test_zero_twist_field_lines_remain_vertical() -> None:
    seeds = seed_grid(3, 1.0, dtype=torch.float64)
    trajectory = integrate_line_tied_field_lines(
        seeds,
        cycles=1,
        twist=0.0,
        guide_field=1.0,
        center_offset=1.0,
        steps_per_cycle=16,
    )
    assert torch.allclose(
        trajectory[:, -1, :, :2],
        seeds.unsqueeze(0),
        atol=1e-12,
        rtol=0.0,
    )
    parallel_current = integrated_parallel_current(
        trajectory,
        cycles=1,
        twist=0.0,
        guide_field=1.0,
        center_offset=1.0,
    )
    assert parallel_current.shape == (1, seeds.shape[0])
    assert torch.count_nonzero(parallel_current) == 0
    direction_length = field_direction_spherical_path_length(
        trajectory,
        cycles=1,
        twist=0.0,
        guide_field=1.0,
        center_offset=1.0,
    )
    assert direction_length.shape == (1, seeds.shape[0])
    assert torch.count_nonzero(direction_length) == 0


def test_identity_mapping_has_minimal_squashing_factor() -> None:
    seeds = seed_grid(5, 1.5, dtype=torch.float64)
    mapping = seeds.unsqueeze(0)
    squashing = squashing_factor_from_mapping(
        mapping,
        grid_size=5,
        extent=1.5,
    )
    assert torch.allclose(squashing, torch.full_like(squashing, 2.0))
    constrained_squashing = squashing_factor_from_mapping(
        mapping,
        grid_size=5,
        extent=1.5,
        normal_field_ratio=1.0,
    )
    assert torch.equal(constrained_squashing, squashing)
    stretch = mapping_stretch_diagnostics(
        mapping,
        grid_size=5,
        extent=1.5,
        axial_length=2.0,
    )
    assert float(stretch["maximum_mapping_stretch"][0]) == pytest.approx(1.0)
    assert float(stretch["maximum_finite_length_lyapunov"][0]) == pytest.approx(
        0.0,
        abs=1e-14,
    )
    assert float(
        stretch["maximum_area_preservation_residual"][0]
    ) == pytest.approx(0.0, abs=1e-14)


def test_pairwise_winding_recovers_one_full_turn() -> None:
    theta = torch.linspace(0.0, 2.0 * torch.pi, 65, dtype=torch.float64)
    first = torch.zeros((65, 3), dtype=torch.float64)
    second = torch.stack(
        (torch.cos(theta), torch.sin(theta), theta),
        dim=-1,
    )
    first[:, 2] = theta
    trajectories = torch.stack((first, second), dim=1).unsqueeze(0)
    pairs = torch.tensor(((0, 1),), dtype=torch.long)
    winding = pairwise_winding(trajectories, pairs, coarse_factor=4)
    assert float(winding[0, 0]) == pytest.approx(1.0, abs=1e-12)
    assert neighbor_pairs(3).shape == (12, 2)


def test_spherical_direction_path_recovers_quarter_turn() -> None:
    theta = torch.linspace(0.0, torch.pi / 2.0, 65, dtype=torch.float64)
    directions = torch.stack(
        (torch.cos(theta), torch.sin(theta), torch.zeros_like(theta)),
        dim=-1,
    ).reshape(1, 65, 1, 3)
    length = spherical_path_length(directions)
    coarse_length = spherical_path_length(directions, coarse_factor=8)
    assert float(length[0, 0]) == pytest.approx(torch.pi / 2.0, abs=1e-10)
    assert float(coarse_length[0, 0]) == pytest.approx(
        torch.pi / 2.0,
        abs=1e-12,
    )
