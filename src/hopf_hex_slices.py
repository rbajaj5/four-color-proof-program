"""Planar Hex windows pulled back from the classical Hopf map."""

from __future__ import annotations

import math

import torch

from src.hopf_fibration_gate import hopf_map, phase_rotate
from src.maxwell_knot_fields import bateman_alpha_beta_gradients


def centered_hex_lattice(
    size: int,
    field_of_view: float,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    if size < 2:
        raise ValueError("size must be at least two")
    if field_of_view <= 0:
        raise ValueError("field_of_view must be positive")
    q, r = torch.meshgrid(
        torch.arange(size, device=device, dtype=dtype),
        torch.arange(size, device=device, dtype=dtype),
        indexing="ij",
    )
    coordinates = torch.stack(
        (q + 0.5 * r, 0.5 * math.sqrt(3.0) * r),
        dim=-1,
    )
    coordinates -= coordinates.mean(dim=(0, 1), keepdim=True)
    diameter = torch.linalg.vector_norm(coordinates, dim=-1).max()
    return coordinates * (0.5 * field_of_view / diameter)


def random_plane_frames(
    batch_size: int,
    *,
    device: torch.device,
    generator: torch.Generator,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    normals = torch.randn(
        batch_size,
        3,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    normals /= torch.linalg.vector_norm(normals, dim=-1, keepdim=True)
    reference = torch.zeros_like(normals)
    reference[:, 2] = 1.0
    near_pole = normals[:, 2].abs() > 0.9
    reference[near_pole] = torch.tensor(
        (1.0, 0.0, 0.0),
        device=device,
        dtype=dtype,
    )
    first = torch.cross(normals, reference, dim=-1)
    first /= torch.linalg.vector_norm(first, dim=-1, keepdim=True)
    second = torch.cross(normals, first, dim=-1)
    angles = 2.0 * math.pi * torch.rand(
        batch_size,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    cosine = torch.cos(angles).unsqueeze(-1)
    sine = torch.sin(angles).unsqueeze(-1)
    rotated_first = cosine * first + sine * second
    rotated_second = -sine * first + cosine * second
    return normals, rotated_first, rotated_second


def sample_hopf_hex_window(
    size: int,
    field_of_view: float,
    batch_size: int,
    *,
    center_extent: float,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return Hopf-map values and their common-phase-rotated controls."""

    if center_extent < 0:
        raise ValueError("center_extent must be nonnegative")
    lattice = centered_hex_lattice(
        size,
        field_of_view,
        device=device,
    )
    _, first, second = random_plane_frames(
        batch_size,
        device=device,
        generator=generator,
    )
    centers = center_extent * (
        2.0
        * torch.rand(
            batch_size,
            3,
            device=device,
            dtype=torch.float64,
            generator=generator,
        )
        - 1.0
    )
    points = (
        centers[:, None, None, :]
        + lattice[None, :, :, :1] * first[:, None, None, :]
        + lattice[None, :, :, 1:] * second[:, None, None, :]
    )
    alpha, beta, _, _ = bateman_alpha_beta_gradients(points)
    sphere_vectors = torch.stack((alpha, beta), dim=-1)
    image = hopf_map(sphere_vectors)
    phases = 2.0 * math.pi * torch.rand(
        batch_size,
        device=device,
        dtype=torch.float64,
        generator=generator,
    )
    phase_grid = phases[:, None, None].expand(batch_size, size, size)
    rotated = hopf_map(phase_rotate(sphere_vectors, phase_grid))
    phase_error = float((image - rotated).abs().max().item())
    return image, rotated, phase_error
