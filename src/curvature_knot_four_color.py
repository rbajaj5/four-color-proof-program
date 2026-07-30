"""Curvature-weighted planar dissections of coarse-grained magnetic knots."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
import math
from typing import Any


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyTorch is required for knot dissection") from exc
    return torch


def _unique_closed_curve(curve: Any) -> Any:
    torch = _torch()
    if curve.ndim != 2 or curve.shape[1] != 3 or curve.shape[0] < 17:
        raise ValueError("curve must have shape [samples + 1, 3]")
    if curve.is_complex():
        raise ValueError("curve must be real")
    if float(torch.linalg.vector_norm(curve[-1] - curve[0]).item()) > 1e-7:
        raise ValueError("curve must repeat its first point")
    return curve[:-1]


def fixed_projection_matrix(*, device: Any, dtype: Any) -> Any:
    """Return the fixed generic projection used by the Maxwell fixtures."""

    torch = _torch()
    azimuth = math.radians(23.0)
    elevation = math.radians(31.0)
    rotate_z = torch.tensor(
        (
            (math.cos(azimuth), -math.sin(azimuth), 0.0),
            (math.sin(azimuth), math.cos(azimuth), 0.0),
            (0.0, 0.0, 1.0),
        ),
        device=device,
        dtype=dtype,
    )
    rotate_x = torch.tensor(
        (
            (1.0, 0.0, 0.0),
            (0.0, math.cos(elevation), -math.sin(elevation)),
            (0.0, math.sin(elevation), math.cos(elevation)),
        ),
        device=device,
        dtype=dtype,
    )
    return rotate_x @ rotate_z


def projected_curve_and_curvature_weights(curve: Any) -> dict[str, Any]:
    """Project a closed curve and return robust integrated-curvature weights."""

    torch = _torch()
    unique = _unique_closed_curve(curve)
    rotation = fixed_projection_matrix(
        device=unique.device,
        dtype=unique.dtype,
    )
    rotated = unique @ rotation.T
    projected = rotated[:, :2]
    projected -= projected.mean(dim=0)
    scale = projected.abs().max().clamp_min(torch.finfo(unique.dtype).eps)
    projected = 0.78 * projected / scale

    previous_edge = unique - torch.roll(unique, shifts=1, dims=0)
    next_edge = torch.roll(unique, shifts=-1, dims=0) - unique
    previous_unit = previous_edge / torch.linalg.vector_norm(
        previous_edge,
        dim=-1,
        keepdim=True,
    ).clamp_min(torch.finfo(unique.dtype).eps)
    next_unit = next_edge / torch.linalg.vector_norm(
        next_edge,
        dim=-1,
        keepdim=True,
    ).clamp_min(torch.finfo(unique.dtype).eps)
    bending_mass = torch.acos(
        torch.sum(previous_unit * next_unit, dim=-1).clamp(-1.0, 1.0)
    )

    projected_previous = projected - torch.roll(projected, shifts=1, dims=0)
    projected_next = torch.roll(projected, shifts=-1, dims=0) - projected
    cross = (
        projected_previous[:, 0] * projected_next[:, 1]
        - projected_previous[:, 1] * projected_next[:, 0]
    )
    dot = torch.sum(projected_previous * projected_next, dim=-1)
    signed_turning = torch.atan2(cross, dot)
    return {
        "projected": projected,
        "curvature_density": bending_mass,
        "signed_turning": signed_turning,
    }


def curvature_potential_surface(
    curve: Any,
    grid_size: int,
    bandwidth: float,
    channel: str,
    *,
    point_chunk_size: int = 256,
) -> Any:
    """Deposit curve curvature into a normalized scalar field on the plane."""

    torch = _torch()
    if grid_size < 5:
        raise ValueError("grid_size must be at least five")
    if bandwidth <= 0.0:
        raise ValueError("bandwidth must be positive")
    if point_chunk_size < 1:
        raise ValueError("point_chunk_size must be positive")
    geometry = projected_curve_and_curvature_weights(curve)
    if channel not in ("curvature_density", "signed_turning"):
        raise ValueError("unknown curvature channel")
    projected = geometry["projected"]
    weights = geometry[channel]
    axis = torch.linspace(
        -1.0,
        1.0,
        grid_size,
        device=projected.device,
        dtype=projected.dtype,
    )
    first, second = torch.meshgrid(axis, axis, indexing="ij")
    query = torch.stack((second, first), dim=-1).reshape(-1, 2)
    potential = torch.zeros(
        query.shape[0],
        device=projected.device,
        dtype=projected.dtype,
    )
    denominator = 2.0 * bandwidth * bandwidth
    for start in range(0, projected.shape[0], point_chunk_size):
        stop = min(start + point_chunk_size, projected.shape[0])
        displacement = query[:, None, :] - projected[None, start:stop, :]
        kernel = torch.exp(-displacement.square().sum(dim=-1) / denominator)
        potential += kernel @ weights[start:stop]
    potential = potential.reshape(grid_size, grid_size)
    potential -= potential.mean()
    potential /= potential.std().clamp_min(torch.finfo(potential.dtype).eps)
    return potential


def compactified_triangulation_faces(diagonals: Any) -> list[tuple[int, int, int]]:
    """Return all triangular faces, including the compactified exterior."""

    if diagonals.ndim != 2 or diagonals.shape[0] != diagonals.shape[1]:
        raise ValueError("diagonals must have shape [n-1, n-1]")
    grid_size = diagonals.shape[0] + 1
    exterior = grid_size * grid_size

    def index(row: int, column: int) -> int:
        return row * grid_size + column

    faces: list[tuple[int, int, int]] = []
    for row in range(grid_size - 1):
        for column in range(grid_size - 1):
            northwest = index(row, column)
            northeast = index(row, column + 1)
            southwest = index(row + 1, column)
            southeast = index(row + 1, column + 1)
            if bool(diagonals[row, column]):
                faces.extend(
                    (
                        (northwest, northeast, southeast),
                        (northwest, southeast, southwest),
                    )
                )
            else:
                faces.extend(
                    (
                        (northwest, northeast, southwest),
                        (northeast, southeast, southwest),
                    )
                )
    boundary = (
        [index(0, column) for column in range(grid_size)]
        + [index(row, grid_size - 1) for row in range(1, grid_size)]
        + [
            index(grid_size - 1, column)
            for column in range(grid_size - 2, -1, -1)
        ]
        + [index(row, 0) for row in range(grid_size - 2, 0, -1)]
    )
    for first, second in zip(boundary, boundary[1:] + boundary[:1], strict=True):
        faces.append((exterior, first, second))
    return faces


def klein_flux_certificate(
    faces: Sequence[tuple[int, int, int]],
    colors: Sequence[int],
) -> dict[str, Any]:
    """Interpret a four-coloring as a conserved Klein-four dual flow."""

    if any(color not in (0, 1, 2, 3) for color in colors):
        raise ValueError("colors must be elements of Z2 x Z2 encoded by 0..3")
    label_counts: Counter[int] = Counter()
    nonzero = True
    conserved = True
    for first, second, third in faces:
        labels = (
            colors[first] ^ colors[second],
            colors[second] ^ colors[third],
            colors[third] ^ colors[first],
        )
        nonzero &= all(label != 0 for label in labels)
        conserved &= labels[0] ^ labels[1] ^ labels[2] == 0
        label_counts.update(labels)
    total = sum(label_counts.values())
    return {
        "face_count": len(faces),
        "all_edge_fluxes_nonzero": nonzero,
        "all_dual_vertices_conserved": conserved,
        "flux_1_fraction": label_counts[1] / total if total else 0.0,
        "flux_2_fraction": label_counts[2] / total if total else 0.0,
        "flux_3_fraction": label_counts[3] / total if total else 0.0,
    }
