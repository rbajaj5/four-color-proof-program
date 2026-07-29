"""Quasiperiodic convex-cell coarse graining for closed space curves."""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np


PHI = (1.0 + math.sqrt(5.0)) / 2.0
PERMUTATIONS = tuple(itertools.permutations(range(3)))


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyTorch is required for polyhedral RG") from exc
    return torch


def fibonacci_word(minimum_length: int) -> str:
    """Return a prefix of the fixed point of ``L -> LS, S -> L``."""

    if minimum_length < 1:
        raise ValueError("minimum_length must be positive")
    word = "L"
    while len(word) < minimum_length:
        word = "".join("LS" if letter == "L" else "L" for letter in word)
    return word[:minimum_length]


def fibonacci_axis(
    tile_count: int,
    *,
    scale: float,
    phase: int = 0,
) -> tuple[np.ndarray, str]:
    """Construct centered coordinates for a finite Fibonacci tiling patch."""

    if tile_count < 8:
        raise ValueError("tile_count must be at least 8")
    if scale <= 0:
        raise ValueError("scale must be positive")
    source = fibonacci_word(tile_count + abs(phase) + 2)
    start = phase % max(1, len(source) - tile_count + 1)
    word = source[start : start + tile_count]
    if len(word) != tile_count:
        word = (word + fibonacci_word(tile_count))[:tile_count]
    lengths = np.array(
        [PHI if letter == "L" else 1.0 for letter in word],
        dtype=np.float64,
    )
    coordinates = np.concatenate(([0.0], np.cumsum(scale * lengths)))
    coordinates -= 0.5 * (coordinates[0] + coordinates[-1])
    return coordinates, word


def golden_shear_matrix(
    *,
    device: Any = None,
    dtype: Any = None,
) -> Any:
    """Return a fixed orientation-preserving golden-ratio affine shear."""

    torch = _torch()
    if dtype is None:
        dtype = torch.float64
    return torch.tensor(
        (
            (1.0, 0.22 / PHI, 0.08),
            (0.11, 1.0, 0.19 / PHI),
            (0.16 / PHI, 0.07, 1.0),
        ),
        device=device,
        dtype=dtype,
    )


def curve_point_curvatures(curve: Any) -> Any:
    """Return parameterization-invariant discrete curvature at each sample."""

    torch = _torch()
    if curve.ndim != 2 or curve.shape[1] != 3 or curve.shape[0] < 17:
        raise ValueError("curve must have shape [samples + 1, 3]")
    points = curve[:-1]
    first = (
        torch.roll(points, shifts=-1, dims=0)
        - torch.roll(points, shifts=1, dims=0)
    )
    second = (
        torch.roll(points, shifts=-1, dims=0)
        - 2.0 * points
        + torch.roll(points, shifts=1, dims=0)
    )
    numerator = torch.linalg.vector_norm(
        torch.cross(first, second, dim=-1),
        dim=-1,
    )
    denominator = torch.linalg.vector_norm(
        first,
        dim=-1,
    ).pow(3).clamp_min(torch.finfo(points.dtype).eps)
    return 4.0 * numerator / denominator


def specular_reflection(velocity: Any, normal: Any) -> Any:
    """Reflect vectors across tangent hyperplanes with unit normals."""

    torch = _torch()
    if velocity.shape[-1] != 3 or normal.shape[-1] != 3:
        raise ValueError("velocity and normal must end in dimension 3")
    normal_norm = torch.linalg.vector_norm(normal, dim=-1, keepdim=True)
    if bool(torch.any(normal_norm <= torch.finfo(velocity.dtype).eps)):
        raise ValueError("normal vectors must be nonzero")
    unit_normal = normal / normal_norm
    normal_component = torch.sum(
        velocity * unit_normal,
        dim=-1,
        keepdim=True,
    )
    return velocity - 2.0 * normal_component * unit_normal


def space_form_coupling_force(
    curvature: Any,
    separation: Any,
    *,
    speed: Any = 1.0,
    mass: Any = 1.0,
) -> Any:
    """Return the Coulton-Galperin force for constant-curvature space forms.

    Positive values are outward forces needed to hold paths apart on a
    sphere. Negative values are inward forces needed to hold paths together
    in the hyperbolic plane. The zero-curvature limit is zero.
    """

    torch = _torch()
    reference = next(
        (
            value
            for value in (curvature, separation, speed, mass)
            if isinstance(value, torch.Tensor)
        ),
        None,
    )
    device = reference.device if reference is not None else None
    dtype = reference.dtype if reference is not None else torch.float64
    curvature_tensor = torch.as_tensor(
        curvature,
        device=device,
        dtype=dtype,
    )
    separation_tensor = torch.as_tensor(
        separation,
        device=device,
        dtype=dtype,
    )
    speed_tensor = torch.as_tensor(speed, device=device, dtype=dtype)
    mass_tensor = torch.as_tensor(mass, device=device, dtype=dtype)
    if bool(torch.any(separation_tensor < 0)):
        raise ValueError("separation must be nonnegative")
    if bool(torch.any(speed_tensor < 0)) or bool(torch.any(mass_tensor < 0)):
        raise ValueError("speed and mass must be nonnegative")
    absolute_root = torch.sqrt(torch.abs(curvature_tensor))
    argument = 0.5 * absolute_root * separation_tensor
    if bool(
        torch.any(
            (curvature_tensor > 0) & (argument >= 0.5 * torch.pi)
        )
    ):
        raise ValueError(
            "positive-curvature separation reaches the antipodal singularity"
        )
    common = 2.0 * mass_tensor * speed_tensor.pow(2) * absolute_root
    positive_force = common * torch.tan(argument)
    negative_force = -common * torch.tanh(argument)
    return torch.where(
        curvature_tensor > 0,
        positive_force,
        torch.where(
            curvature_tensor < 0,
            negative_force,
            torch.zeros_like(positive_force),
        ),
    )


def polyhedral_force_susceptibility(
    itinerary: dict[str, Any],
    *,
    separation_fraction: float = 0.5,
) -> dict[str, float]:
    """Evaluate a unit-``m v^2`` space-form response envelope per cell.

    Each sampled tetrahedron is assigned radius equal to its Euclidean
    diameter and a virtual pair separation equal to
    ``separation_fraction * diameter``. This is a normalized local-chart
    diagnostic, not a physical force exerted by a Euclidean tiling.
    """

    torch = _torch()
    if not 0.0 < separation_fraction < math.pi:
        raise ValueError("separation_fraction must lie in (0, pi)")
    vertices = itinerary["sample_tetrahedron_vertices"]
    pairwise = torch.linalg.vector_norm(
        vertices[:, :, None, :] - vertices[:, None, :, :],
        dim=-1,
    )
    diameter = pairwise.amax(dim=(-2, -1)).clamp_min(
        torch.finfo(vertices.dtype).eps
    )
    curvature_magnitude = diameter.reciprocal().pow(2)
    separation = separation_fraction * diameter
    positive = space_form_coupling_force(
        curvature_magnitude,
        separation,
    )
    negative = space_form_coupling_force(
        -curvature_magnitude,
        separation,
    )
    negative_magnitude_mean = torch.mean(torch.abs(negative))
    return {
        "mean_cell_diameter": float(torch.mean(diameter).item()),
        "positive_space_form_force_mean_per_mv2": float(
            torch.mean(positive).item()
        ),
        "positive_space_form_force_max_per_mv2": float(
            torch.max(positive).item()
        ),
        "negative_space_form_force_mean_per_mv2": float(
            torch.mean(negative).item()
        ),
        "negative_space_form_force_min_per_mv2": float(
            torch.min(negative).item()
        ),
        "space_form_force_asymmetry_ratio": float(
            (torch.mean(positive) / negative_magnitude_mean).item()
        ),
    }


def _permutation_codes(fractions: Any) -> Any:
    torch = _torch()
    orders = torch.argsort(fractions, dim=-1, descending=True)
    codes = torch.full(
        (fractions.shape[0],),
        -1,
        device=fractions.device,
        dtype=torch.long,
    )
    for code, permutation in enumerate(PERMUTATIONS):
        target = torch.tensor(
            permutation,
            device=fractions.device,
            dtype=torch.long,
        )
        codes[torch.all(orders == target, dim=-1)] = code
    if bool(torch.any(codes < 0)):
        raise AssertionError("every point must receive a simplex permutation")
    return codes


def quasiperiodic_tetrahedral_itinerary(
    curve: Any,
    *,
    inflation_level: int,
    base_scale: float = 0.18,
    tile_count: int = 55,
    phases: tuple[int, int, int] = (0, 7, 13),
) -> dict[str, Any]:
    """Map curve samples into a sheared Fibonacci tetrahedral tiling patch."""

    torch = _torch()
    if curve.ndim != 2 or curve.shape[1] != 3:
        raise ValueError("curve must have shape [samples + 1, 3]")
    scale = base_scale * PHI**inflation_level
    axes_and_words = [
        fibonacci_axis(tile_count, scale=scale, phase=phase)
        for phase in phases
    ]
    axes = [
        torch.as_tensor(
            coordinates,
            device=curve.device,
            dtype=curve.dtype,
        )
        for coordinates, _ in axes_and_words
    ]
    words = [word for _, word in axes_and_words]
    shear = golden_shear_matrix(device=curve.device, dtype=curve.dtype)
    unsheared = torch.linalg.solve(shear, curve[:-1].T).T
    cell_indices = []
    fractions = []
    lower_boundaries = []
    side_lengths = []
    inside = torch.ones(
        unsheared.shape[0],
        device=curve.device,
        dtype=torch.bool,
    )
    for coordinate, boundaries in zip(
        unsheared.T,
        axes,
        strict=True,
    ):
        index = (
            torch.searchsorted(
                boundaries,
                coordinate.contiguous(),
                right=True,
            )
            - 1
        )
        valid = (index >= 0) & (index < tile_count)
        inside &= valid
        safe_index = index.clamp(0, tile_count - 1)
        lower = boundaries[safe_index]
        upper = boundaries[safe_index + 1]
        cell_indices.append(safe_index)
        fractions.append((coordinate - lower) / (upper - lower))
        lower_boundaries.append(lower)
        side_lengths.append(upper - lower)
    if not bool(torch.all(inside)):
        raise ValueError("curve escaped the finite quasiperiodic tiling patch")
    index_tensor = torch.stack(cell_indices, dim=-1)
    fraction_tensor = torch.stack(fractions, dim=-1)
    simplex_codes = _permutation_codes(fraction_tensor)
    box_ids = (
        (index_tensor[:, 0] * tile_count + index_tensor[:, 1])
        * tile_count
        + index_tensor[:, 2]
    )
    tetrahedron_ids = box_ids * len(PERMUTATIONS) + simplex_codes
    tile_type_codes = torch.zeros_like(simplex_codes)
    for axis_index, (indices, word) in enumerate(
        zip(cell_indices, words, strict=True)
    ):
        long_flags = torch.tensor(
            [letter == "L" for letter in word],
            device=curve.device,
            dtype=torch.long,
        )[indices]
        tile_type_codes += long_flags << axis_index
    type_simplex_codes = tile_type_codes * len(PERMUTATIONS) + simplex_codes
    centers_unsheared = torch.stack(
        [
            0.5 * (boundaries[indices] + boundaries[indices + 1])
            for boundaries, indices in zip(
                axes,
                cell_indices,
                strict=True,
            )
        ],
        dim=-1,
    )
    centers = centers_unsheared @ shear.T
    lower_tensor = torch.stack(lower_boundaries, dim=-1)
    side_tensor = torch.stack(side_lengths, dim=-1)
    permutation_tensor = torch.tensor(
        PERMUTATIONS,
        device=curve.device,
        dtype=torch.long,
    )[simplex_codes]
    current = lower_tensor
    tetrahedron_vertices_unsheared = [current]
    for step in range(3):
        axis = permutation_tensor[:, step]
        increment = torch.nn.functional.one_hot(
            axis,
            num_classes=3,
        ).to(curve.dtype) * side_tensor
        current = current + increment
        tetrahedron_vertices_unsheared.append(current)
    sample_tetrahedron_vertices = torch.stack(
        tetrahedron_vertices_unsheared,
        dim=1,
    ) @ shear.T
    return {
        "inflation_level": inflation_level,
        "cell_scale": scale,
        "cell_indices": index_tensor,
        "simplex_codes": simplex_codes,
        "tetrahedron_ids": tetrahedron_ids,
        "type_simplex_codes": type_simplex_codes,
        "cell_centers": centers,
        "tile_type_codes": tile_type_codes,
        "sample_tetrahedron_vertices": sample_tetrahedron_vertices,
    }


def itinerary_metrics(
    itinerary: dict[str, Any],
    point_curvatures: Any,
) -> dict[str, float | int]:
    """Summarize occupancy, transitions, entropy, and curvature localization."""

    torch = _torch()
    identifiers = itinerary["tetrahedron_ids"]
    types = itinerary["type_simplex_codes"]
    if identifiers.shape[0] != point_curvatures.shape[0]:
        raise ValueError("curvature and itinerary sample counts differ")
    next_identifiers = torch.roll(identifiers, shifts=-1, dims=0)
    transitions = identifiers != next_identifiers
    transition_count = int(torch.sum(transitions).item())
    unique_identifiers, inverse, counts = torch.unique(
        identifiers,
        return_inverse=True,
        return_counts=True,
    )
    probabilities = counts.double() / counts.sum()
    occupancy_entropy = -torch.sum(
        probabilities * torch.log(probabilities)
    )
    curvature_load = torch.zeros(
        unique_identifiers.shape[0],
        device=identifiers.device,
        dtype=point_curvatures.dtype,
    )
    curvature_load.scatter_add_(0, inverse, point_curvatures)
    total_curvature = curvature_load.sum().clamp_min(
        torch.finfo(point_curvatures.dtype).eps
    )
    curvature_concentration = curvature_load.max() / total_curvature
    transition_pairs = torch.stack(
        (identifiers, next_identifiers),
        dim=-1,
    )[transitions]
    unique_transition_count = (
        int(torch.unique(transition_pairs, dim=0).shape[0])
        if transition_count
        else 0
    )
    type_count = int(torch.unique(types).shape[0])
    return {
        "occupied_tetrahedra": int(unique_identifiers.shape[0]),
        "itinerary_transitions": transition_count,
        "unique_transition_pairs": unique_transition_count,
        "occupied_type_simplex_classes": type_count,
        "occupancy_entropy": float(occupancy_entropy.item()),
        "maximum_cell_curvature_fraction": float(
            curvature_concentration.item()
        ),
    }


def tetrahedron_vertices(
    side_lengths: tuple[float, float, float],
    simplex_code: int,
) -> np.ndarray:
    """Return one Freudenthal tetrahedron after the golden affine shear."""

    if simplex_code < 0 or simplex_code >= len(PERMUTATIONS):
        raise ValueError("simplex_code must lie in 0..5")
    lengths = np.asarray(side_lengths, dtype=np.float64)
    if np.any(lengths <= 0):
        raise ValueError("side lengths must be positive")
    permutation = PERMUTATIONS[simplex_code]
    vertices = [np.zeros(3, dtype=np.float64)]
    current = vertices[0].copy()
    for axis in permutation:
        current = current.copy()
        current[axis] += lengths[axis]
        vertices.append(current)
    shear = np.array(
        (
            (1.0, 0.22 / PHI, 0.08),
            (0.11, 1.0, 0.19 / PHI),
            (0.16 / PHI, 0.07, 1.0),
        ),
        dtype=np.float64,
    )
    return np.asarray(vertices) @ shear.T


def tetrahedron_vertex_defects(vertices: np.ndarray) -> np.ndarray:
    """Compute the four surface angle defects of a convex tetrahedron."""

    if vertices.shape != (4, 3):
        raise ValueError("vertices must have shape [4, 3]")
    defects = []
    for vertex_index in range(4):
        edges = np.delete(vertices, vertex_index, axis=0) - vertices[
            vertex_index
        ]
        angle_sum = 0.0
        for first, second in ((0, 1), (0, 2), (1, 2)):
            cosine = np.dot(edges[first], edges[second]) / (
                np.linalg.norm(edges[first])
                * np.linalg.norm(edges[second])
            )
            angle_sum += math.acos(float(np.clip(cosine, -1.0, 1.0)))
        defects.append(2.0 * math.pi - angle_sum)
    return np.asarray(defects)


def bounded_defect_relation_residual(
    defects: np.ndarray,
    *,
    coefficient_bound: int = 3,
) -> float:
    """Search for a small proper-subset integer relation with ``pi``."""

    if defects.shape != (4,):
        raise ValueError("defects must contain four values")
    if coefficient_bound < 1:
        raise ValueError("coefficient_bound must be positive")
    values = range(-coefficient_bound, coefficient_bound + 1)
    best = math.inf
    for coefficients in itertools.product(values, repeat=4):
        if all(coefficient != 0 for coefficient in coefficients):
            continue
        for pi_coefficient in values:
            if pi_coefficient == 0 and not any(coefficients):
                continue
            residual = abs(
                pi_coefficient * math.pi
                + float(np.dot(coefficients, defects))
            )
            best = min(best, residual / math.pi)
    return best


def defect_library(coefficient_bound: int = 3) -> dict[int, dict[str, Any]]:
    """Build defect spectra for the 48 tile-type/simplex classes."""

    result: dict[int, dict[str, Any]] = {}
    for tile_type in range(8):
        lengths = tuple(
            PHI if tile_type & (1 << axis) else 1.0
            for axis in range(3)
        )
        for simplex_code in range(len(PERMUTATIONS)):
            code = tile_type * len(PERMUTATIONS) + simplex_code
            vertices = tetrahedron_vertices(lengths, simplex_code)
            defects = tetrahedron_vertex_defects(vertices)
            result[code] = {
                "tile_type": tile_type,
                "simplex_code": simplex_code,
                "defects": defects,
                "defect_sum": float(np.sum(defects)),
                "bounded_relation_residual": (
                    bounded_defect_relation_residual(
                        defects,
                        coefficient_bound=coefficient_bound,
                    )
                ),
            }
    return result


def tangent_holonomy(curve: Any) -> dict[str, float]:
    """Parallel transport a normal around the polygonal tangent indicatrix."""

    points = curve.detach().cpu().numpy()[:-1]
    edges = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(edges, axis=1)
    tangents = edges / lengths[:, None]
    initial_tangent = tangents[0]
    reference_axis = np.eye(3)[np.argmin(np.abs(initial_tangent))]
    initial_normal = reference_axis - np.dot(
        reference_axis,
        initial_tangent,
    ) * initial_tangent
    initial_normal /= np.linalg.norm(initial_normal)
    normal = initial_normal.copy()
    total_turning = 0.0
    for index in range(len(tangents)):
        first = tangents[index]
        second = tangents[(index + 1) % len(tangents)]
        cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
        angle = math.acos(cosine)
        total_turning += angle
        axis = np.cross(first, second)
        sine = np.linalg.norm(axis)
        if sine < 1e-14:
            continue
        axis /= sine
        normal = (
            normal * cosine
            + np.cross(axis, normal) * sine
            + axis * np.dot(axis, normal) * (1.0 - cosine)
        )
    normal -= np.dot(normal, initial_tangent) * initial_tangent
    normal /= np.linalg.norm(normal)
    holonomy = math.atan2(
        np.dot(initial_tangent, np.cross(initial_normal, normal)),
        np.dot(initial_normal, normal),
    )
    return {
        "total_polygonal_turning": total_turning,
        "tangent_holonomy_angle": holonomy,
    }


def projected_winding_numbers(
    curve: Any,
    centers: Any,
    *,
    projection_axes: tuple[int, int] = (0, 1),
) -> Any:
    """Return planar winding proxies around supplied cell centers."""

    torch = _torch()
    points = curve[:-1, list(projection_axes)]
    projected_centers = centers[:, list(projection_axes)]
    relative = points[:, None, :] - projected_centers[None, :, :]
    angles = torch.atan2(relative[..., 1], relative[..., 0])
    differences = torch.roll(angles, shifts=-1, dims=0) - angles
    differences = torch.atan2(torch.sin(differences), torch.cos(differences))
    winding = torch.sum(differences, dim=0) / (2.0 * torch.pi)
    return torch.round(winding).to(torch.long)
