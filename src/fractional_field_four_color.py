"""Fractional Gaussian surfaces and curvature-induced planar triangulations."""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any


def _torch() -> Any:
    import torch

    return torch


def fractional_gaussian_surfaces(
    batch_size: int,
    resolution: int,
    hurst: float,
    *,
    seed: int,
    device: Any = None,
    dtype: Any = None,
) -> Any:
    """Sample periodic cutoff ``FGF_s(R^2)`` surfaces with ``s = H + 1``."""

    torch = _torch()
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if resolution < 8:
        raise ValueError("resolution must be at least 8")
    if not 0.0 < hurst < 1.0:
        raise ValueError("hurst must lie strictly between zero and one")
    if dtype is None:
        dtype = torch.float32
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    white_noise = torch.randn(
        (batch_size, resolution, resolution),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    frequencies = torch.fft.fftfreq(
        resolution,
        d=1.0 / resolution,
        device=device,
        dtype=dtype,
    )
    first, second = torch.meshgrid(frequencies, frequencies, indexing="ij")
    radius = torch.sqrt(first.square() + second.square())
    spectral_order = hurst + 1.0
    multiplier = torch.where(
        radius > 0,
        radius.pow(-spectral_order),
        torch.zeros_like(radius),
    )
    coefficients = torch.fft.fft2(white_noise, norm="ortho") * multiplier
    surfaces = torch.fft.ifft2(coefficients, norm="ortho").real
    surfaces -= surfaces.mean(dim=(-2, -1), keepdim=True)
    surfaces /= surfaces.std(dim=(-2, -1), keepdim=True).clamp_min(
        torch.finfo(dtype).eps
    )
    return surfaces


def periodic_closed_window(surfaces: Any) -> Any:
    """Repeat the first row and column to close a periodic square window."""

    if surfaces.ndim != 3:
        raise ValueError("surfaces must have shape [batch, n, n]")
    if surfaces.shape[-2] != surfaces.shape[-1]:
        raise ValueError("surfaces must be square")
    row_closed = _torch().cat((surfaces, surfaces[:, :1, :]), dim=1)
    return _torch().cat((row_closed, row_closed[:, :, :1]), dim=2)


def subsample_closed_window(window: Any, stride: int) -> Any:
    """Subsample a closed window while retaining both boundary endpoints."""

    torch = _torch()
    if window.ndim != 3 or window.shape[-2] != window.shape[-1]:
        raise ValueError("window must have shape [batch, n, n]")
    if stride < 1:
        raise ValueError("stride must be positive")
    if (window.shape[-1] - 1) % stride != 0:
        raise ValueError("stride must divide the periodic resolution")
    indices = torch.arange(
        0,
        window.shape[-1],
        stride,
        device=window.device,
    )
    return window.index_select(1, indices).index_select(2, indices)


def mixed_curvature_diagonals(window: Any) -> Any:
    """Choose each square diagonal from the sign of mixed curvature.

    ``True`` selects the northwest-to-southeast diagonal joining
    ``(row, column)`` to ``(row+1, column+1)``.
    """

    if window.ndim != 3 or window.shape[-2] != window.shape[-1]:
        raise ValueError("window must have shape [batch, n, n]")
    mixed_curvature = (
        window[:, :-1, :-1]
        + window[:, 1:, 1:]
        - window[:, 1:, :-1]
        - window[:, :-1, 1:]
    )
    return mixed_curvature >= 0


def interior_parity_defects(diagonals: Any) -> Any:
    """Return odd-degree indicators at interior grid vertices.

    The four cells surrounding an interior vertex contribute one diagonal
    endpoint each. The grid contributes degree four, so the degree parity is
    exactly the XOR of the four diagonal choices.
    """

    if diagonals.ndim != 3 or diagonals.shape[-2] != diagonals.shape[-1]:
        raise ValueError("diagonals must have shape [batch, n-1, n-1]")
    if diagonals.shape[-1] < 2:
        raise ValueError("at least two cells per axis are required")
    return (
        diagonals[:, :-1, :-1]
        ^ diagonals[:, :-1, 1:]
        ^ diagonals[:, 1:, :-1]
        ^ diagonals[:, 1:, 1:]
    )


def interior_four_spin_correlation(diagonals: Any) -> Any:
    """Return the four-spin correlation controlling interior parity defects."""

    torch = _torch()
    defects = interior_parity_defects(diagonals)
    return (1.0 - 2.0 * defects.to(torch.float64)).mean(dim=(1, 2))


def mixed_curvature_neighbor_prediction(
    resolution: int,
    hurst: float,
    stride: int,
    *,
    device: Any = None,
    dtype: Any = None,
) -> dict[str, float]:
    """Predict neighbor sign agreement from the finite spectral covariance.

    Mixed differences of a cutoff fractional Gaussian surface are centered
    jointly Gaussian. If neighboring values have correlation ``rho``, the
    Gaussian arcsine identity gives

    ``P(sign X = sign Y) = 1/2 + asin(rho)/pi``.
    """

    torch = _torch()
    if resolution < 8:
        raise ValueError("resolution must be at least 8")
    if not 0.0 < hurst < 1.0:
        raise ValueError("hurst must lie strictly between zero and one")
    if stride < 1 or resolution % stride != 0:
        raise ValueError("stride must be a positive divisor of resolution")
    if dtype is None:
        dtype = torch.float64
    frequencies = torch.fft.fftfreq(
        resolution,
        d=1.0 / resolution,
        device=device,
        dtype=dtype,
    )
    first, second = torch.meshgrid(frequencies, frequencies, indexing="ij")
    radius = torch.sqrt(first.square() + second.square())
    spectral_density = torch.where(
        radius > 0,
        radius.pow(-2.0 * (hurst + 1.0)),
        torch.zeros_like(radius),
    )
    first_phase = 2.0 * torch.pi * first * stride / resolution
    second_phase = 2.0 * torch.pi * second * stride / resolution
    mixed_filter_power = (
        16.0
        * torch.sin(0.5 * first_phase).square()
        * torch.sin(0.5 * second_phase).square()
    )
    filtered_power = spectral_density * mixed_filter_power
    variance = filtered_power.sum()
    if float(variance.item()) <= 0.0:
        raise ValueError("mixed-curvature variance vanished")
    horizontal = (
        filtered_power * torch.cos(second_phase)
    ).sum() / variance
    vertical = (
        filtered_power * torch.cos(first_phase)
    ).sum() / variance
    correlation = 0.5 * (horizontal + vertical)
    correlation = correlation.clamp(-1.0, 1.0)
    agreement = 0.5 + torch.asin(correlation) / math.pi
    return {
        "neighbor_correlation": float(correlation.item()),
        "predicted_sign_agreement": float(agreement.item()),
    }


def compactified_triangulation_degrees(diagonals: Any) -> Any:
    """Return degrees after adding one vertex outside the square boundary."""

    torch = _torch()
    if diagonals.ndim != 3 or diagonals.shape[-2] != diagonals.shape[-1]:
        raise ValueError("diagonals must have shape [batch, n-1, n-1]")
    batch, cell_count, _ = diagonals.shape
    grid_size = cell_count + 1
    degrees = torch.zeros(
        (batch, grid_size, grid_size),
        device=diagonals.device,
        dtype=torch.int64,
    )

    degrees[:, :, :-1] += 1
    degrees[:, :, 1:] += 1
    degrees[:, :-1, :] += 1
    degrees[:, 1:, :] += 1

    main = diagonals.to(torch.int64)
    anti = (~diagonals).to(torch.int64)
    degrees[:, :-1, :-1] += main
    degrees[:, 1:, 1:] += main
    degrees[:, 1:, :-1] += anti
    degrees[:, :-1, 1:] += anti

    boundary = torch.zeros(
        (grid_size, grid_size),
        device=diagonals.device,
        dtype=torch.int64,
    )
    boundary[0, :] = 1
    boundary[-1, :] = 1
    boundary[:, 0] = 1
    boundary[:, -1] = 1
    degrees += boundary
    exterior_degree = torch.full(
        (batch, 1),
        4 * (grid_size - 1),
        device=diagonals.device,
        dtype=torch.int64,
    )
    return torch.cat((degrees.reshape(batch, -1), exterior_degree), dim=1)


def curvature_triangulation_statistics(window: Any) -> dict[str, Any]:
    """Return exact finite coloring and roughness statistics per surface."""

    torch = _torch()
    diagonals = mixed_curvature_diagonals(window)
    degrees = compactified_triangulation_degrees(diagonals)
    odd_count = torch.sum(torch.remainder(degrees, 2), dim=1)
    vertex_count = degrees.shape[1]
    grid_size = window.shape[-1]
    expected_edge_count = 3 * vertex_count - 6
    same_horizontal = diagonals[:, :, 1:] == diagonals[:, :, :-1]
    same_vertical = diagonals[:, 1:, :] == diagonals[:, :-1, :]
    sign_agreement = torch.cat(
        (same_horizontal.flatten(1), same_vertical.flatten(1)),
        dim=1,
    ).float().mean(dim=1)
    mixed_curvature = (
        window[:, :-1, :-1]
        + window[:, 1:, 1:]
        - window[:, 1:, :-1]
        - window[:, :-1, 1:]
    )
    return {
        "grid_size": grid_size,
        "vertex_count": vertex_count,
        "edge_count": expected_edge_count,
        "odd_degree_vertex_count": odd_count,
        "odd_degree_vertex_fraction": odd_count.float() / vertex_count,
        # A sphere triangulation is 3-colorable iff it is Eulerian.
        # Otherwise its triangle lower bound and Four Color upper bound meet.
        "chromatic_number": torch.where(
            odd_count == 0,
            torch.full_like(odd_count, 3),
            torch.full_like(odd_count, 4),
        ),
        "main_diagonal_fraction": diagonals.float().mean(dim=(1, 2)),
        "neighbor_diagonal_sign_agreement": sign_agreement,
        "mean_absolute_mixed_curvature": mixed_curvature.abs().mean(
            dim=(1, 2)
        ),
    }


def diagonal_majority_transport_mismatch(
    coarse_diagonals: Any,
    fine_diagonals: Any,
) -> Any:
    """Compare coarse choices with majority fine choices in nested cells."""

    if (
        coarse_diagonals.ndim != 3
        or fine_diagonals.ndim != 3
        or coarse_diagonals.shape[0] != fine_diagonals.shape[0]
        or coarse_diagonals.shape[1] != coarse_diagonals.shape[2]
        or fine_diagonals.shape[1] != fine_diagonals.shape[2]
    ):
        raise ValueError("diagonal batches must be square and batch-aligned")
    coarse_count = coarse_diagonals.shape[1]
    fine_count = fine_diagonals.shape[1]
    if fine_count % coarse_count != 0:
        raise ValueError("fine cell count must be divisible by coarse count")
    ratio = fine_count // coarse_count
    blocks = (
        fine_diagonals.reshape(
            fine_diagonals.shape[0],
            coarse_count,
            ratio,
            coarse_count,
            ratio,
        )
        .permute(0, 1, 3, 2, 4)
        .reshape(fine_diagonals.shape[0], coarse_count, coarse_count, -1)
    )
    majority = blocks.float().mean(dim=-1) >= 0.5
    return (majority != coarse_diagonals).float().mean(dim=(1, 2))


def second_order_structure_function(
    surfaces: Any,
    shifts: Sequence[int],
) -> Any:
    """Return mean squared increments at periodic lattice shifts."""

    torch = _torch()
    if surfaces.ndim != 3:
        raise ValueError("surfaces must have shape [batch, n, n]")
    values = []
    for shift in shifts:
        if shift < 1 or shift >= surfaces.shape[-1] // 2:
            raise ValueError("each shift must lie in [1, resolution/2)")
        horizontal = torch.roll(surfaces, shifts=shift, dims=2) - surfaces
        vertical = torch.roll(surfaces, shifts=shift, dims=1) - surfaces
        values.append(
            0.5
            * (
                horizontal.square().mean(dim=(1, 2))
                + vertical.square().mean(dim=(1, 2))
            )
        )
    return torch.stack(values, dim=1)


def estimate_hurst_from_structure_function(
    surfaces: Any,
    shifts: Sequence[int],
) -> Any:
    """Estimate ``H`` from the slope of ``log E|h(x+r)-h(x)|^2``."""

    torch = _torch()
    structure = second_order_structure_function(surfaces, shifts)
    log_shifts = torch.log(
        torch.tensor(
            shifts,
            device=surfaces.device,
            dtype=surfaces.dtype,
        )
    )
    centered_shifts = log_shifts - log_shifts.mean()
    log_structure = torch.log(
        structure.clamp_min(torch.finfo(surfaces.dtype).eps)
    )
    slopes = torch.sum(
        centered_shifts * (log_structure - log_structure.mean(dim=1, keepdim=True)),
        dim=1,
    ) / torch.sum(centered_shifts.square())
    return 0.5 * slopes


def compactified_triangulation_adjacency(diagonals: Any) -> list[set[int]]:
    """Build one explicit finite adjacency graph for coloring/visualization."""

    if diagonals.ndim != 2 or diagonals.shape[0] != diagonals.shape[1]:
        raise ValueError("diagonals must have shape [n-1, n-1]")
    grid_size = diagonals.shape[0] + 1
    exterior = grid_size * grid_size
    adjacency = [set() for _ in range(exterior + 1)]

    def add_edge(first: int, second: int) -> None:
        adjacency[first].add(second)
        adjacency[second].add(first)

    def index(row: int, column: int) -> int:
        return row * grid_size + column

    for row in range(grid_size):
        for column in range(grid_size - 1):
            add_edge(index(row, column), index(row, column + 1))
    for row in range(grid_size - 1):
        for column in range(grid_size):
            add_edge(index(row, column), index(row + 1, column))
    for row in range(grid_size - 1):
        for column in range(grid_size - 1):
            if bool(diagonals[row, column]):
                add_edge(index(row, column), index(row + 1, column + 1))
            else:
                add_edge(index(row + 1, column), index(row, column + 1))
    boundary = (
        [index(0, column) for column in range(grid_size)]
        + [index(row, grid_size - 1) for row in range(1, grid_size)]
        + [
            index(grid_size - 1, column)
            for column in range(grid_size - 2, -1, -1)
        ]
        + [index(row, 0) for row in range(grid_size - 2, 0, -1)]
    )
    for vertex in boundary:
        add_edge(exterior, vertex)
    return adjacency


def dsatur_coloring(
    adjacency: Sequence[set[int]],
    color_count: int = 4,
) -> list[int]:
    """Find a deterministic coloring using exact DSATUR backtracking."""

    if color_count < 1:
        raise ValueError("color_count must be positive")
    vertex_count = len(adjacency)
    colors = [-1] * vertex_count

    def choose_vertex() -> int:
        candidates = [index for index, color in enumerate(colors) if color < 0]
        return max(
            candidates,
            key=lambda index: (
                len(
                    {
                        colors[neighbor]
                        for neighbor in adjacency[index]
                        if colors[neighbor] >= 0
                    }
                ),
                len(adjacency[index]),
                -index,
            ),
        )

    def search(colored_count: int) -> bool:
        if colored_count == vertex_count:
            return True
        vertex = choose_vertex()
        forbidden = {
            colors[neighbor]
            for neighbor in adjacency[vertex]
            if colors[neighbor] >= 0
        }
        for color in range(color_count):
            if color in forbidden:
                continue
            colors[vertex] = color
            if search(colored_count + 1):
                return True
            colors[vertex] = -1
        return False

    if not search(0):
        raise ValueError(f"graph is not {color_count}-colorable")
    return colors


def coloring_is_proper(
    adjacency: Sequence[set[int]],
    colors: Sequence[int],
) -> bool:
    """Return whether every graph edge has distinct endpoint colors."""

    if len(adjacency) != len(colors):
        return False
    return all(
        colors[first] != colors[second]
        for first, neighbors in enumerate(adjacency)
        for second in neighbors
    )
