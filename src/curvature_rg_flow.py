"""Fourier heat coarse-graining for closed polygonal curves.

The periodic heat semigroup suppresses Fourier mode ``k`` by
``exp(-tau*k**2)``.  For an arclength-parametrized curve its infinitesimal
velocity is the curvature vector, making it a precise abstract
curvature-driven renormalization flow.
"""

from __future__ import annotations

from typing import Any


DEFAULT_SCALE_BANDS = (
    ("macroscopic", 1, 2),
    ("mesoscopic", 3, 8),
    ("microscopic", 9, None),
)


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyTorch is required for curvature RG flow") from exc
    return torch


def _unique_closed_points(curve: Any) -> Any:
    torch = _torch()
    if curve.ndim != 2 or curve.shape[1] != 3 or curve.shape[0] < 17:
        raise ValueError("curve must have shape [samples + 1, 3]")
    if curve.is_complex():
        raise ValueError("curve must be real")
    closure_error = torch.linalg.vector_norm(curve[-1] - curve[0])
    if float(closure_error.item()) > 1e-7:
        raise ValueError("curve must repeat its first point at the end")
    return curve[:-1]


def periodic_heat_coarse_grain(
    curve: Any,
    tau: float,
    *,
    normalize: bool = False,
) -> Any:
    """Apply ``exp(tau * d_s^2)`` to a periodic sampled curve."""

    torch = _torch()
    if tau < 0:
        raise ValueError("tau must be nonnegative")
    unique = _unique_closed_points(curve)
    sample_count = unique.shape[0]
    coefficients = torch.fft.fft(unique, dim=0)
    frequencies = torch.fft.fftfreq(
        sample_count,
        d=1.0 / sample_count,
        device=unique.device,
        dtype=unique.dtype,
    )
    multiplier = torch.exp(-tau * frequencies.square())
    smoothed = torch.fft.ifft(
        coefficients * multiplier.unsqueeze(-1),
        dim=0,
    ).real
    if normalize:
        smoothed = smoothed - smoothed.mean(dim=0)
        radius = torch.sqrt(
            torch.sum(smoothed.square(), dim=-1).mean()
        )
        epsilon = torch.finfo(unique.dtype).eps
        if float(radius.item()) <= epsilon:
            raise ValueError("normalization encountered a collapsed curve")
        smoothed = smoothed / radius
    return torch.cat((smoothed, smoothed[:1]), dim=0)


def fourier_mode_amplitudes(curve: Any) -> Any:
    """Return vector-valued Fourier amplitudes for nonnegative modes."""

    torch = _torch()
    unique = _unique_closed_points(curve)
    centered = unique - unique.mean(dim=0)
    coefficients = torch.fft.fft(centered, dim=0) / unique.shape[0]
    positive = coefficients[: unique.shape[0] // 2 + 1]
    return torch.linalg.vector_norm(positive, dim=-1)


def first_nonzero_mode(curve: Any, relative_tolerance: float = 1e-10) -> int:
    """Return the first positive Fourier mode above a relative tolerance."""

    if relative_tolerance <= 0:
        raise ValueError("relative_tolerance must be positive")
    amplitudes = fourier_mode_amplitudes(curve)
    maximum = float(amplitudes[1:].max().item())
    if maximum == 0.0:
        raise ValueError("curve has no nonconstant Fourier mode")
    threshold = relative_tolerance * maximum
    for mode in range(1, len(amplitudes)):
        if float(amplitudes[mode].item()) > threshold:
            return mode
    raise AssertionError("a nonzero mode must exceed the relative threshold")


def effective_mode_count(
    curve: Any,
    relative_tolerance: float = 1e-6,
) -> int:
    """Count positive modes that remain visible at the requested scale."""

    if relative_tolerance <= 0:
        raise ValueError("relative_tolerance must be positive")
    amplitudes = fourier_mode_amplitudes(curve)
    maximum = float(amplitudes[1:].max().item())
    if maximum == 0.0:
        return 0
    return sum(
        float(amplitude.item()) > relative_tolerance * maximum
        for amplitude in amplitudes[1:]
    )


def spectral_energies(curve: Any) -> dict[str, float]:
    """Return exact-discrete Fourier Dirichlet and bending energies."""

    torch = _torch()
    unique = _unique_closed_points(curve)
    centered = unique - unique.mean(dim=0)
    coefficients = torch.fft.fft(centered, dim=0) / unique.shape[0]
    frequencies = torch.fft.fftfreq(
        unique.shape[0],
        d=1.0 / unique.shape[0],
        device=unique.device,
        dtype=unique.dtype,
    )
    power = torch.sum(coefficients.abs().square(), dim=-1)
    dirichlet = torch.sum(frequencies.square() * power)
    bending = torch.sum(frequencies.pow(4) * power)
    return {
        "spectral_dirichlet_energy": float(dirichlet.real.item()),
        "spectral_bending_energy": float(bending.real.item()),
    }


def spectral_scale_bands(
    curve: Any,
    bands: tuple[tuple[str, int, int | None], ...] = DEFAULT_SCALE_BANDS,
) -> list[dict[str, float | int | str]]:
    """Partition spectral power and derivative energies by wavelength scale."""

    torch = _torch()
    unique = _unique_closed_points(curve)
    centered = unique - unique.mean(dim=0)
    coefficients = torch.fft.fft(centered, dim=0) / unique.shape[0]
    frequencies = torch.fft.fftfreq(
        unique.shape[0],
        d=1.0 / unique.shape[0],
        device=unique.device,
        dtype=unique.dtype,
    )
    absolute_frequency = frequencies.abs()
    power = torch.sum(coefficients.abs().square(), dim=-1)
    rows: list[dict[str, float | int | str]] = []
    previous_maximum = 0
    for name, minimum_mode, maximum_mode in bands:
        if not name:
            raise ValueError("scale-band names must be nonempty")
        if minimum_mode < 1:
            raise ValueError("scale bands must exclude the centroid mode")
        if minimum_mode <= previous_maximum:
            raise ValueError("scale bands must be strictly ordered and disjoint")
        if maximum_mode is not None and maximum_mode < minimum_mode:
            raise ValueError("scale-band maximum must not precede its minimum")
        mask = absolute_frequency >= minimum_mode
        if maximum_mode is not None:
            mask &= absolute_frequency <= maximum_mode
            previous_maximum = maximum_mode
        else:
            previous_maximum = unique.shape[0]
        band_power = torch.sum(power[mask])
        dirichlet = torch.sum(
            frequencies[mask].square() * power[mask]
        )
        bending = torch.sum(
            frequencies[mask].pow(4) * power[mask]
        )
        rows.append(
            {
                "scale": name,
                "minimum_mode": minimum_mode,
                "maximum_mode": (
                    maximum_mode if maximum_mode is not None else -1
                ),
                "spectral_power": float(band_power.real.item()),
                "dirichlet_energy": float(dirichlet.real.item()),
                "bending_energy": float(bending.real.item()),
            }
        )
    return rows


def geometric_curve_metrics(curve: Any) -> dict[str, float]:
    """Return polygon length and parameterization-invariant curvature metrics."""

    torch = _torch()
    unique = _unique_closed_points(curve)
    sample_count = unique.shape[0]
    parameter_step = 2.0 * torch.pi / sample_count
    first_derivative = (
        torch.roll(unique, shifts=-1, dims=0)
        - torch.roll(unique, shifts=1, dims=0)
    ) / (2.0 * parameter_step)
    second_derivative = (
        torch.roll(unique, shifts=-1, dims=0)
        - 2.0 * unique
        + torch.roll(unique, shifts=1, dims=0)
    ) / (parameter_step * parameter_step)
    speed = torch.linalg.vector_norm(first_derivative, dim=-1)
    curvature = torch.linalg.vector_norm(
        torch.cross(first_derivative, second_derivative, dim=-1),
        dim=-1,
    ) / speed.clamp_min(torch.finfo(unique.dtype).eps).pow(3)
    edges = torch.roll(unique, shifts=-1, dims=0) - unique
    edge_lengths = torch.linalg.vector_norm(edges, dim=-1)
    arc_weights = speed * parameter_step
    bending_energy = torch.sum(curvature.square() * arc_weights)
    return {
        "polygon_length": float(edge_lengths.sum().item()),
        "mean_curvature": float(curvature.mean().item()),
        "maximum_curvature": float(curvature.max().item()),
        "geometric_bending_energy": float(bending_energy.item()),
        "median_edge_length": float(edge_lengths.median().item()),
    }


def normalized_nonlocal_clearance(
    curve: Any,
    *,
    excluded_neighbors: int = 3,
) -> float:
    """Return minimum nonlocal vertex distance divided by median edge length."""

    torch = _torch()
    if excluded_neighbors < 1:
        raise ValueError("excluded_neighbors must be positive")
    unique = _unique_closed_points(curve)
    count = unique.shape[0]
    distances = torch.cdist(unique, unique)
    indices = torch.arange(count, device=unique.device)
    difference = (indices[:, None] - indices[None, :]).abs()
    cyclic = torch.minimum(difference, count - difference)
    distances = distances.masked_fill(cyclic <= excluded_neighbors, torch.inf)
    minimum = distances.min()
    edges = torch.roll(unique, shifts=-1, dims=0) - unique
    median_edge = torch.linalg.vector_norm(edges, dim=-1).median()
    return float((minimum / median_edge).item())


def leading_mode_rank(curve: Any, mode: int | None = None) -> int:
    """Return the real span rank of the leading harmonic's cosine/sine vectors."""

    torch = _torch()
    unique = _unique_closed_points(curve)
    centered = unique - unique.mean(dim=0)
    if mode is None:
        mode = first_nonzero_mode(curve)
    coefficient = torch.fft.fft(centered, dim=0)[mode] / unique.shape[0]
    frame = torch.stack((coefficient.real, -coefficient.imag), dim=0)
    singular_values = torch.linalg.svdvals(frame)
    tolerance = (
        torch.finfo(unique.dtype).eps
        * max(frame.shape)
        * singular_values.max()
    )
    return int(torch.sum(singular_values > tolerance).item())
