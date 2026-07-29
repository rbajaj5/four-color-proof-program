"""Exact Bateman-field formulas and numerical Maxwell diagnostics.

The field family follows Kedia, Bialynicki-Birula, Peralta-Salas, and
Irvine, Phys. Rev. Lett. 111, 150404 (2013).  PyTorch is imported lazily
so the exact graph-coloring core of this repository stays dependency-free.
"""

from __future__ import annotations

from math import gcd
from typing import Any


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "PyTorch is required for the Maxwell field experiment"
        ) from exc
    return torch


def _validate_pq(p: int, q: int) -> None:
    if not isinstance(p, int) or not isinstance(q, int) or p < 1 or q < 1:
        raise ValueError("p and q must be positive integers")


def bateman_alpha_beta_gradients(
    points: Any,
    time: float | Any = 0.0,
) -> tuple[Any, Any, Any, Any]:
    """Evaluate Bateman potentials and their spatial gradients.

    ``points`` has shape ``[..., 3]`` and a real floating dtype.  Natural
    units with wave speed ``c = 1`` are used.
    """

    torch = _torch()
    if points.shape[-1] != 3 or points.is_complex():
        raise ValueError("points must be a real tensor with shape [..., 3]")
    if points.dtype not in (torch.float32, torch.float64):
        raise ValueError("points must use float32 or float64")

    complex_dtype = (
        torch.complex128 if points.dtype == torch.float64 else torch.complex64
    )
    xyz = points.to(dtype=complex_dtype)
    x, y, z = xyz.unbind(dim=-1)
    real_time = torch.as_tensor(
        time,
        dtype=points.dtype,
        device=points.device,
    )
    complex_time = real_time.to(dtype=complex_dtype)
    imaginary_unit = torch.as_tensor(
        1j,
        dtype=complex_dtype,
        device=points.device,
    )

    radius_squared = x * x + y * y + z * z
    denominator = radius_squared - (complex_time - imaginary_unit) ** 2
    numerator_alpha = (
        radius_squared
        - complex_time * complex_time
        - 1.0
        + 2.0 * imaginary_unit * z
    )
    numerator_beta = 2.0 * (x - imaginary_unit * y)

    grad_denominator = 2.0 * xyz
    grad_numerator_alpha = 2.0 * xyz
    grad_numerator_alpha = grad_numerator_alpha.clone()
    grad_numerator_alpha[..., 2] += 2.0 * imaginary_unit
    grad_numerator_beta = torch.zeros_like(xyz)
    grad_numerator_beta[..., 0] = 2.0
    grad_numerator_beta[..., 1] = -2.0 * imaginary_unit

    denominator_squared = denominator * denominator
    grad_alpha = (
        grad_numerator_alpha * denominator.unsqueeze(-1)
        - numerator_alpha.unsqueeze(-1) * grad_denominator
    ) / denominator_squared.unsqueeze(-1)
    grad_beta = (
        grad_numerator_beta * denominator.unsqueeze(-1)
        - numerator_beta.unsqueeze(-1) * grad_denominator
    ) / denominator_squared.unsqueeze(-1)
    return (
        numerator_alpha / denominator,
        numerator_beta / denominator,
        grad_alpha,
        grad_beta,
    )


def riemann_silberstein_field(
    points: Any,
    p: int,
    q: int,
    time: float | Any = 0.0,
) -> Any:
    """Return ``F = E + iB = grad(alpha**p) x grad(beta**q)``."""

    _validate_pq(p, q)
    torch = _torch()
    alpha, beta, grad_alpha, grad_beta = bateman_alpha_beta_gradients(
        points,
        time=time,
    )
    prefactor = (
        float(p * q)
        * alpha.pow(p - 1)
        * beta.pow(q - 1)
    )
    return prefactor.unsqueeze(-1) * torch.cross(
        grad_alpha,
        grad_beta,
        dim=-1,
    )


def electric_magnetic_fields(
    points: Any,
    p: int,
    q: int,
    time: float | Any = 0.0,
) -> tuple[Any, Any]:
    """Return real electric and magnetic fields for the Bateman solution."""

    field = riemann_silberstein_field(points, p=p, q=q, time=time)
    return field.real, field.imag


def stereographic_inverse(alpha: Any, beta: Any) -> Any:
    """Map standard ``S^3`` coordinates ``(alpha, beta)`` to ``R^3``."""

    torch = _torch()
    denominator = 1.0 - alpha.real
    if bool(torch.any(denominator.abs() < 1e-12)):
        raise ValueError("stereographic inverse encountered the north pole")
    return torch.stack(
        (
            beta.real / denominator,
            -beta.imag / denominator,
            alpha.imag / denominator,
        ),
        dim=-1,
    )


def magnetic_core_curve(
    p: int,
    q: int,
    samples: int,
    *,
    sign: int = 1,
    component_index: int = 0,
    device: str | Any = "cpu",
    dtype: Any | None = None,
) -> Any:
    """Sample one closed magnetic core component at ``t = 0``.

    The returned tensor has shape ``[samples + 1, 3]``; the final point
    repeats the first point to expose numerical closure error directly.
    """

    _validate_pq(p, q)
    if samples < 16:
        raise ValueError("samples must be at least 16")
    if sign not in (-1, 1):
        raise ValueError("sign must be -1 or 1")
    component_count_per_sign = gcd(p, q)
    if not 0 <= component_index < component_count_per_sign:
        raise ValueError("component_index is outside the gcd component range")

    torch = _torch()
    if dtype is None:
        dtype = torch.float64
    complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    theta = torch.linspace(
        0.0,
        2.0 * torch.pi,
        samples + 1,
        device=device,
        dtype=dtype,
    )
    # The level-set condition is p*arg(alpha) + q*arg(beta) = 0 or pi.
    # Dividing p and q by their gcd traces each component once; the q-th
    # roots of the level-set phase supply the distinct gcd components.
    reduced_p = p // component_count_per_sign
    reduced_q = q // component_count_per_sign
    extremum_phase = 0.0 if sign == 1 else -torch.pi
    alpha_phase = reduced_q * theta
    beta_phase = (
        -reduced_p * theta
        + (extremum_phase + 2.0 * torch.pi * component_index) / q
    )
    imaginary_unit = torch.as_tensor(
        1j,
        device=device,
        dtype=complex_dtype,
    )
    alpha = (
        (p / (p + q)) ** 0.5
        * torch.exp(imaginary_unit * alpha_phase)
    )
    beta = (
        (q / (p + q)) ** 0.5
        * torch.exp(imaginary_unit * beta_phase)
    )
    return stereographic_inverse(alpha, beta).to(dtype=dtype)


def all_magnetic_core_curves(
    p: int,
    q: int,
    samples: int,
    *,
    device: str | Any = "cpu",
    dtype: Any | None = None,
) -> tuple[Any, ...]:
    """Return all ``2*gcd(p,q)`` magnetic core components."""

    return tuple(
        magnetic_core_curve(
            p,
            q,
            samples,
            sign=sign,
            component_index=component_index,
            device=device,
            dtype=dtype,
        )
        for sign in (1, -1)
        for component_index in range(gcd(p, q))
    )


def core_tangent_alignment(
    curve: Any,
    p: int,
    q: int,
) -> Any:
    """Return absolute cosine alignment of ``B`` with a sampled core curve."""

    torch = _torch()
    if curve.ndim != 2 or curve.shape[1] != 3 or curve.shape[0] < 17:
        raise ValueError("curve must have shape [samples + 1, 3]")
    unique = curve[:-1]
    tangent = torch.roll(unique, shifts=-1, dims=0) - torch.roll(
        unique,
        shifts=1,
        dims=0,
    )
    _, magnetic = electric_magnetic_fields(unique, p=p, q=q, time=0.0)
    tangent_norm = torch.linalg.vector_norm(tangent, dim=-1)
    magnetic_norm = torch.linalg.vector_norm(magnetic, dim=-1)
    denominator = (tangent_norm * magnetic_norm).clamp_min(
        torch.finfo(curve.dtype).eps
    )
    return (
        torch.sum(tangent * magnetic, dim=-1).abs() / denominator
    ).clamp(max=1.0)


def maxwell_residuals(
    points: Any,
    p: int,
    q: int,
    *,
    step: float = 1e-4,
) -> dict[str, float]:
    """Estimate free-space Maxwell and null-condition residuals.

    Spatial and time derivatives use centered finite differences.  The
    returned values are diagnostics, not symbolic proof certificates.
    """

    torch = _torch()
    if step <= 0:
        raise ValueError("step must be positive")
    electric, magnetic = electric_magnetic_fields(points, p=p, q=q)

    electric_derivatives = []
    magnetic_derivatives = []
    for axis in range(3):
        offset = torch.zeros_like(points)
        offset[..., axis] = step
        electric_plus, magnetic_plus = electric_magnetic_fields(
            points + offset,
            p=p,
            q=q,
        )
        electric_minus, magnetic_minus = electric_magnetic_fields(
            points - offset,
            p=p,
            q=q,
        )
        electric_derivatives.append(
            (electric_plus - electric_minus) / (2.0 * step)
        )
        magnetic_derivatives.append(
            (magnetic_plus - magnetic_minus) / (2.0 * step)
        )

    jacobian_electric = torch.stack(electric_derivatives, dim=-1)
    jacobian_magnetic = torch.stack(magnetic_derivatives, dim=-1)
    divergence_electric = torch.diagonal(
        jacobian_electric,
        dim1=-2,
        dim2=-1,
    ).sum(dim=-1)
    divergence_magnetic = torch.diagonal(
        jacobian_magnetic,
        dim1=-2,
        dim2=-1,
    ).sum(dim=-1)

    def curl(jacobian: Any) -> Any:
        return torch.stack(
            (
                jacobian[..., 2, 1] - jacobian[..., 1, 2],
                jacobian[..., 0, 2] - jacobian[..., 2, 0],
                jacobian[..., 1, 0] - jacobian[..., 0, 1],
            ),
            dim=-1,
        )

    electric_plus_time, magnetic_plus_time = electric_magnetic_fields(
        points,
        p=p,
        q=q,
        time=step,
    )
    electric_minus_time, magnetic_minus_time = electric_magnetic_fields(
        points,
        p=p,
        q=q,
        time=-step,
    )
    time_electric = (
        electric_plus_time - electric_minus_time
    ) / (2.0 * step)
    time_magnetic = (
        magnetic_plus_time - magnetic_minus_time
    ) / (2.0 * step)
    faraday = time_magnetic + curl(jacobian_electric)
    ampere = time_electric - curl(jacobian_magnetic)
    null_dot = torch.sum(electric * magnetic, dim=-1)
    null_norm = (
        torch.sum(electric * electric, dim=-1)
        - torch.sum(magnetic * magnetic, dim=-1)
    )
    field_scale = torch.sqrt(
        torch.sum(electric * electric + magnetic * magnetic, dim=-1)
    )
    scale = float(field_scale.mean().item())
    scale = max(scale, torch.finfo(points.dtype).eps)

    def mean_abs(values: Any) -> float:
        return float(values.abs().mean().item())

    def max_abs(values: Any) -> float:
        return float(values.abs().max().item())

    def mean_vector_norm(values: Any) -> float:
        return float(torch.linalg.vector_norm(values, dim=-1).mean().item())

    return {
        "field_scale_mean": scale,
        "divergence_electric_mean_abs": mean_abs(divergence_electric),
        "divergence_electric_max_abs": max_abs(divergence_electric),
        "divergence_magnetic_mean_abs": mean_abs(divergence_magnetic),
        "divergence_magnetic_max_abs": max_abs(divergence_magnetic),
        "faraday_mean_norm": mean_vector_norm(faraday),
        "ampere_mean_norm": mean_vector_norm(ampere),
        "null_dot_mean_abs": mean_abs(null_dot),
        "null_norm_mean_abs": mean_abs(null_norm),
        "relative_divergence_magnetic_mean": (
            mean_abs(divergence_magnetic) / scale
        ),
        "relative_faraday_mean": mean_vector_norm(faraday) / scale,
        "relative_ampere_mean": mean_vector_norm(ampere) / scale,
    }


def expected_core_component_count(p: int, q: int) -> int:
    """Return the number of magnetic core components in the full family."""

    _validate_pq(p, q)
    return 2 * gcd(p, q)
