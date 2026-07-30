"""Classical complex Hopf-fibration identities used as a source gate."""

from __future__ import annotations

import math

import torch


def normalize_complex(vectors: torch.Tensor) -> torch.Tensor:
    if not vectors.is_complex() or vectors.ndim < 1:
        raise ValueError("vectors must be a complex tensor")
    norms = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
    if bool(torch.any(norms == 0)):
        raise ValueError("zero vectors do not define projective points")
    return vectors / norms


def projective_projector(vectors: torch.Tensor) -> torch.Tensor:
    """Represent points of CP^n by rank-one Hermitian projectors."""

    normalized = normalize_complex(vectors)
    return normalized.unsqueeze(-1) * normalized.conj().unsqueeze(-2)


def phase_rotate(vectors: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
    if phases.shape != vectors.shape[:-1]:
        raise ValueError("one phase is required per vector")
    return vectors * torch.exp(1j * phases).unsqueeze(-1)


def hopf_map(vectors: torch.Tensor) -> torch.Tensor:
    """Map normalized points of S^3 in C^2 to S^2."""

    if vectors.shape[-1] != 2:
        raise ValueError("the S^3 Hopf map requires two complex coordinates")
    first, second = normalize_complex(vectors).unbind(dim=-1)
    product = first * second.conj()
    return torch.stack(
        (
            2.0 * product.real,
            2.0 * product.imag,
            first.abs().square() - second.abs().square(),
        ),
        dim=-1,
    )


def projector_distance_residual(
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Check ||P_x-P_y||_F^2 = 2(1-|<x,y>|^2)."""

    left_normalized = normalize_complex(left)
    right_normalized = normalize_complex(right)
    left_projector = projective_projector(left_normalized)
    right_projector = projective_projector(right_normalized)
    observed = (
        left_projector - right_projector
    ).abs().square().sum(dim=(-2, -1))
    overlap = (
        left_normalized.conj() * right_normalized
    ).sum(dim=-1).abs().square()
    return observed - 2.0 * (1.0 - overlap)


def hopf_section(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """A standard local section of S^3 -> CP^1 away from one pole."""

    return torch.stack(
        (
            torch.cos(0.5 * theta).to(torch.complex128),
            torch.exp(1j * phi) * torch.sin(0.5 * theta),
        ),
        dim=-1,
    )


def chern_number_midpoint(theta_steps: int, *, device: torch.device) -> float:
    """Integrate F/(2*pi), where F=0.5 sin(theta)dtheta wedge dphi."""

    if theta_steps < 2:
        raise ValueError("theta_steps must be at least two")
    delta = math.pi / theta_steps
    theta = (
        torch.arange(theta_steps, device=device, dtype=torch.float64) + 0.5
    ) * delta
    integral = 0.5 * torch.sin(theta).sum() * delta * 2.0 * math.pi
    return float((integral / (2.0 * math.pi)).item())


def hopf_fiber(
    base_theta: float,
    base_phi: float,
    samples: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    if samples < 16:
        raise ValueError("samples must be at least 16")
    theta = torch.tensor(base_theta, dtype=torch.float64, device=device)
    phi = torch.tensor(base_phi, dtype=torch.float64, device=device)
    section = hopf_section(theta, phi)
    fiber_phase = torch.linspace(
        0.0,
        2.0 * math.pi,
        samples + 1,
        dtype=torch.float64,
        device=device,
    )
    return torch.exp(1j * fiber_phase).unsqueeze(-1) * section


def stereographic_s3(vectors: torch.Tensor) -> torch.Tensor:
    """Project S^3 subset C^2 to R^3 from (1,0)."""

    if vectors.shape[-1] != 2:
        raise ValueError("S^3 vectors require two complex coordinates")
    normalized = normalize_complex(vectors)
    first, second = normalized.unbind(dim=-1)
    denominator = 1.0 - first.real
    if bool(torch.any(denominator.abs() < 1e-12)):
        raise ValueError("fiber intersects the stereographic projection pole")
    return torch.stack(
        (
            second.real / denominator,
            -second.imag / denominator,
            first.imag / denominator,
        ),
        dim=-1,
    )


def gauss_linking_number(
    first_curve: torch.Tensor,
    second_curve: torch.Tensor,
) -> float:
    """Midpoint quadrature of the Gauss linking integral."""

    if first_curve.shape[-1] != 3 or second_curve.shape[-1] != 3:
        raise ValueError("curves must have shape [samples + 1, 3]")
    first_segments = first_curve[1:] - first_curve[:-1]
    second_segments = second_curve[1:] - second_curve[:-1]
    first_midpoints = 0.5 * (first_curve[1:] + first_curve[:-1])
    second_midpoints = 0.5 * (second_curve[1:] + second_curve[:-1])
    displacement = (
        first_midpoints[:, None, :] - second_midpoints[None, :, :]
    )
    cross = torch.cross(
        first_segments[:, None, :],
        second_segments[None, :, :],
        dim=-1,
    )
    denominator = torch.linalg.vector_norm(displacement, dim=-1).pow(3)
    integral = (cross * displacement).sum(dim=-1) / denominator
    return float((integral.sum() / (4.0 * math.pi)).item())
