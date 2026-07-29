"""CUDA-friendly line-tied magnetic braid fields and diagnostics.

The field family follows the localized toroidal-flux construction used by
Wilmot-Smith, Hornig, and Pontin for model coronal loops.  The optional
``diffusion_time`` parameter applies the exact free heat evolution to each
Gaussian ring perturbation.  It is not a full resistive-MHD evolution.
"""

from __future__ import annotations

import math
from typing import Any


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyTorch is required for magnetic braid work") from exc
    return torch


def _as_batch_parameter(value: Any, points: Any) -> Any:
    torch = _torch()
    tensor = torch.as_tensor(
        value,
        device=points.device,
        dtype=points.dtype,
    )
    batch = points.shape[0]
    if tensor.ndim == 0:
        tensor = tensor.expand(batch)
    if tensor.shape != (batch,):
        raise ValueError("parameters must be scalar or have shape [batch]")
    return tensor.reshape((batch,) + (1,) * (points.ndim - 2))


def braid_ring_centers(
    cycles: int,
    *,
    axial_spacing: float = 8.0,
) -> tuple[tuple[float, int], ...]:
    """Return ``(z_center, sign)`` for the concatenated elementary braid."""

    if cycles < 1:
        raise ValueError("cycles must be positive")
    if axial_spacing <= 0:
        raise ValueError("axial_spacing must be positive")
    ring_count = 2 * cycles
    return tuple(
        (
            axial_spacing * (index - 0.5 * (ring_count - 1)),
            1 if index % 2 == 0 else -1,
        )
        for index in range(ring_count)
    )


def braided_field_and_current(
    points: Any,
    *,
    cycles: int,
    twist: Any,
    guide_field: Any,
    center_offset: Any,
    diffusion_time: Any = 0.0,
    radial_scale: float = math.sqrt(2.0),
    axial_scale: float = 2.0,
    axial_spacing: float = 8.0,
) -> tuple[Any, Any]:
    """Evaluate the braided magnetic field ``B`` and current ``curl(B)``.

    ``points`` has shape ``[batch, ..., 3]``.  Natural units set ``mu_0=1``.
    The guide field is independent of the ring amplitude so their ratio can be
    swept explicitly.
    """

    torch = _torch()
    if points.ndim < 3 or points.shape[-1] != 3 or points.is_complex():
        raise ValueError("points must be real with shape [batch, ..., 3]")
    if points.dtype not in (torch.float32, torch.float64):
        raise ValueError("points must use float32 or float64")
    if radial_scale <= 0 or axial_scale <= 0 or axial_spacing <= 0:
        raise ValueError("field length scales must be positive")
    twist_tensor = _as_batch_parameter(twist, points)
    guide_tensor = _as_batch_parameter(guide_field, points)
    offset_tensor = _as_batch_parameter(center_offset, points)
    diffusion_tensor = _as_batch_parameter(diffusion_time, points)
    if bool(torch.any(guide_tensor <= 0)):
        raise ValueError("guide_field must be positive")
    if bool(torch.any(offset_tensor < 0)):
        raise ValueError("center_offset must be nonnegative")
    if bool(torch.any(diffusion_tensor < 0)):
        raise ValueError("diffusion_time must be nonnegative")

    radial_variance_0 = radial_scale * radial_scale
    axial_variance_0 = axial_scale * axial_scale
    radial_variance = radial_variance_0 + 4.0 * diffusion_tensor
    axial_variance = axial_variance_0 + 4.0 * diffusion_tensor
    diffusion_amplitude = (
        (radial_variance_0 / radial_variance).pow(2)
        * axial_scale
        / torch.sqrt(axial_variance)
    )
    base_coefficient = 2.0 * twist_tensor / radial_scale

    magnetic = torch.zeros_like(points)
    current = torch.zeros_like(points)
    magnetic[..., 2] = guide_tensor
    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]
    for z_center, sign in braid_ring_centers(
        cycles,
        axial_spacing=axial_spacing,
    ):
        x_center = sign * offset_tensor
        dx = x - x_center
        dy = y
        dz = z - z_center
        gaussian = torch.exp(
            -(dx.square() + dy.square()) / radial_variance
            - dz.square() / axial_variance
        )
        coefficient = (
            sign * base_coefficient * diffusion_amplitude * gaussian
        )
        magnetic[..., 0] += -coefficient * dy
        magnetic[..., 1] += coefficient * dx

        current[..., 0] += (
            2.0 * coefficient * dx * dz / axial_variance
        )
        current[..., 1] += (
            2.0 * coefficient * dy * dz / axial_variance
        )
        current[..., 2] += 2.0 * coefficient * (
            1.0 - (dx.square() + dy.square()) / radial_variance
        )
    return magnetic, current


def braided_magnetic_field(points: Any, **kwargs: Any) -> Any:
    """Evaluate only the magnetic field."""

    magnetic, _ = braided_field_and_current(points, **kwargs)
    return magnetic


def integrate_line_tied_field_lines(
    seed_xy: Any,
    *,
    cycles: int,
    twist: Any,
    guide_field: Any,
    center_offset: Any,
    diffusion_time: Any = 0.0,
    steps_per_cycle: int = 128,
    axial_spacing: float = 8.0,
) -> Any:
    """Integrate batched field lines from the lower to upper tied boundary.

    Integration uses ``d(x,y)/dz = (B_x/B_z, B_y/B_z)`` and fixed-step RK4.
    The result has shape ``[batch, steps + 1, seeds, 3]``.
    """

    torch = _torch()
    if seed_xy.ndim != 2 or seed_xy.shape[-1] != 2:
        raise ValueError("seed_xy must have shape [seeds, 2]")
    if steps_per_cycle < 16:
        raise ValueError("steps_per_cycle must be at least 16")
    parameters = (twist, guide_field, center_offset, diffusion_time)
    batch_sizes = []
    for parameter in parameters:
        tensor = torch.as_tensor(parameter)
        if tensor.ndim == 1:
            batch_sizes.append(int(tensor.shape[0]))
        elif tensor.ndim > 1:
            raise ValueError("integration parameters must be scalar or vectors")
    batch = batch_sizes[0] if batch_sizes else 1
    if any(size != batch for size in batch_sizes):
        raise ValueError("all vector parameters must share one batch size")
    xy = seed_xy.unsqueeze(0).expand(batch, -1, -1).clone()
    step_count = steps_per_cycle * cycles
    z_min = -axial_spacing * cycles
    z_max = axial_spacing * cycles
    dz = (z_max - z_min) / step_count

    def slope(state_xy: Any, z_value: float) -> Any:
        z_column = torch.full(
            (*state_xy.shape[:-1], 1),
            z_value,
            device=state_xy.device,
            dtype=state_xy.dtype,
        )
        points = torch.cat((state_xy, z_column), dim=-1)
        magnetic = braided_magnetic_field(
            points,
            cycles=cycles,
            twist=twist,
            guide_field=guide_field,
            center_offset=center_offset,
            diffusion_time=diffusion_time,
            axial_spacing=axial_spacing,
        )
        return magnetic[..., :2] / magnetic[..., 2:].clamp_min(
            torch.finfo(state_xy.dtype).eps
        )

    trajectory = []
    z_value = z_min
    z_column = torch.full(
        (batch, seed_xy.shape[0], 1),
        z_value,
        device=seed_xy.device,
        dtype=seed_xy.dtype,
    )
    trajectory.append(torch.cat((xy, z_column), dim=-1))
    for _ in range(step_count):
        first = slope(xy, z_value)
        second = slope(xy + 0.5 * dz * first, z_value + 0.5 * dz)
        third = slope(xy + 0.5 * dz * second, z_value + 0.5 * dz)
        fourth = slope(xy + dz * third, z_value + dz)
        xy = xy + dz * (first + 2.0 * second + 2.0 * third + fourth) / 6.0
        z_value += dz
        z_column = torch.full(
            (batch, seed_xy.shape[0], 1),
            z_value,
            device=seed_xy.device,
            dtype=seed_xy.dtype,
        )
        trajectory.append(torch.cat((xy, z_column), dim=-1))
    return torch.stack(trajectory, dim=1)


def seed_grid(
    grid_size: int,
    extent: float,
    *,
    device: Any = None,
    dtype: Any = None,
) -> Any:
    """Return a square ``grid_size x grid_size`` seed grid."""

    torch = _torch()
    if grid_size < 3 or grid_size % 2 == 0:
        raise ValueError("grid_size must be odd and at least 3")
    if extent <= 0:
        raise ValueError("extent must be positive")
    if dtype is None:
        dtype = torch.float64
    coordinates = torch.linspace(
        -extent,
        extent,
        grid_size,
        device=device,
        dtype=dtype,
    )
    x, y = torch.meshgrid(coordinates, coordinates, indexing="ij")
    return torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1)


def neighbor_pairs(grid_size: int, *, device: Any = None) -> Any:
    """Return horizontal and vertical nearest-neighbor seed-index pairs."""

    torch = _torch()
    pairs = []
    for first in range(grid_size):
        for second in range(grid_size):
            index = first * grid_size + second
            if first + 1 < grid_size:
                pairs.append((index, (first + 1) * grid_size + second))
            if second + 1 < grid_size:
                pairs.append((index, first * grid_size + second + 1))
    return torch.tensor(pairs, device=device, dtype=torch.long)


def pairwise_winding(
    trajectories: Any,
    pairs: Any,
    *,
    coarse_factor: int = 1,
) -> Any:
    """Return winding of selected field-line pairs about one another."""

    torch = _torch()
    if trajectories.ndim != 4 or trajectories.shape[-1] != 3:
        raise ValueError("trajectories must have shape [batch, steps, seeds, 3]")
    if coarse_factor < 1:
        raise ValueError("coarse_factor must be positive")
    indices = torch.arange(
        0,
        trajectories.shape[1],
        coarse_factor,
        device=trajectories.device,
    )
    if int(indices[-1]) != trajectories.shape[1] - 1:
        indices = torch.cat(
            (
                indices,
                torch.tensor(
                    [trajectories.shape[1] - 1],
                    device=indices.device,
                    dtype=indices.dtype,
                ),
            )
        )
    selected = trajectories[:, indices, :, :2]
    displacement = (
        selected[:, :, pairs[:, 1], :]
        - selected[:, :, pairs[:, 0], :]
    )
    angles = torch.atan2(displacement[..., 1], displacement[..., 0])
    differences = angles[:, 1:] - angles[:, :-1]
    differences = torch.atan2(torch.sin(differences), torch.cos(differences))
    return torch.sum(differences, dim=1) / (2.0 * torch.pi)


def spherical_path_length(
    directions: Any,
    *,
    coarse_factor: int = 1,
) -> Any:
    """Return spherical arc length for normalized direction trajectories.

    The input shape is ``[batch, steps, paths, 3]``.  Decimation gives a
    direction-space coarse graining analogous to straightening a reflected
    billiard path after unfolding.
    """

    torch = _torch()
    if directions.ndim != 4 or directions.shape[-1] != 3:
        raise ValueError("directions must have shape [batch, steps, paths, 3]")
    if coarse_factor < 1:
        raise ValueError("coarse_factor must be positive")
    indices = torch.arange(
        0,
        directions.shape[1],
        coarse_factor,
        device=directions.device,
    )
    if int(indices[-1]) != directions.shape[1] - 1:
        indices = torch.cat(
            (
                indices,
                torch.tensor(
                    [directions.shape[1] - 1],
                    device=indices.device,
                    dtype=indices.dtype,
                ),
            )
        )
    selected = directions[:, indices]
    selected = selected / torch.linalg.vector_norm(
        selected,
        dim=-1,
        keepdim=True,
    ).clamp_min(torch.finfo(selected.dtype).eps)
    dot_products = torch.sum(selected[:, 1:] * selected[:, :-1], dim=-1)
    increments = torch.acos(dot_products.clamp(-1.0, 1.0))
    return torch.sum(increments, dim=1)


def field_direction_spherical_path_length(
    trajectories: Any,
    *,
    cycles: int,
    twist: Any,
    guide_field: Any,
    center_offset: Any,
    diffusion_time: Any = 0.0,
    axial_spacing: float = 8.0,
    coarse_factor: int = 1,
) -> Any:
    """Evaluate magnetic directions and return their spherical path lengths."""

    magnetic = braided_magnetic_field(
        trajectories,
        cycles=cycles,
        twist=twist,
        guide_field=guide_field,
        center_offset=center_offset,
        diffusion_time=diffusion_time,
        axial_spacing=axial_spacing,
    )
    return spherical_path_length(magnetic, coarse_factor=coarse_factor)


def field_line_mapping_jacobian(
    endpoint_xy: Any,
    *,
    grid_size: int,
    extent: float,
) -> Any:
    """Estimate the line-tied mapping Jacobian on grid interiors."""

    if endpoint_xy.ndim != 3 or endpoint_xy.shape[-1] != 2:
        raise ValueError("endpoint_xy must have shape [batch, seeds, 2]")
    if endpoint_xy.shape[1] != grid_size * grid_size:
        raise ValueError("endpoint count does not match grid_size")
    mapping = endpoint_xy.reshape(endpoint_xy.shape[0], grid_size, grid_size, 2)
    spacing = 2.0 * extent / (grid_size - 1)
    derivative_x = (
        mapping[:, 2:, 1:-1] - mapping[:, :-2, 1:-1]
    ) / (2.0 * spacing)
    derivative_y = (
        mapping[:, 1:-1, 2:] - mapping[:, 1:-1, :-2]
    ) / (2.0 * spacing)
    a = derivative_x[..., 0]
    b = derivative_y[..., 0]
    c = derivative_x[..., 1]
    d = derivative_y[..., 1]
    return _torch().stack(
        (
            _torch().stack((a, b), dim=-1),
            _torch().stack((c, d), dim=-1),
        ),
        dim=-2,
    )


def squashing_factor_from_mapping(
    endpoint_xy: Any,
    *,
    grid_size: int,
    extent: float,
    normal_field_ratio: float | None = None,
) -> Any:
    """Estimate the line-tied mapping squashing factor on grid interiors.

    ``normal_field_ratio`` supplies the exact absolute Jacobian determinant
    ``|B_n(start) / B_n(end)|`` when it is analytically known.  Leaving it
    unset uses the finite-difference determinant of the footpoint map.
    """

    torch = _torch()
    jacobian = field_line_mapping_jacobian(
        endpoint_xy,
        grid_size=grid_size,
        extent=extent,
    )
    a = jacobian[..., 0, 0]
    b = jacobian[..., 0, 1]
    c = jacobian[..., 1, 0]
    d = jacobian[..., 1, 1]
    if normal_field_ratio is None:
        determinant = (a * d - b * c).abs().clamp_min(
            torch.finfo(endpoint_xy.dtype).eps
        )
    else:
        if normal_field_ratio <= 0:
            raise ValueError("normal_field_ratio must be positive")
        determinant = torch.as_tensor(
            normal_field_ratio,
            device=endpoint_xy.device,
            dtype=endpoint_xy.dtype,
        )
    return (a.square() + b.square() + c.square() + d.square()) / determinant


def mapping_stretch_diagnostics(
    endpoint_xy: Any,
    *,
    grid_size: int,
    extent: float,
    axial_length: float,
) -> dict[str, Any]:
    """Return singular-value stretching and finite-length Lyapunov metrics."""

    torch = _torch()
    if axial_length <= 0:
        raise ValueError("axial_length must be positive")
    jacobian = field_line_mapping_jacobian(
        endpoint_xy,
        grid_size=grid_size,
        extent=extent,
    )
    singular_values = torch.linalg.svdvals(jacobian)
    maximum_stretch = singular_values[..., 0].clamp_min(
        torch.finfo(endpoint_xy.dtype).eps
    )
    determinant = torch.linalg.det(jacobian).abs()
    area_residual = (determinant - 1.0).abs()
    reduction_dimensions = tuple(range(1, maximum_stretch.ndim))
    return {
        "maximum_mapping_stretch": torch.amax(
            maximum_stretch,
            dim=reduction_dimensions,
        ),
        "mean_log_mapping_stretch": torch.mean(
            torch.log(maximum_stretch),
            dim=reduction_dimensions,
        ),
        "maximum_finite_length_lyapunov": torch.amax(
            torch.log(maximum_stretch) / axial_length,
            dim=reduction_dimensions,
        ),
        "minimum_mapping_jacobian_abs_determinant": torch.amin(
            determinant,
            dim=reduction_dimensions,
        ),
        "mean_area_preservation_residual": torch.mean(
            area_residual,
            dim=reduction_dimensions,
        ),
        "maximum_area_preservation_residual": torch.amax(
            area_residual,
            dim=reduction_dimensions,
        ),
    }


def integrated_parallel_current(
    trajectories: Any,
    *,
    cycles: int,
    twist: Any,
    guide_field: Any,
    center_offset: Any,
    diffusion_time: Any = 0.0,
    axial_spacing: float = 8.0,
) -> Any:
    """Integrate ``J_parallel dl = (J dot B) dz / B_z`` on each line."""

    torch = _torch()
    magnetic, current = braided_field_and_current(
        trajectories,
        cycles=cycles,
        twist=twist,
        guide_field=guide_field,
        center_offset=center_offset,
        diffusion_time=diffusion_time,
        axial_spacing=axial_spacing,
    )
    integrand = torch.sum(current * magnetic, dim=-1) / magnetic[
        ..., 2
    ].clamp_min(torch.finfo(trajectories.dtype).eps)
    # ``integrand`` is [batch, axial_step, field_line].  Keep a singleton
    # field-line axis so PyTorch broadcasts the shared z grid correctly.
    z_values = trajectories[:, :, 0, 2].unsqueeze(-1)
    return torch.trapezoid(integrand, x=z_values, dim=1)


def volume_field_diagnostics(
    points: Any,
    *,
    cycles: int,
    twist: Any,
    guide_field: Any,
    center_offset: Any,
    diffusion_time: Any = 0.0,
) -> dict[str, Any]:
    """Return per-batch energy, current, and Lorentz-force diagnostics."""

    torch = _torch()
    magnetic, current = braided_field_and_current(
        points,
        cycles=cycles,
        twist=twist,
        guide_field=guide_field,
        center_offset=center_offset,
        diffusion_time=diffusion_time,
    )
    guide = _as_batch_parameter(guide_field, points)
    magnetic_norm = torch.linalg.vector_norm(magnetic, dim=-1)
    current_norm = torch.linalg.vector_norm(current, dim=-1)
    lorentz = torch.cross(current, magnetic, dim=-1)
    lorentz_norm = torch.linalg.vector_norm(lorentz, dim=-1)
    alignment_sine = lorentz_norm / (
        magnetic_norm * current_norm
    ).clamp_min(torch.finfo(points.dtype).eps)
    reduction_dimensions = tuple(range(1, points.ndim - 1))
    magnetic_energy = 0.5 * torch.mean(
        magnetic_norm.square(),
        dim=reduction_dimensions,
    )
    guide_energy = 0.5 * guide.reshape(points.shape[0]).square()
    return {
        "mean_magnetic_energy_density": magnetic_energy,
        "mean_free_energy_density_above_guide": magnetic_energy - guide_energy,
        "mean_current_density": torch.mean(
            current_norm,
            dim=reduction_dimensions,
        ),
        "maximum_current_density": torch.amax(
            current_norm,
            dim=reduction_dimensions,
        ),
        "mean_lorentz_force_density": torch.mean(
            lorentz_norm,
            dim=reduction_dimensions,
        ),
        "maximum_lorentz_force_density": torch.amax(
            lorentz_norm,
            dim=reduction_dimensions,
        ),
        "mean_force_free_misalignment_sine": torch.mean(
            alignment_sine,
            dim=reduction_dimensions,
        ),
    }


def finite_difference_divergence(
    points: Any,
    *,
    step: float = 1e-4,
    **field_kwargs: Any,
) -> Any:
    """Estimate ``div(B)`` for regression tests and audit samples."""

    if step <= 0:
        raise ValueError("step must be positive")
    derivatives = []
    for axis in range(3):
        offset = points.new_zeros(points.shape)
        offset[..., axis] = step
        plus = braided_magnetic_field(points + offset, **field_kwargs)
        minus = braided_magnetic_field(points - offset, **field_kwargs)
        derivatives.append((plus[..., axis] - minus[..., axis]) / (2.0 * step))
    return sum(derivatives)
