"""Sweep line-tied braided magnetic fields on CUDA.

The branch evaluates analytic divergence-free initial fields, exact free
diffusion of their Gaussian ring perturbations, field-line mappings, and
grammar/topology proxies.  It does not solve the full resistive-MHD system.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.magnetic_braid_mhd import (
    field_direction_spherical_path_length,
    finite_difference_divergence,
    integrate_line_tied_field_lines,
    integrated_parallel_current,
    mapping_stretch_diagnostics,
    neighbor_pairs,
    pairwise_winding,
    seed_grid,
    squashing_factor_from_mapping,
    volume_field_diagnostics,
)


OUTPUT_DIR = ROOT / "results" / "magnetic_braid_mhd_gpu"
SMOKE_OUTPUT_DIR = ROOT / "results" / "magnetic_braid_mhd_gpu_smoke"
COARSE_FACTORS = (1, 2, 4, 8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--seed-grid", type=int)
    parser.add_argument("--steps-per-cycle", type=int)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def select_device(allow_cpu: bool) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if allow_cpu:
        return torch.device("cpu")
    raise RuntimeError(
        "CUDA is unavailable; pass --allow-cpu only for a deliberate CPU run"
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parameter_grid(smoke: bool) -> list[dict[str, float | int | str]]:
    if smoke:
        cycles_values = (1,)
        twist_values = (0.5, 1.0)
        guide_values = (1.0,)
        offset_values = (1.0,)
        diffusion_values = (0.0, 0.5)
    else:
        cycles_values = (1, 2, 3)
        twist_values = (0.5, 1.0, 1.5)
        guide_values = (0.75, 1.25)
        offset_values = (0.75, 1.25)
        diffusion_values = (0.0, 0.25, 1.0)
    rows = []
    for cycles in cycles_values:
        for twist in twist_values:
            for guide in guide_values:
                for offset in offset_values:
                    for diffusion in diffusion_values:
                        rows.append(
                            {
                                "config_id": (
                                    f"n{cycles}_k{twist:g}_b{guide:g}_"
                                    f"s{offset:g}_rho{diffusion:g}"
                                ),
                                "cycles": cycles,
                                "twist": twist,
                                "guide_field": guide,
                                "center_offset": offset,
                                "diffusion_time": diffusion,
                            }
                        )
    return rows


def volume_grid(
    cycles: int,
    *,
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
    xy_points: int = 9,
) -> torch.Tensor:
    x = torch.linspace(-3.0, 3.0, xy_points, device=device, dtype=dtype)
    y = torch.linspace(-3.0, 3.0, xy_points, device=device, dtype=dtype)
    z = torch.linspace(
        -8.0 * cycles,
        8.0 * cycles,
        8 * cycles + 1,
        device=device,
        dtype=dtype,
    )
    x_grid, y_grid, z_grid = torch.meshgrid(x, y, z, indexing="ij")
    points = torch.stack(
        (x_grid.reshape(-1), y_grid.reshape(-1), z_grid.reshape(-1)),
        dim=-1,
    )
    return points.unsqueeze(0).expand(batch, -1, -1)


def tensor_values(values: list[dict[str, Any]], key: str, device: Any) -> Any:
    return torch.tensor(
        [float(row[key]) for row in values],
        device=device,
        dtype=torch.float32,
    )


def render_field_lines(
    trajectory: np.ndarray,
    config: dict[str, Any],
    output_dir: Path,
) -> None:
    figure = plt.figure(figsize=(8.2, 7.2))
    axis = figure.add_subplot(111, projection="3d")
    line_count = trajectory.shape[1]
    stride = max(1, int(math.sqrt(line_count)) // 5)
    grid_size = int(round(math.sqrt(line_count)))
    selected = [
        index
        for row in range(0, grid_size, stride)
        for column in range(0, grid_size, stride)
        for index in (row * grid_size + column,)
    ]
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(selected)))
    for color, index in zip(colors, selected, strict=True):
        line = trajectory[:, index]
        axis.plot(
            line[:, 0],
            line[:, 1],
            line[:, 2],
            color=color,
            linewidth=1.1,
            alpha=0.9,
        )
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("line-tied z")
    axis.set_title(
        "Braided flux-tube field lines\n"
        f"n={config['cycles']}, k={config['twist']}, "
        f"B0={config['guide_field']}, rho={config['diffusion_time']}"
    )
    axis.view_init(elev=21, azim=39)
    figure.tight_layout()
    figure.savefig(output_dir / "maximum_q_field_lines.png", dpi=190)
    plt.close(figure)


def render_phase_diagram(rows: list[dict[str, Any]], output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    cycles = np.array([float(row["cycles"]) for row in rows])
    diffusion = np.array([float(row["diffusion_time"]) for row in rows])
    winding = np.array(
        [float(row["mean_absolute_neighbor_winding"]) for row in rows]
    )
    q_max = np.array([float(row["maximum_squashing_factor"]) for row in rows])
    current = np.array(
        [float(row["mean_integrated_parallel_current_abs"]) for row in rows]
    )
    lorentz = np.array(
        [float(row["mean_lorentz_force_density"]) for row in rows]
    )
    scatter = axes[0].scatter(
        winding,
        np.log10(np.maximum(q_max, 2.0)),
        c=diffusion,
        s=45 + 25 * cycles,
        cmap="plasma_r",
        alpha=0.82,
        edgecolor="white",
        linewidth=0.4,
    )
    axes[0].set_xlabel("mean |neighbor winding|")
    axes[0].set_ylabel("log10(max Q)")
    axes[0].set_title("strand winding vs bundle stretching")
    figure.colorbar(scatter, ax=axes[0], label="diffusion time")

    for cycle in sorted(set(cycles)):
        selected = cycles == cycle
        axes[1].scatter(
            current[selected],
            lorentz[selected],
            label=f"cycles={int(cycle)}",
            alpha=0.78,
        )
    axes[1].set_xlabel("mean |integrated parallel current|")
    axes[1].set_ylabel("mean |J x B|")
    axes[1].set_title("current and non-force-free stress")
    axes[1].legend()

    grouped = {}
    for row in rows:
        key = (int(row["cycles"]), float(row["diffusion_time"]))
        grouped.setdefault(key, []).append(
            float(row["maximum_squashing_factor"])
        )
    for cycle in sorted({key[0] for key in grouped}):
        x_values = sorted({key[1] for key in grouped if key[0] == cycle})
        y_values = [
            float(np.mean(grouped[(cycle, value)])) for value in x_values
        ]
        axes[2].plot(
            x_values,
            np.log10(np.maximum(y_values, 2.0)),
            marker="o",
            label=f"cycles={cycle}",
        )
    axes[2].set_xlabel("free-diffusion time")
    axes[2].set_ylabel("log10(mean max Q)")
    axes[2].set_title("diffusive topology loss")
    axes[2].legend()
    for axis in axes:
        axis.grid(True, alpha=0.25)
    figure.suptitle("CUDA line-tied magnetic-braid parameter sweep")
    figure.tight_layout()
    figure.savefig(output_dir / "magnetic_braid_phase_diagram.png", dpi=190)
    plt.close(figure)


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["cycles"]), float(row["diffusion_time"]))
        grouped.setdefault(key, []).append(row)
    summaries = []
    fields = (
        "maximum_squashing_factor",
        "mean_absolute_neighbor_winding",
        "mean_direction_sphere_path_length",
        "maximum_finite_length_lyapunov",
        "mean_integrated_parallel_current_abs",
        "mean_free_energy_density_above_guide",
        "mean_lorentz_force_density",
    )
    for (cycles, diffusion), selected in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "cycles": cycles,
            "diffusion_time": diffusion,
            "config_count": len(selected),
        }
        for field in fields:
            values = np.array([float(row[field]) for row in selected])
            summary[f"mean_{field}"] = float(np.mean(values))
            summary[f"max_{field}"] = float(np.max(values))
        summaries.append(summary)
    return summaries


def run_resolution_audit(
    rows: list[dict[str, Any]],
    *,
    device: torch.device,
    dtype: torch.dtype,
    steps_per_cycle: int,
    baseline_grid_size: int,
) -> list[dict[str, Any]]:
    """Refine the strongest mappings to separate signal from grid artifacts."""

    selections: dict[str, tuple[dict[str, Any], set[str]]] = {}

    def select(row: dict[str, Any], reason: str) -> None:
        config_id = str(row["config_id"])
        if config_id not in selections:
            selections[config_id] = (row, set())
        selections[config_id][1].add(reason)

    for row in sorted(
        rows,
        key=lambda item: float(item["mean_absolute_neighbor_winding"]),
        reverse=True,
    )[:2]:
        select(row, "top_winding")
    for row in sorted(
        rows,
        key=lambda item: float(item["maximum_squashing_factor"]),
        reverse=True,
    )[:2]:
        select(row, "top_q")
    adequately_resolved = [
        row
        for row in rows
        if float(row["maximum_area_preservation_residual"]) <= 0.5
    ]
    if adequately_resolved:
        select(
            max(
                adequately_resolved,
                key=lambda item: float(
                    item["mean_absolute_neighbor_winding"]
                ),
            ),
            "top_winding_area_residual_le_0.5",
        )

    audit_rows: list[dict[str, Any]] = []
    for row, reasons in selections.values():
        audit_rows.append(
            {
                "config_id": row["config_id"],
                "selection_reason": ";".join(sorted(reasons)),
                "grid_size": baseline_grid_size,
                "field_line_count": baseline_grid_size**2,
                "cycles": row["cycles"],
                "twist": row["twist"],
                "guide_field": row["guide_field"],
                "center_offset": row["center_offset"],
                "diffusion_time": row["diffusion_time"],
                "mean_absolute_neighbor_winding": row[
                    "mean_absolute_neighbor_winding"
                ],
                "maximum_absolute_neighbor_winding": row[
                    "maximum_absolute_neighbor_winding"
                ],
                "maximum_squashing_factor": row[
                    "maximum_squashing_factor"
                ],
                "maximum_unconstrained_squashing_factor": row[
                    "maximum_unconstrained_squashing_factor"
                ],
                "maximum_mapping_stretch": row["maximum_mapping_stretch"],
                "minimum_mapping_jacobian_abs_determinant": row[
                    "minimum_mapping_jacobian_abs_determinant"
                ],
                "maximum_area_preservation_residual": row[
                    "maximum_area_preservation_residual"
                ],
                "integration_and_diagnostics_seconds": 0.0,
            }
        )

    selected_rows = [item[0] for item in selections.values()]
    for grid_size in (25, 49):
        refined_seeds = seed_grid(
            grid_size,
            2.5,
            device=device,
            dtype=dtype,
        )
        refined_pairs = neighbor_pairs(grid_size, device=device)
        for cycles in sorted({int(row["cycles"]) for row in selected_rows}):
            group = [
                row for row in selected_rows if int(row["cycles"]) == cycles
            ]
            group_start = time.perf_counter()
            trajectory = integrate_line_tied_field_lines(
                refined_seeds,
                cycles=cycles,
                twist=tensor_values(group, "twist", device),
                guide_field=tensor_values(group, "guide_field", device),
                center_offset=tensor_values(group, "center_offset", device),
                diffusion_time=tensor_values(
                    group,
                    "diffusion_time",
                    device,
                ),
                steps_per_cycle=steps_per_cycle,
            )
            endpoint = trajectory[:, -1, :, :2]
            q_known = squashing_factor_from_mapping(
                endpoint,
                grid_size=grid_size,
                extent=2.5,
                normal_field_ratio=1.0,
            )
            q_finite_difference = squashing_factor_from_mapping(
                endpoint,
                grid_size=grid_size,
                extent=2.5,
            )
            stretch = mapping_stretch_diagnostics(
                endpoint,
                grid_size=grid_size,
                extent=2.5,
                axial_length=16.0 * cycles,
            )
            winding = pairwise_winding(trajectory, refined_pairs).abs()
            synchronize(device)
            group_seconds = time.perf_counter() - group_start
            for index, row in enumerate(group):
                audit_rows.append(
                    {
                        "config_id": row["config_id"],
                        "selection_reason": ";".join(
                            sorted(selections[str(row["config_id"])][1])
                        ),
                        "grid_size": grid_size,
                        "field_line_count": grid_size**2,
                        "cycles": row["cycles"],
                        "twist": row["twist"],
                        "guide_field": row["guide_field"],
                        "center_offset": row["center_offset"],
                        "diffusion_time": row["diffusion_time"],
                        "mean_absolute_neighbor_winding": float(
                            winding[index].mean().item()
                        ),
                        "maximum_absolute_neighbor_winding": float(
                            winding[index].max().item()
                        ),
                        "maximum_squashing_factor": float(
                            q_known[index].max().item()
                        ),
                        "maximum_unconstrained_squashing_factor": float(
                            q_finite_difference[index].max().item()
                        ),
                        "maximum_mapping_stretch": float(
                            stretch["maximum_mapping_stretch"][index].item()
                        ),
                        "minimum_mapping_jacobian_abs_determinant": float(
                            stretch[
                                "minimum_mapping_jacobian_abs_determinant"
                            ][index].item()
                        ),
                        "maximum_area_preservation_residual": float(
                            stretch[
                                "maximum_area_preservation_residual"
                            ][index].item()
                        ),
                        "integration_and_diagnostics_seconds": group_seconds,
                    }
                )
            print(
                f"resolution grid={grid_size} cycles={cycles} "
                f"configs={len(group)} elapsed={group_seconds:.3f}s"
            )
    return audit_rows


def make_report(
    rows: list[dict[str, Any]],
    coarse_rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    timing_rows: list[dict[str, Any]],
    audit: dict[str, Any],
    resolution_rows: list[dict[str, Any]],
) -> str:
    best_q = max(rows, key=lambda row: float(row["maximum_squashing_factor"]))
    best_winding = max(
        rows,
        key=lambda row: float(row["mean_absolute_neighbor_winding"]),
    )
    smallest_coarse_error = min(
        (
            row
            for row in coarse_rows
            if int(row["coarse_factor"]) > 1
        ),
        key=lambda row: float(row["mean_winding_error_from_fine"]),
    )
    mean_group_time = float(
        np.mean([float(row["total_group_seconds"]) for row in timing_rows])
    )
    adequately_resolved_count = sum(
        float(row["maximum_area_preservation_residual"]) <= 0.5
        for row in rows
    )
    maximum_coarse_winding_error = max(
        float(row["maximum_winding_error_from_fine"])
        for row in coarse_rows
    )
    maximum_direction_loss = max(
        float(row["maximum_direction_path_length_loss"])
        for row in coarse_rows
    )
    resolution_table = ""
    unresolved_refinement_ids: list[str] = []
    converged_refinement_ids: list[str] = []
    if resolution_rows:
        finest_grid = max(int(item["grid_size"]) for item in resolution_rows)
        finest = [
            row
            for row in resolution_rows
            if int(row["grid_size"]) == finest_grid
        ]
        unresolved_refinement_ids = [
            str(row["config_id"])
            for row in finest
            if float(row["maximum_area_preservation_residual"]) > 0.5
        ]
        converged_refinement_ids = [
            str(row["config_id"])
            for row in finest
            if float(row["maximum_area_preservation_residual"]) <= 0.1
        ]
        resolution_table = "\n".join(
            "| {config_id} | {grid_size} | "
            "{mean_absolute_neighbor_winding:.4g} | "
            "{maximum_squashing_factor:.4g} | "
            "{maximum_mapping_stretch:.4g} | "
            "{maximum_area_preservation_residual:.4g} |".format(**row)
            for row in finest
        )
    unresolved_text = (
        ", ".join(f"`{value}`" for value in unresolved_refinement_ids)
        or "none"
    )
    converged_text = (
        ", ".join(f"`{value}`" for value in converged_refinement_ids)
        or "none"
    )
    coarse_by_factor = []
    for factor in COARSE_FACTORS:
        selected = [
            row
            for row in coarse_rows
            if int(row["coarse_factor"]) == factor
        ]
        coarse_by_factor.append(
            {
                "factor": factor,
                "mean_direction_loss": float(
                    np.mean(
                        [
                            float(row["mean_direction_path_length_loss"])
                            for row in selected
                        ]
                    )
                ),
                "maximum_direction_loss": max(
                    float(row["maximum_direction_path_length_loss"])
                    for row in selected
                ),
                "maximum_winding_error": max(
                    float(row["maximum_winding_error_from_fine"])
                    for row in selected
                ),
            }
        )
    coarse_table = "\n".join(
        "| {factor} | {mean_direction_loss:.4g} | "
        "{maximum_direction_loss:.4g} | "
        "{maximum_winding_error:.4g} |".format(**row)
        for row in coarse_by_factor
    )
    table = "\n".join(
        "| {cycles} | {diffusion_time:g} | "
        "{mean_maximum_squashing_factor:.4g} | "
        "{mean_mean_absolute_neighbor_winding:.4g} | "
        "{mean_mean_direction_sphere_path_length:.4g} | "
        "{mean_mean_integrated_parallel_current_abs:.4g} | "
        "{mean_mean_lorentz_force_density:.4g} |".format(**row)
        for row in summaries
    )
    return f"""# Magnetic Braid MHD GPU Report

## Scope

This branch sweeps the analytic line-tied magnetic braid family introduced by
Wilmot-Smith, Hornig, and Pontin for model solar coronal loops. Each field is
a uniform vertical guide field plus localized toroidal flux rings. The
construction is analytically divergence-free.

The `diffusion_time` parameter applies the exact free heat evolution
`partial_t B = eta Laplacian(B)` to each Gaussian ring perturbation, with the
product `eta t` represented by the parameter. It is a resistive coarse-
graining subflow, **not** a complete resistive-MHD solution: velocity,
pressure, density, energy transport, and reconnection feedback are absent.

## Bundle Geometry

The individual-line observable is pairwise winding of neighboring field
lines. The bundle observable is the differential of the lower-to-upper
footpoint map. Its squashing factor `Q`, largest singular value, and
finite-length Lyapunov exponent quantify the wavefront-like deformation of a
small line bundle. This is the magnetic analogue adopted from the billiard
wavefront intuition; magnetic field lines remain line-tied and are not
specularly reflected.

The normalized magnetic tangent is also projected to the upper direction
sphere. Its spherical path length is a smooth analogue of Galperin's finite
angular budget for reflected rays. Spatial decimation and direction-space
arc loss are reported separately.

At each axial slice, the collection of distinct line positions lies in a
configuration-space complement of pair-collision diagonals. Pairwise winding
records local motion around those forbidden sets. The many direction tubes
have a Kakeya-like visual appearance, but this finite, constrained family is
not a Kakeya set and no Kakeya dimension claim is made.

## CUDA Audit

- Actual device: `{audit["actual_device"]}`
- GPU: `{audit["gpu_name"]}`
- CUDA used: `{audit["cuda_used"]}`
- Dtype: `{audit["dtype"]}`
- Configurations: `{audit["configuration_count"]}`
- Field lines per configuration: `{audit["field_lines_per_config"]}`
- Steps per elementary cycle: `{audit["steps_per_cycle"]}`
- Total integrated line steps: `{audit["total_field_line_steps"]}`
- Peak CUDA allocation: `{audit["peak_cuda_memory_bytes"]}` bytes
- Total elapsed: `{audit["elapsed_seconds"]:.3f}` seconds

## Results by Braid Complexity and Diffusion

| Cycles | Diffusion | Mean max Q | Mean |winding| | Mean direction-sphere arc | Mean |integrated J_parallel| | Mean |J x B| |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

The largest squashing factor occurred for `{best_q["config_id"]}`:
`Q_max={float(best_q["maximum_squashing_factor"]):.5g}` with maximum
finite-length Lyapunov exponent
`{float(best_q["maximum_finite_length_lyapunov"]):.5g}`.

The largest mean neighbor winding occurred for
`{best_winding["config_id"]}`:
`{float(best_winding["mean_absolute_neighbor_winding"]):.5g}` turns.

Because the normal field is the same constant on both boundaries, the
primary `Q` uses the exact determinant ratio `|B_z(start)/B_z(end)|=1`.
The unconstrained finite-difference `Q` is retained in the runs CSV as a
resolution diagnostic. At the base grid,
`{adequately_resolved_count}/{len(rows)}` configurations had maximum local
area-preservation residual at most `0.5`.

## Targeted Resolution Audit

| Configuration | Grid | Mean |winding| | Max Q | Max stretch | Max area residual |
| --- | ---: | ---: | ---: | ---: | ---: |
{resolution_table or "| not run | - | - | - | - | - |"}

At the finest audit grid, the area-preservation residual was at most `0.1`
for {converged_text}. These estimates are numerically resolved within the
declared audit. The residual remained above `0.5` for {unresolved_text}.
For those high-gradient states, winding persists under refinement but `Q`
and maximum stretch continue to expose finer scales; they are not converged
point estimates.

## Multiscale Direction/Trajectory Ledger

Trajectory decimation is independent of physical diffusion. Galperin-style
angular-budget compression is represented by lost spherical tangent arc,
while braid information is checked by pairwise-winding error.

| Decimation factor | Mean direction arc loss | Max direction arc loss | Max winding error |
| ---: | ---: | ---: | ---: |
{coarse_table}

The smallest nontrivial mean winding error was
`{float(smallest_coarse_error["mean_winding_error_from_fine"]):.4g}` at
coarse factor `{smallest_coarse_error["coarse_factor"]}`. Across every row,
the largest winding error was `{maximum_coarse_winding_error:.4g}` turns and
the largest lost direction-sphere arc was `{maximum_direction_loss:.4g}`
radians. Thus winding is exceptionally stable here, while aggressive
decimation can hide substantial local bending.

## Physical Interpretation

- `Q` and finite-length Lyapunov stretching measure sensitivity of the
  footpoint map, not magnetic reconnection itself.
- `integrated_parallel_current` is relevant to three-dimensional
  reconnection, but a reconnection rate would require an electric field and
  a specified resistivity.
- `J x B` measures how far these analytic initial fields are from force-free
  balance. No ideal or magnetofrictional relaxation was performed.
- Pairwise winding of open, line-tied strands is a geometric statistic, not a
  closed-knot invariant.
- Magnetic energy is reported in normalized units and cannot be converted to
  solar-flare energy without a dimensional calibration.

## Performance

Mean group time was `{mean_group_time:.3f}` seconds. The expensive operation
is batched RK4 field-line integration followed by current evaluation along
the trajectories. Only summaries and one selected trajectory bundle are
transferred to CPU.

## Decision

This branch is suitable for locating parameter regimes with simultaneously
large winding, large bundle stretching, and tolerable force imbalance.
The robust finding is a hierarchy: braid cycles increase winding, current,
and tangent-sphere variation, while free diffusion suppresses all three.
The largest undiffused winding states require adaptive spatial derivatives
before quantitative `Q` claims. A subsequent physics branch should select a
small number of states for magnetofrictional or resistive-MHD evolution
rather than increasing this static parameter grid indefinitely.
"""


def main() -> None:
    args = parse_args()
    device = select_device(args.allow_cpu)
    dtype = torch.float32
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    seed_grid_size = args.seed_grid or (5 if args.smoke else 13)
    steps_per_cycle = args.steps_per_cycle or (48 if args.smoke else 128)
    output_dir = (
        args.output_dir
        or (SMOKE_OUTPUT_DIR if args.smoke else OUTPUT_DIR)
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configurations = parameter_grid(args.smoke)
    seeds = seed_grid(
        seed_grid_size,
        2.5,
        device=device,
        dtype=dtype,
    )
    pairs = neighbor_pairs(seed_grid_size, device=device)
    rows: list[dict[str, Any]] = []
    coarse_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    best_q_value = -math.inf
    best_trajectory: np.ndarray | None = None
    best_config: dict[str, Any] | None = None
    start = time.perf_counter()
    print(
        f"device={device} configs={len(configurations)} "
        f"lines_per_config={len(seeds)} steps_per_cycle={steps_per_cycle}"
    )

    for cycles in sorted({int(row["cycles"]) for row in configurations}):
        selected = [
            row for row in configurations if int(row["cycles"]) == cycles
        ]
        batch = len(selected)
        twist = tensor_values(selected, "twist", device)
        guide = tensor_values(selected, "guide_field", device)
        offset = tensor_values(selected, "center_offset", device)
        diffusion = tensor_values(selected, "diffusion_time", device)
        group_start = time.perf_counter()

        synchronize(device)
        integration_start = time.perf_counter()
        trajectory = integrate_line_tied_field_lines(
            seeds,
            cycles=cycles,
            twist=twist,
            guide_field=guide,
            center_offset=offset,
            diffusion_time=diffusion,
            steps_per_cycle=steps_per_cycle,
        )
        synchronize(device)
        integration_seconds = time.perf_counter() - integration_start
        if device.type == "cuda":
            assert trajectory.is_cuda
        if not bool(torch.all(torch.isfinite(trajectory))):
            raise AssertionError("field-line integration produced nonfinite values")

        diagnostics_start = time.perf_counter()
        endpoint = trajectory[:, -1, :, :2]
        unconstrained_squashing = squashing_factor_from_mapping(
            endpoint,
            grid_size=seed_grid_size,
            extent=2.5,
        )
        # B_z is the same positive constant on both line-tied boundaries, so
        # the exact normal-field ratio in the Q denominator is one.
        squashing = squashing_factor_from_mapping(
            endpoint,
            grid_size=seed_grid_size,
            extent=2.5,
            normal_field_ratio=1.0,
        )
        stretch = mapping_stretch_diagnostics(
            endpoint,
            grid_size=seed_grid_size,
            extent=2.5,
            axial_length=16.0 * cycles,
        )
        winding_fine = pairwise_winding(
            trajectory,
            pairs,
            coarse_factor=1,
        )
        direction_length_by_factor = {
            factor: field_direction_spherical_path_length(
                trajectory,
                cycles=cycles,
                twist=twist,
                guide_field=guide,
                center_offset=offset,
                diffusion_time=diffusion,
                coarse_factor=factor,
            )
            for factor in COARSE_FACTORS
        }
        direction_length_fine = direction_length_by_factor[1]
        parallel_current = integrated_parallel_current(
            trajectory,
            cycles=cycles,
            twist=twist,
            guide_field=guide,
            center_offset=offset,
            diffusion_time=diffusion,
        )
        points = volume_grid(
            cycles,
            batch=batch,
            device=device,
            dtype=dtype,
        )
        volume = volume_field_diagnostics(
            points,
            cycles=cycles,
            twist=twist,
            guide_field=guide,
            center_offset=offset,
            diffusion_time=diffusion,
        )
        divergence_points = points[:, :: max(1, points.shape[1] // 32)]
        divergence = finite_difference_divergence(
            divergence_points,
            cycles=cycles,
            twist=twist,
            guide_field=guide,
            center_offset=offset,
            diffusion_time=diffusion,
            step=2e-3,
        )
        displacement = torch.linalg.vector_norm(
            endpoint - seeds.unsqueeze(0),
            dim=-1,
        )
        synchronize(device)
        diagnostics_seconds = time.perf_counter() - diagnostics_start

        for index, config in enumerate(selected):
            q_values = squashing[index]
            unconstrained_q_values = unconstrained_squashing[index]
            winding_values = winding_fine[index]
            direction_lengths = direction_length_fine[index]
            parallel_values = parallel_current[index]
            row: dict[str, Any] = {
                **config,
                "field_line_count": seeds.shape[0],
                "integration_step_count": steps_per_cycle * cycles,
                "endpoint_displacement_mean": float(
                    displacement[index].mean().item()
                ),
                "endpoint_displacement_max": float(
                    displacement[index].max().item()
                ),
                "maximum_squashing_factor": float(q_values.max().item()),
                "p95_squashing_factor": float(
                    torch.quantile(q_values.flatten(), 0.95).item()
                ),
                "mean_log10_squashing_factor": float(
                    torch.log10(q_values).mean().item()
                ),
                "maximum_unconstrained_squashing_factor": float(
                    unconstrained_q_values.max().item()
                ),
                "maximum_mapping_stretch": float(
                    stretch["maximum_mapping_stretch"][index].item()
                ),
                "mean_log_mapping_stretch": float(
                    stretch["mean_log_mapping_stretch"][index].item()
                ),
                "maximum_finite_length_lyapunov": float(
                    stretch["maximum_finite_length_lyapunov"][index].item()
                ),
                "minimum_mapping_jacobian_abs_determinant": float(
                    stretch[
                        "minimum_mapping_jacobian_abs_determinant"
                    ][index].item()
                ),
                "mean_area_preservation_residual": float(
                    stretch["mean_area_preservation_residual"][index].item()
                ),
                "maximum_area_preservation_residual": float(
                    stretch["maximum_area_preservation_residual"][index].item()
                ),
                "mean_absolute_neighbor_winding": float(
                    winding_values.abs().mean().item()
                ),
                "maximum_absolute_neighbor_winding": float(
                    winding_values.abs().max().item()
                ),
                "neighbor_winding_over_quarter_turn_rate": float(
                    (winding_values.abs() >= 0.25).float().mean().item()
                ),
                "mean_direction_sphere_path_length": float(
                    direction_lengths.mean().item()
                ),
                "maximum_direction_sphere_path_length": float(
                    direction_lengths.max().item()
                ),
                "mean_integrated_parallel_current_abs": float(
                    parallel_values.abs().mean().item()
                ),
                "maximum_integrated_parallel_current_abs": float(
                    parallel_values.abs().max().item()
                ),
                "finite_trajectory": True,
                "maximum_numerical_divergence_abs": float(
                    divergence[index].abs().max().item()
                ),
            }
            for key, values in volume.items():
                row[key] = float(values[index].item())
            rows.append(row)

            if float(row["maximum_squashing_factor"]) > best_q_value:
                best_q_value = float(row["maximum_squashing_factor"])
                best_trajectory = trajectory[index].detach().cpu().numpy()
                best_config = row.copy()

            for factor in COARSE_FACTORS:
                winding_coarse = pairwise_winding(
                    trajectory[index : index + 1],
                    pairs,
                    coarse_factor=factor,
                )[0]
                error = (winding_coarse - winding_values).abs()
                direction_coarse = direction_length_by_factor[factor][index]
                direction_error = (
                    direction_lengths - direction_coarse
                ).clamp_min(0.0)
                coarse_rows.append(
                    {
                        "config_id": config["config_id"],
                        "cycles": cycles,
                        "diffusion_time": config["diffusion_time"],
                        "coarse_factor": factor,
                        "trajectory_samples": (
                            (trajectory.shape[1] - 1) // factor + 1
                        ),
                        "mean_absolute_winding": float(
                            winding_coarse.abs().mean().item()
                        ),
                        "mean_winding_error_from_fine": float(
                            error.mean().item()
                        ),
                        "maximum_winding_error_from_fine": float(
                            error.max().item()
                        ),
                        "mean_direction_sphere_path_length": float(
                            direction_coarse.mean().item()
                        ),
                        "mean_direction_path_length_loss": float(
                            direction_error.mean().item()
                        ),
                        "maximum_direction_path_length_loss": float(
                            direction_error.max().item()
                        ),
                    }
                )

        synchronize(device)
        total_group_seconds = time.perf_counter() - group_start
        timing_rows.append(
            {
                "cycles": cycles,
                "config_count": batch,
                "field_line_integration_seconds": integration_seconds,
                "diagnostics_seconds": diagnostics_seconds,
                "total_group_seconds": total_group_seconds,
                "integrated_line_steps": (
                    batch
                    * seeds.shape[0]
                    * steps_per_cycle
                    * cycles
                ),
            }
        )
        print(
            f"completed cycles={cycles} configs={batch} "
            f"integration={integration_seconds:.3f}s "
            f"diagnostics={diagnostics_seconds:.3f}s"
        )

    resolution_rows = (
        []
        if args.smoke
        else run_resolution_audit(
            rows,
            device=device,
            dtype=dtype,
            steps_per_cycle=steps_per_cycle,
            baseline_grid_size=seed_grid_size,
        )
    )
    synchronize(device)
    elapsed = time.perf_counter() - start
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    summaries = summarize_rows(rows)
    audit = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "actual_device": str(device),
        "gpu_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "cuda_used": device.type == "cuda",
        "dtype": str(dtype),
        "smoke": args.smoke,
        "configuration_count": len(configurations),
        "field_lines_per_config": int(seeds.shape[0]),
        "steps_per_cycle": steps_per_cycle,
        "total_field_line_steps": sum(
            int(row["integrated_line_steps"]) for row in timing_rows
        ),
        "peak_cuda_memory_bytes": peak_memory,
        "elapsed_seconds": elapsed,
        "full_resistive_mhd_solved": False,
        "free_diffusion_subflow_exact": True,
        "line_tied_boundaries": True,
        "specular_reflection_used_for_field_lines": False,
    }
    write_csv(output_dir / "magnetic_braid_mhd_runs.csv", rows)
    write_csv(
        output_dir / "magnetic_braid_mhd_coarse_graining.csv",
        coarse_rows,
    )
    write_csv(output_dir / "magnetic_braid_mhd_summary.csv", summaries)
    write_csv(output_dir / "magnetic_braid_mhd_timing.csv", timing_rows)
    if resolution_rows:
        write_csv(
            output_dir / "magnetic_braid_mhd_resolution_audit.csv",
            resolution_rows,
        )
    (output_dir / "device_report.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "device_report.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in audit.items()) + "\n",
        encoding="utf-8",
    )
    render_phase_diagram(rows, output_dir)
    if best_trajectory is None or best_config is None:
        raise AssertionError("no field-line visualization state was selected")
    render_field_lines(best_trajectory, best_config, output_dir)
    (output_dir / "MAGNETIC_BRAID_MHD_REPORT.md").write_text(
        make_report(
            rows,
            coarse_rows,
            summaries,
            timing_rows,
            audit,
            resolution_rows,
        ),
        encoding="utf-8",
    )
    print(
        f"wrote {len(rows)} configs in {elapsed:.3f}s "
        f"(peak_cuda_memory={peak_memory})"
    )


if __name__ == "__main__":
    main()
