"""CUDA sweep for curvature dissections of coarse-grained magnetic knots."""

from __future__ import annotations

import argparse
import csv
import json
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

from src.curvature_knot_four_color import (
    compactified_triangulation_faces,
    curvature_potential_surface,
    klein_flux_certificate,
)
from src.curvature_rg_flow import (
    effective_mode_count,
    geometric_curve_metrics,
    periodic_heat_coarse_grain,
    spectral_energies,
)
from src.fractional_field_four_color import (
    coloring_is_proper,
    compactified_triangulation_adjacency,
    curvature_triangulation_statistics,
    dsatur_coloring,
    interior_four_spin_correlation,
    mixed_curvature_diagonals,
    subsample_closed_window,
)
from src.maxwell_knot_fields import magnetic_core_curve


OUTPUT_DIR = ROOT / "results" / "curvature_knot_four_color_gpu"
SMOKE_OUTPUT_DIR = ROOT / "results" / "curvature_knot_four_color_gpu_smoke"
FIXTURES = (
    ("trefoil_T_2_3", 2, 3, 3),
    ("cinquefoil_T_2_5", 2, 5, 5),
    ("torus_T_3_4", 3, 4, 8),
)
CHANNELS = ("curvature_density", "signed_turning")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--samples", type=int)
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


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, float, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["fixture_id"]),
            str(row["channel"]),
            float(row["bandwidth"]),
            int(row["grid_size"]),
        )
        grouped.setdefault(key, []).append(row)
    summaries = []
    for key, selected in sorted(grouped.items()):
        selected.sort(key=lambda row: float(row["tau"]))
        bending = np.array(
            [float(row["spectral_bending_energy"]) for row in selected]
        )
        frustration = np.array(
            [float(row["odd_degree_vertex_fraction"]) for row in selected]
        )
        if np.std(bending) > 0.0 and np.std(frustration) > 0.0:
            correlation = float(
                np.corrcoef(np.log1p(bending), frustration)[0, 1]
            )
        else:
            correlation = float("nan")
        summaries.append(
            {
                "fixture_id": key[0],
                "channel": key[1],
                "bandwidth": key[2],
                "grid_size": key[3],
                "tau_count": len(selected),
                "four_chromatic_count": sum(
                    int(row["chromatic_number"]) == 4 for row in selected
                ),
                "three_chromatic_count": sum(
                    int(row["chromatic_number"]) == 3 for row in selected
                ),
                "initial_odd_degree_fraction": float(
                    selected[0]["odd_degree_vertex_fraction"]
                ),
                "final_odd_degree_fraction": float(
                    selected[-1]["odd_degree_vertex_fraction"]
                ),
                "mean_odd_degree_fraction": float(frustration.mean()),
                "final_initial_diagonal_mismatch": float(
                    selected[-1]["initial_diagonal_mismatch"]
                ),
                "maximum_initial_diagonal_mismatch": max(
                    float(row["initial_diagonal_mismatch"])
                    for row in selected
                ),
                "initial_four_spin_correlation": float(
                    selected[0]["interior_four_spin_correlation"]
                ),
                "final_four_spin_correlation": float(
                    selected[-1]["interior_four_spin_correlation"]
                ),
                "bending_frustration_correlation": correlation,
            }
        )
    return summaries


def summarize_scaling(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fit odd-defect count versus inverse mesh size at fixed geometry."""

    grouped: dict[
        tuple[str, str, float, float],
        list[dict[str, Any]],
    ] = {}
    for row in rows:
        key = (
            str(row["fixture_id"]),
            str(row["channel"]),
            float(row["bandwidth"]),
            float(row["tau"]),
        )
        grouped.setdefault(key, []).append(row)
    scaling_rows = []
    for key, selected in sorted(grouped.items()):
        selected.sort(key=lambda row: int(row["grid_size"]))
        mesh_counts = np.array(
            [int(row["grid_size"]) - 1 for row in selected],
            dtype=float,
        )
        odd_counts = np.array(
            [int(row["odd_degree_vertex_count"]) for row in selected],
            dtype=float,
        )
        slope, intercept = np.polyfit(
            np.log(mesh_counts),
            np.log(odd_counts),
            1,
        )
        fitted = intercept + slope * np.log(mesh_counts)
        residual = np.log(odd_counts) - fitted
        total = float(
            np.sum(
                (np.log(odd_counts) - np.log(odd_counts).mean()) ** 2
            )
        )
        r_squared = (
            1.0 - float(np.sum(residual**2)) / total
            if total > 0.0
            else 1.0
        )
        scaling_rows.append(
            {
                "fixture_id": key[0],
                "channel": key[1],
                "bandwidth": key[2],
                "tau": key[3],
                "grid_count": len(selected),
                "odd_count_scaling_exponent": float(slope),
                "odd_fraction_scaling_exponent": float(slope - 2.0),
                "log_log_r_squared": r_squared,
                "minimum_grid_size": int(selected[0]["grid_size"]),
                "maximum_grid_size": int(selected[-1]["grid_size"]),
            }
        )
    return scaling_rows


def render_colored_map(
    axis: Any,
    diagonals: np.ndarray,
    colors: list[int],
    title: str,
) -> None:
    grid_size = diagonals.shape[0] + 1
    palette = np.array(("#2563eb", "#dc2626", "#16a34a", "#f59e0b"))
    for row in range(grid_size):
        axis.plot(
            (0, grid_size - 1),
            (row, row),
            color="#b8c4d6",
            linewidth=0.55,
            zorder=1,
        )
    for column in range(grid_size):
        axis.plot(
            (column, column),
            (0, grid_size - 1),
            color="#b8c4d6",
            linewidth=0.55,
            zorder=1,
        )
    for row in range(grid_size - 1):
        for column in range(grid_size - 1):
            if diagonals[row, column]:
                axis.plot(
                    (column, column + 1),
                    (row, row + 1),
                    color="#94a3b8",
                    linewidth=0.55,
                )
            else:
                axis.plot(
                    (column, column + 1),
                    (row + 1, row),
                    color="#94a3b8",
                    linewidth=0.55,
                )
    grid_colors = np.array(colors[:-1]).reshape(grid_size, grid_size)
    x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    axis.scatter(
        x.ravel(),
        y.ravel(),
        c=palette[grid_colors.ravel()],
        s=13,
        zorder=3,
    )
    axis.set_aspect("equal")
    axis.invert_yaxis()
    axis.set_xticks(())
    axis.set_yticks(())
    axis.set_title(title)


def render_examples(
    visual_states: dict[tuple[str, str], dict[str, Any]],
    output_dir: Path,
) -> None:
    fixture_ids = [fixture[0] for fixture in FIXTURES if (fixture[0], "initial") in visual_states]
    figure, axes = plt.subplots(
        len(fixture_ids),
        2,
        figsize=(8.5, 3.9 * len(fixture_ids)),
        squeeze=False,
    )
    for row_index, fixture_id in enumerate(fixture_ids):
        for column_index, stage in enumerate(("initial", "coarse")):
            state = visual_states[(fixture_id, stage)]
            render_colored_map(
                axes[row_index, column_index],
                state["diagonals"],
                state["colors"],
                (
                    f"{fixture_id}, tau={state['tau']:g}\n"
                    f"exact chi={state['chromatic_number']}, "
                    f"odd={state['odd_fraction']:.3f}"
                ),
            )
    figure.suptitle(
        "Curvature-weighted magnetic-knot dissections\n"
        "(exterior compactification vertex omitted)"
    )
    figure.tight_layout()
    figure.savefig(
        output_dir / "curvature_knot_four_color_examples.png",
        dpi=190,
    )
    plt.close(figure)


def render_phase(
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    finest = max(int(row["grid_size"]) for row in rows)
    bandwidth = max(float(row["bandwidth"]) for row in rows)
    selected = [
        row
        for row in rows
        if int(row["grid_size"]) == finest
        and float(row["bandwidth"]) == bandwidth
    ]
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 9.2))
    colors = {
        "trefoil_T_2_3": "#2563eb",
        "cinquefoil_T_2_5": "#dc2626",
        "torus_T_3_4": "#16a34a",
    }
    styles = {"curvature_density": "-", "signed_turning": "--"}
    for fixture_id, _, _, _ in FIXTURES:
        for channel in CHANNELS:
            group = [
                row
                for row in selected
                if row["fixture_id"] == fixture_id and row["channel"] == channel
            ]
            if not group:
                continue
            group.sort(key=lambda row: float(row["tau"]))
            tau = [float(row["tau"]) for row in group]
            label = f"{fixture_id}, {channel}"
            kwargs = {
                "color": colors[fixture_id],
                "linestyle": styles[channel],
                "marker": "o",
                "markersize": 3,
                "label": label,
            }
            axes[0, 0].plot(
                tau,
                [float(row["odd_degree_vertex_fraction"]) for row in group],
                **kwargs,
            )
            axes[0, 1].plot(
                tau,
                [float(row["initial_diagonal_mismatch"]) for row in group],
                **kwargs,
            )
            axes[1, 0].plot(
                tau,
                [float(row["spectral_bending_energy"]) for row in group],
                **kwargs,
            )
            axes[1, 1].plot(
                tau,
                [
                    float(row["neighbor_diagonal_sign_agreement"])
                    for row in group
                ],
                **kwargs,
            )
    axes[0, 0].set_title("Four-color frustration")
    axes[0, 0].set_ylabel("odd-degree vertex fraction")
    axes[0, 1].set_title("RG transport from tau=0")
    axes[0, 1].set_ylabel("diagonal mismatch")
    axes[1, 0].set_title("Knot bending energy")
    axes[1, 0].set_ylabel("spectral bending energy")
    axes[1, 0].set_yscale("log")
    axes[1, 1].set_title("Curvature-map coherence")
    axes[1, 1].set_ylabel("neighbor diagonal agreement")
    for axis in axes.flat:
        axis.set_xscale("symlog", linthresh=1e-4)
        axis.set_xlim(left=0.0)
        axis.set_xlabel("heat-flow scale tau")
        axis.grid(True, alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=3,
        fontsize=8,
    )
    figure.suptitle(
        f"Magnetic knot curvature maps at grid {finest}, bandwidth {bandwidth:g}",
        y=0.99,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    figure.savefig(
        output_dir / "curvature_knot_four_color_phase.png",
        dpi=190,
    )
    plt.close(figure)


def render_defect_scaling(
    rows: list[dict[str, Any]],
    scaling_rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    bandwidth = max(float(row["bandwidth"]) for row in rows)
    initial = [
        row
        for row in rows
        if float(row["bandwidth"]) == bandwidth and float(row["tau"]) == 0.0
    ]
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    colors = {
        "trefoil_T_2_3": "#2563eb",
        "cinquefoil_T_2_5": "#dc2626",
        "torus_T_3_4": "#16a34a",
    }
    styles = {"curvature_density": "-", "signed_turning": "--"}
    for fixture_id, _, _, _ in FIXTURES:
        for channel in CHANNELS:
            group = [
                row
                for row in initial
                if row["fixture_id"] == fixture_id and row["channel"] == channel
            ]
            if not group:
                continue
            group.sort(key=lambda row: int(row["grid_size"]))
            axes[0].plot(
                [int(row["grid_size"]) - 1 for row in group],
                [int(row["odd_degree_vertex_count"]) for row in group],
                color=colors[fixture_id],
                linestyle=styles[channel],
                marker="o",
                label=f"{fixture_id}, {channel}",
            )
    mesh_reference = np.array((8.0, 64.0))
    axes[0].plot(
        mesh_reference,
        6.0 * mesh_reference,
        color="#111827",
        linestyle=":",
        label="reference slope 1",
    )
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("cells per axis")
    axes[0].set_ylabel("odd-degree vertex count")
    axes[0].set_title("Initial defect support across resolution")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=7)

    for fixture_id, _, _, _ in FIXTURES:
        for channel in CHANNELS:
            selected = [
                row
                for row in scaling_rows
                if row["fixture_id"] == fixture_id
                and row["channel"] == channel
                and float(row["bandwidth"]) == bandwidth
            ]
            selected.sort(key=lambda row: float(row["tau"]))
            axes[1].plot(
                [float(row["tau"]) for row in selected],
                [
                    float(row["odd_count_scaling_exponent"])
                    for row in selected
                ],
                color=colors[fixture_id],
                linestyle=styles[channel],
                marker="o",
                markersize=3,
            )
    axes[1].axhline(1.0, color="#111827", linestyle=":", label="curve support")
    axes[1].axhline(2.0, color="#64748b", linestyle=":", label="area support")
    axes[1].set_xscale("symlog", linthresh=1e-4)
    axes[1].set_xlim(left=0.0)
    axes[1].set_xlabel("heat-flow scale tau")
    axes[1].set_ylabel("fitted odd-count exponent")
    axes[1].set_title("Defect-support dimension proxy")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)
    figure.suptitle(
        "Four-color defects concentrate near curvature nodal sets"
    )
    figure.tight_layout()
    figure.savefig(
        output_dir / "curvature_knot_defect_scaling.png",
        dpi=190,
    )
    plt.close(figure)


def make_report(
    summaries: list[dict[str, Any]],
    scaling_rows: list[dict[str, Any]],
    flux_rows: list[dict[str, Any]],
    audit: dict[str, Any],
) -> str:
    finest = max(int(row["grid_size"]) for row in summaries)
    selected = [row for row in summaries if int(row["grid_size"]) == finest]
    largest_transport = max(
        selected,
        key=lambda row: float(row["final_initial_diagonal_mismatch"]),
    )
    smallest_transport = min(
        selected,
        key=lambda row: float(row["final_initial_diagonal_mismatch"]),
    )
    table = "\n".join(
        "| {fixture_id} | {channel} | {bandwidth:g} | {grid_size} | "
        "{four_chromatic_count}/{tau_count} | "
        "{initial_odd_degree_fraction:.3f} | "
        "{final_odd_degree_fraction:.3f} | "
        "{final_initial_diagonal_mismatch:.3f} |".format(**row)
        for row in summaries
    )
    flux_ok = all(
        bool(row["all_edge_fluxes_nonzero"])
        and bool(row["all_dual_vertices_conserved"])
        for row in flux_rows
    )
    exponents = np.array(
        [
            float(row["odd_count_scaling_exponent"])
            for row in scaling_rows
        ]
    )
    return f"""# Curvature-Knot Four Color GPU Report

## Question

Can Four Color diagnostics be attached to the curvature coarse graining of
closed magnetic knot cores in a way that records geometry rather than merely
coloring the regions of an ordinary knot diagram?

## Construction

The input curves are exact Bateman magnetic-core torus knots. The periodic
heat semigroup suppresses Fourier mode `k` by `exp(-tau k^2)`. At each RG
scale, a fixed generic projection is deposited onto the plane with either
integrated three-dimensional bending mass or signed projected turning as the
Gaussian-kernel weight.

The mixed curvature of that scalar potential chooses one diagonal in every
grid square. Adding one exterior vertex makes a sphere triangulation. The
map has exact chromatic number three exactly when all vertex degrees are
even; otherwise it has exact chromatic number four.

This is deliberately not ordinary knot-diagram face coloring, which is
checkerboard colorable and would make Four Color uninformative.

## Exact Discrete Interpretation

For interior vertices, odd degree is exactly the XOR of the four surrounding
diagonal choices. It is therefore a local `Z2` curvature defect.

Four vertex colors are encoded as `Z2 x Z2`. The color difference across
each primal edge is one of the three nonzero group elements. On the dual
cubic graph, the three incident labels at every triangular face XOR to zero.
The exported certificates therefore interpret every explicit four-coloring
as a nowhere-zero conserved Klein-four flux.

## Regular-Nodal-Set Bound

Let `g_h` be the continuous finite-difference mixed-curvature field whose
sign chooses the diagonals at mesh width `h`. An interior parity defect
requires both signs around that vertex, so a zero of `g_h` lies within an
`O(h)` neighborhood. If zero is a regular value and the nodal set has
uniformly bounded total length, a tubular-neighborhood count gives

`number of parity defects = O(h^(-1))`

and hence defect density `O(h)` on a two-dimensional grid. This is a direct
geometric consequence of the local XOR identity. It is conditional on the
regularity and uniform nodal-length assumptions.

## CUDA Audit

- Device: `{audit["actual_device"]}`
- GPU: `{audit["gpu_name"]}`
- CUDA used: `{audit["cuda_used"]}`
- Samples per knot: `{audit["samples"]}`
- RG scales: `{audit["tau_count"]}`
- Maps analyzed: `{audit["map_count"]}`
- Peak CUDA allocation: `{audit["peak_cuda_memory_bytes"]}` bytes
- Total elapsed: `{audit["elapsed_seconds"]:.3f}` seconds

## Results

| Knot | Dissection | Bandwidth | Grid | Four-color scales | Initial odd fraction | Final odd fraction | Final mismatch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

All `{len(flux_rows)}` representative Klein-flux certificates were nonzero
and conserved: `{flux_ok}`.

Across `{len(scaling_rows)}` fixed knot/channel/bandwidth/RG configurations,
the fitted odd-defect count exponent had mean `{float(exponents.mean()):.3f}`,
range `{float(exponents.min()):.3f}` to `{float(exponents.max()):.3f}`.
Curve-supported defects predict exponent one; area-filling defects predict
two. The fit supports, but does not certify, the regular-nodal-set picture.

The largest final diagonal displacement was
`{float(largest_transport["final_initial_diagonal_mismatch"]):.3f}` for
`{largest_transport["fixture_id"]}`,
`{largest_transport["channel"]}`, bandwidth
`{float(largest_transport["bandwidth"]):g}`. The smallest was
`{float(smallest_transport["final_initial_diagonal_mismatch"]):.3f}` for
`{smallest_transport["fixture_id"]}`,
`{smallest_transport["channel"]}`.

The coloring ceiling itself is not the differentiator: most generic
triangulations require four colors. The useful observables are defect
density, defect correlation, and transport of the curvature-selected
diagonals under coarse graining.

## What This Upgrades

1. Four colors become a `Z2 x Z2` potential whose differences define a
   conserved dual flux.
2. Odd-degree obstructions become local `Z2` curvature defects.
3. RG transport asks whether the same defect/flux organization survives
   when fine knot curvature is removed.

These are exact finite statements attached to declared discretizations.
They do not strengthen the Four Color Theorem itself.

## Claim Boundary

- The potential map is not an intrinsic invariant of a knot; it depends on
  projection, bandwidth, channel, grid, and compactification.
- Heat flow can leave the original knot type after a self-intersection.
- Four-colorability does not classify a knot or magnetic field.
- Klein-four labels are discrete graph flows, not physical magnetic flux.
- No new Four Color or Maxwell theorem is claimed.
"""


def main() -> None:
    args = parse_args()
    device = select_device(args.allow_cpu)
    smoke = args.smoke
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (SMOKE_OUTPUT_DIR if smoke else OUTPUT_DIR)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = args.samples or (128 if smoke else 512)
    fixtures = FIXTURES[:1] if smoke else FIXTURES
    tau_values = (
        (0.0, 0.01, 0.1)
        if smoke
        else (0.0, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
    )
    # The 9x9 level keeps explicit DSATUR/Klein-flow certificates bounded.
    # Finer levels remain in the statistical sweep.
    grid_sizes = (9, 17) if smoke else (9, 17, 33, 65)
    bandwidths = (0.16,) if smoke else (0.10, 0.18)
    dtype = torch.float64
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    rows: list[dict[str, Any]] = []
    flux_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    visual_states: dict[tuple[str, str], dict[str, Any]] = {}
    baselines: dict[tuple[str, str, float, int], torch.Tensor] = {}
    start = time.perf_counter()
    print(
        f"device={device} fixtures={len(fixtures)} samples={samples} "
        f"tau_count={len(tau_values)}"
    )

    for fixture_id, p, q, minimal_crossing in fixtures:
        initial = magnetic_core_curve(
            p,
            q,
            samples,
            device=device,
            dtype=dtype,
        )
        assert initial.device.type == device.type
        for tau_index, tau in enumerate(tau_values):
            group_start = time.perf_counter()
            curve = periodic_heat_coarse_grain(
                initial,
                tau,
                normalize=True,
            )
            geometry = geometric_curve_metrics(curve)
            spectral = spectral_energies(curve)
            mode_count = effective_mode_count(curve)
            for channel in CHANNELS:
                for bandwidth in bandwidths:
                    surface = curvature_potential_surface(
                        curve,
                        max(grid_sizes),
                        bandwidth,
                        channel,
                    )
                    if device.type == "cuda":
                        assert surface.is_cuda
                    for grid_size in grid_sizes:
                        stride = (max(grid_sizes) - 1) // (grid_size - 1)
                        window = subsample_closed_window(
                            surface.unsqueeze(0),
                            stride,
                        )
                        diagonals = mixed_curvature_diagonals(window)
                        statistics = curvature_triangulation_statistics(window)
                        spin = interior_four_spin_correlation(diagonals)
                        key = (fixture_id, channel, bandwidth, grid_size)
                        if tau_index == 0:
                            baselines[key] = diagonals.detach().clone()
                        mismatch = (
                            diagonals != baselines[key]
                        ).to(torch.float64).mean()
                        row = {
                            "fixture_id": fixture_id,
                            "p": p,
                            "q": q,
                            "minimal_crossing_number": minimal_crossing,
                            "tau_index": tau_index,
                            "tau": tau,
                            "channel": channel,
                            "bandwidth": bandwidth,
                            "grid_size": grid_size,
                            "vertex_count": int(statistics["vertex_count"]),
                            "edge_count": int(statistics["edge_count"]),
                            "chromatic_number": int(
                                statistics["chromatic_number"][0].item()
                            ),
                            "odd_degree_vertex_count": int(
                                statistics["odd_degree_vertex_count"][0].item()
                            ),
                            "odd_degree_vertex_fraction": float(
                                statistics[
                                    "odd_degree_vertex_fraction"
                                ][0].item()
                            ),
                            "neighbor_diagonal_sign_agreement": float(
                                statistics[
                                    "neighbor_diagonal_sign_agreement"
                                ][0].item()
                            ),
                            "interior_four_spin_correlation": float(
                                spin[0].item()
                            ),
                            "initial_diagonal_mismatch": float(mismatch.item()),
                            "effective_mode_count": mode_count,
                            **geometry,
                            **spectral,
                        }
                        rows.append(row)

                        representative = (
                            grid_size == min(grid_sizes)
                            and tau_index in (0, len(tau_values) - 1)
                        )
                        if representative:
                            diagonal_cpu = diagonals[0].detach().cpu()
                            adjacency = compactified_triangulation_adjacency(
                                diagonal_cpu
                            )
                            chromatic = int(row["chromatic_number"])
                            colors = dsatur_coloring(adjacency, chromatic)
                            if not coloring_is_proper(adjacency, colors):
                                raise AssertionError("improper coloring")
                            faces = compactified_triangulation_faces(
                                diagonal_cpu
                            )
                            flux_rows.append(
                                {
                                    "fixture_id": fixture_id,
                                    "tau": tau,
                                    "channel": channel,
                                    "bandwidth": bandwidth,
                                    "grid_size": grid_size,
                                    "chromatic_number": chromatic,
                                    **klein_flux_certificate(faces, colors),
                                }
                            )
                            visual_choice = (
                                channel == "curvature_density"
                                and bandwidth == max(bandwidths)
                            )
                            if visual_choice:
                                stage = "initial" if tau_index == 0 else "coarse"
                                visual_states[(fixture_id, stage)] = {
                                    "tau": tau,
                                    "diagonals": diagonal_cpu.numpy(),
                                    "colors": colors,
                                    "chromatic_number": chromatic,
                                    "odd_fraction": row[
                                        "odd_degree_vertex_fraction"
                                    ],
                                }
            synchronize(device)
            timing_rows.append(
                {
                    "fixture_id": fixture_id,
                    "tau": tau,
                    "maps_written": len(CHANNELS)
                    * len(bandwidths)
                    * len(grid_sizes),
                    "elapsed_seconds": time.perf_counter() - group_start,
                }
            )
        print(f"completed {fixture_id}")

    synchronize(device)
    elapsed = time.perf_counter() - start
    summaries = summarize(rows)
    scaling_rows = summarize_scaling(rows)
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
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
            else "not used"
        ),
        "cuda_used": device.type == "cuda",
        "dtype": str(dtype),
        "samples": samples,
        "fixtures": [fixture[0] for fixture in fixtures],
        "tau_count": len(tau_values),
        "grid_sizes": list(grid_sizes),
        "bandwidths": list(bandwidths),
        "channels": list(CHANNELS),
        "map_count": len(rows),
        "flux_certificate_count": len(flux_rows),
        "peak_cuda_memory_bytes": peak_memory,
        "elapsed_seconds": elapsed,
        "new_four_color_theorem_claimed": False,
        "physical_flux_identity_claimed": False,
    }
    write_csv(output_dir / "curvature_knot_four_color_rows.csv", rows)
    write_csv(output_dir / "curvature_knot_four_color_summary.csv", summaries)
    write_csv(
        output_dir / "curvature_knot_defect_scaling.csv",
        scaling_rows,
    )
    write_csv(output_dir / "curvature_knot_klein_flux_certificates.csv", flux_rows)
    write_csv(output_dir / "curvature_knot_four_color_timing.csv", timing_rows)
    (output_dir / "device_report.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "device_report.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in audit.items()) + "\n",
        encoding="utf-8",
    )
    render_phase(rows, output_dir)
    render_defect_scaling(rows, scaling_rows, output_dir)
    render_examples(visual_states, output_dir)
    (output_dir / "CURVATURE_KNOT_FOUR_COLOR_REPORT.md").write_text(
        make_report(summaries, scaling_rows, flux_rows, audit),
        encoding="utf-8",
    )
    print(
        f"wrote {len(rows)} maps in {elapsed:.3f}s "
        f"(peak_cuda_memory={peak_memory})"
    )


if __name__ == "__main__":
    main()
