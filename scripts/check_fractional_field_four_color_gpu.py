"""CUDA sweep for fractional-field curvature triangulations and Four Color."""

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

from src.fractional_field_four_color import (
    coloring_is_proper,
    compactified_triangulation_adjacency,
    compactified_triangulation_degrees,
    curvature_triangulation_statistics,
    diagonal_majority_transport_mismatch,
    dsatur_coloring,
    estimate_hurst_from_structure_function,
    fractional_gaussian_surfaces,
    mixed_curvature_diagonals,
    periodic_closed_window,
    subsample_closed_window,
)


OUTPUT_DIR = ROOT / "results" / "fractional_field_four_color_gpu"
SMOKE_OUTPUT_DIR = ROOT / "results" / "fractional_field_four_color_gpu_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--resolution", type=int)
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


def exact_fixture_rows() -> list[dict[str, Any]]:
    row = torch.tensor((True, False, True, False))
    checkerboard = torch.stack((row, ~row, row, ~row))
    flipped = checkerboard.clone()
    flipped[0, 0] = ~flipped[0, 0]
    rows = []
    for fixture_id, diagonals, expected in (
        ("eulerian_checkerboard", checkerboard, 3),
        ("single_curvature_flip", flipped, 4),
    ):
        degrees = compactified_triangulation_degrees(
            diagonals.unsqueeze(0)
        )[0]
        adjacency = compactified_triangulation_adjacency(diagonals)
        colors = dsatur_coloring(adjacency, expected)
        rows.append(
            {
                "fixture_id": fixture_id,
                "grid_size": 5,
                "vertex_count": len(adjacency),
                "edge_count": sum(map(len, adjacency)) // 2,
                "odd_degree_vertex_count": int((degrees % 2).sum()),
                "expected_chromatic_number": expected,
                "color_count_used": max(colors) + 1,
                "proper_coloring": coloring_is_proper(adjacency, colors),
                "sphere_triangulation_edge_identity": (
                    sum(map(len, adjacency)) // 2
                    == 3 * len(adjacency) - 6
                ),
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (float(row["hurst"]), int(row["grid_size"])),
            [],
        ).append(row)
    summaries = []
    fields = (
        "estimated_hurst",
        "odd_degree_vertex_fraction",
        "main_diagonal_fraction",
        "neighbor_diagonal_sign_agreement",
        "mean_absolute_mixed_curvature",
    )
    for (hurst, grid_size), selected in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "hurst": hurst,
            "grid_size": grid_size,
            "sample_count": len(selected),
            "four_chromatic_count": sum(
                int(row["chromatic_number"]) == 4 for row in selected
            ),
            "four_chromatic_rate": float(
                np.mean(
                    [
                        int(row["chromatic_number"]) == 4
                        for row in selected
                    ]
                )
            ),
        }
        for field in fields:
            values = np.array([float(row[field]) for row in selected])
            summary[f"mean_{field}"] = float(values.mean())
            summary[f"std_{field}"] = float(values.std())
        summaries.append(summary)
    return summaries


def summarize_transport(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, int, int], list[float]] = {}
    for row in rows:
        key = (
            float(row["hurst"]),
            int(row["coarse_grid_size"]),
            int(row["fine_grid_size"]),
        )
        grouped.setdefault(key, []).append(float(row["mismatch_rate"]))
    return [
        {
            "hurst": key[0],
            "coarse_grid_size": key[1],
            "fine_grid_size": key[2],
            "sample_count": len(values),
            "mean_mismatch_rate": float(np.mean(values)),
            "std_mismatch_rate": float(np.std(values)),
            "maximum_mismatch_rate": float(np.max(values)),
        }
        for key, values in sorted(grouped.items())
    ]


def render_sample(
    window: np.ndarray,
    diagonals: np.ndarray,
    colors: list[int],
    chromatic_number: int,
    hurst: float,
    output_dir: Path,
) -> None:
    grid_size = window.shape[0]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    image = axes[0].imshow(window, cmap="coolwarm", origin="lower")
    axes[0].set_title(f"cutoff fractional surface, H={hurst:g}")
    figure.colorbar(image, ax=axes[0], shrink=0.82)
    axes[0].set_xlabel("periodic x")
    axes[0].set_ylabel("periodic y")

    for index in range(grid_size):
        axes[1].plot(
            (0, grid_size - 1),
            (index, index),
            color="#cbd5e1",
            linewidth=0.35,
        )
        axes[1].plot(
            (index, index),
            (0, grid_size - 1),
            color="#cbd5e1",
            linewidth=0.35,
        )
    for row in range(grid_size - 1):
        for column in range(grid_size - 1):
            if bool(diagonals[row, column]):
                axes[1].plot(
                    (column, column + 1),
                    (row, row + 1),
                    color="#94a3b8",
                    linewidth=0.5,
                )
            else:
                axes[1].plot(
                    (column + 1, column),
                    (row, row + 1),
                    color="#94a3b8",
                    linewidth=0.5,
                )
    palette = np.array(("#2563eb", "#dc2626", "#16a34a", "#f59e0b"))
    grid_colors = np.array(colors[:-1]).reshape(grid_size, grid_size)
    x_values, y_values = np.meshgrid(
        np.arange(grid_size),
        np.arange(grid_size),
    )
    axes[1].scatter(
        x_values.reshape(-1),
        y_values.reshape(-1),
        c=palette[grid_colors].reshape(-1),
        s=19,
        edgecolor="white",
        linewidth=0.25,
        zorder=3,
    )
    axes[1].set_aspect("equal")
    axes[1].set_title(
        f"curvature triangulation: exact chi={chromatic_number}\n"
        "(exterior compactification vertex omitted)"
    )
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    figure.tight_layout()
    figure.savefig(
        output_dir / "fractional_curvature_four_coloring.png",
        dpi=190,
    )
    plt.close(figure)


def render_summary(
    summaries: list[dict[str, Any]],
    transport: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    grid_sizes = sorted({int(row["grid_size"]) for row in summaries})
    for grid_size in grid_sizes:
        selected = [
            row for row in summaries if int(row["grid_size"]) == grid_size
        ]
        hurst = [float(row["hurst"]) for row in selected]
        axes[0].plot(
            hurst,
            [float(row["mean_odd_degree_vertex_fraction"]) for row in selected],
            marker="o",
            label=f"{grid_size}x{grid_size}",
        )
        axes[1].plot(
            hurst,
            [
                float(row["mean_neighbor_diagonal_sign_agreement"])
                for row in selected
            ],
            marker="o",
            label=f"{grid_size}x{grid_size}",
        )
    finest_pair = max(
        {
            (int(row["coarse_grid_size"]), int(row["fine_grid_size"]))
            for row in transport
        },
        key=lambda pair: pair[1] - pair[0],
    )
    selected_transport = [
        row
        for row in transport
        if (
            int(row["coarse_grid_size"]),
            int(row["fine_grid_size"]),
        )
        == finest_pair
    ]
    axes[2].plot(
        [float(row["hurst"]) for row in selected_transport],
        [float(row["mean_mismatch_rate"]) for row in selected_transport],
        marker="o",
        color="#7c3aed",
    )
    axes[0].set_title("Four-color frustration")
    axes[0].set_xlabel("Hurst parameter H")
    axes[0].set_ylabel("odd-degree vertex fraction")
    axes[1].set_title("curvature-sign coherence")
    axes[1].set_xlabel("Hurst parameter H")
    axes[1].set_ylabel("neighbor diagonal agreement")
    axes[2].set_title(
        f"scale transport {finest_pair[0]} to {finest_pair[1]}"
    )
    axes[2].set_xlabel("Hurst parameter H")
    axes[2].set_ylabel("coarse/majority-fine mismatch")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].legend()
    figure.suptitle("Fractional-field curvature and finite Four Color maps")
    figure.tight_layout()
    figure.savefig(
        output_dir / "fractional_four_color_phase_diagram.png",
        dpi=190,
    )
    plt.close(figure)


def make_report(
    summaries: list[dict[str, Any]],
    transport: list[dict[str, Any]],
    fixtures: list[dict[str, Any]],
    timing: list[dict[str, Any]],
    audit: dict[str, Any],
) -> str:
    finest_grid = max(int(row["grid_size"]) for row in summaries)
    finest = [
        row for row in summaries if int(row["grid_size"]) == finest_grid
    ]
    rough = min(finest, key=lambda row: float(row["hurst"]))
    smooth = max(finest, key=lambda row: float(row["hurst"]))
    largest_transport = max(
        transport,
        key=lambda row: float(row["mean_mismatch_rate"]),
    )
    summary_table = "\n".join(
        "| {hurst:g} | {grid_size} | {mean_estimated_hurst:.3f} | "
        "{mean_odd_degree_vertex_fraction:.4f} | "
        "{mean_neighbor_diagonal_sign_agreement:.4f} | "
        "{four_chromatic_rate:.4f} |".format(**row)
        for row in summaries
    )
    fixture_table = "\n".join(
        "| {fixture_id} | {odd_degree_vertex_count} | "
        "{expected_chromatic_number} | {proper_coloring} |".format(**row)
        for row in fixtures
    )
    return f"""# Fractional-Field Four Color GPU Report

## Question

Can the regularity parameter of a fractional Gaussian field generate a
natural family of finite planar maps on which Four Color has a measurable,
nontrivial role?

This experiment uses the Cao-Sheffield convention
`FGF_s(R^2) = (-Delta)^(-s/2) W`, with `H = s - 1`. A finite Fourier cutoff
produces a continuous lattice surface. It is not an infinite coloring and it
is not a sample of a pointwise field when the continuum object exists only
as a generalized function.

## Curvature-to-Map Construction

For every lattice square, the sign of

`h00 + h11 - h10 - h01`

selects one of its two diagonals. The square window is compactified by adding
one exterior vertex connected to its boundary cycle. The result is a sphere
triangulation with the exact identity `E = 3V - 6`.

A sphere triangulation is vertex-3-colorable exactly when every vertex has
even degree. Because every face is a triangle, any non-Eulerian instance
needs at least four colors; the Four Color Theorem supplies the matching
upper bound. Thus each finite instance has exact chromatic number three or
four without treating a rendered color palette as evidence.

## Exact Fixtures

| Fixture | Odd vertices | Exact chromatic number | Explicit coloring proper |
| --- | ---: | ---: | --- |
{fixture_table}

## CUDA Audit

- Device: `{audit["actual_device"]}`
- GPU: `{audit["gpu_name"]}`
- CUDA used: `{audit["cuda_used"]}`
- Dtype: `{audit["dtype"]}`
- Samples per H: `{audit["batch_size"]}`
- Periodic Fourier resolution: `{audit["periodic_resolution"]}`
- H values: `{audit["hurst_values"]}`
- Peak CUDA allocation: `{audit["peak_cuda_memory_bytes"]}` bytes
- Total elapsed: `{audit["elapsed_seconds"]:.3f}` seconds

## Results

| H | Grid | Estimated H | Odd-degree fraction | Neighbor sign agreement | Four-chromatic rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
{summary_table}

At the finest grid, changing `H` from `{float(rough["hurst"]):g}` to
`{float(smooth["hurst"]):g}` changed curvature-sign agreement from
`{float(rough["mean_neighbor_diagonal_sign_agreement"]):.4f}` to
`{float(smooth["mean_neighbor_diagonal_sign_agreement"]):.4f}` and the
odd-degree fraction from
`{float(rough["mean_odd_degree_vertex_fraction"]):.4f}` to
`{float(smooth["mean_odd_degree_vertex_fraction"]):.4f}`.

The largest mean disagreement between a coarse diagonal and the majority of
its nested fine diagonals was
`{float(largest_transport["mean_mismatch_rate"]):.4f}` for
`H={float(largest_transport["hurst"]):g}` between grids
`{largest_transport["coarse_grid_size"]}` and
`{largest_transport["fine_grid_size"]}`.

The new finite statistic is the **odd-degree or Four-color frustration
density**. It measures how often local curvature choices obstruct an
Eulerian/three-color triangulation. Four Color gives a uniform ceiling, while
`H` and scale control the density and persistence of the obstructions.

## Relation to Fractional Forms and Flux Tubes

Cao and Sheffield define fractional Gaussian differential forms, their
curl-free/divergence-free projections, lattice versions, and restrictions.
Their Chern-Simons discussion also recovers the Gauss linking integral from
the quadratic form `(J, curl^(-1) J)` for divergence-free currents.

This run uses only scalar cutoff 0-forms to generate planar triangulations.
A future magnetic branch can add a divergence-free fractional 1-form to the
analytic flux-tube field, then apply the same finite coloring construction to
transverse slices. That would connect random roughness, flux-tube winding,
and dynamic four-color transport without identifying colors with gauge
states.

## Claim Boundary

- `H` is a scaling/regularity parameter, not a chromatic or fractal dimension
  by itself.
- The continuum field is not literally an infinite Four Color map.
- Four Color applies to each finite compactified triangulation. It does not
  provide a canonical, measurable, or scale-consistent limiting coloring.
- Curvature-selected diagonals are one declared discretization, not an
  intrinsic decomposition forced by the continuum field.
- This is not a new Four Color theorem, gauge-theory theorem, or result about
  the Hausdorff dimension of fractional-field graphs.

## Performance

Mean per-H GPU group time was
`{float(np.mean([float(row["total_seconds"]) for row in timing])):.3f}`
seconds. Only finite summaries and one visualization fixture were
transferred to CPU.

## Source

Sky Cao and Scott Sheffield, *Fractional Gaussian Forms and Gauge Theory: An
Overview*: https://arxiv.org/abs/2406.19321
"""


def main() -> None:
    args = parse_args()
    device = select_device(args.allow_cpu)
    dtype = torch.float32
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    batch_size = args.batch_size or (16 if args.smoke else 256)
    periodic_resolution = args.resolution or (32 if args.smoke else 64)
    if periodic_resolution % 4:
        raise ValueError("resolution must be divisible by four")
    hurst_values = (
        (0.25, 0.75)
        if args.smoke
        else (0.1, 0.25, 0.5, 0.75, 0.9)
    )
    strides = (4, 2, 1)
    shifts = (1, 2, 4, 8)
    output_dir = (
        args.output_dir
        or (SMOKE_OUTPUT_DIR if args.smoke else OUTPUT_DIR)
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    transport_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    visual_state: tuple[np.ndarray, np.ndarray, float, int] | None = None
    start = time.perf_counter()
    print(
        f"device={device} batch={batch_size} "
        f"resolution={periodic_resolution} H={hurst_values}"
    )

    for hurst_index, hurst in enumerate(hurst_values):
        synchronize(device)
        group_start = time.perf_counter()
        generation_start = time.perf_counter()
        surfaces = fractional_gaussian_surfaces(
            batch_size,
            periodic_resolution,
            hurst,
            seed=7301 + 101 * hurst_index,
            device=device,
            dtype=dtype,
        )
        if device.type == "cuda":
            assert surfaces.is_cuda
        estimated_hurst = estimate_hurst_from_structure_function(
            surfaces,
            shifts,
        )
        closed = periodic_closed_window(surfaces)
        synchronize(device)
        generation_seconds = time.perf_counter() - generation_start

        analysis_start = time.perf_counter()
        diagonals_by_grid: dict[int, Any] = {}
        statistics_by_grid: dict[int, dict[str, Any]] = {}
        windows_by_grid: dict[int, Any] = {}
        for stride in strides:
            window = subsample_closed_window(closed, stride)
            grid_size = int(window.shape[-1])
            windows_by_grid[grid_size] = window
            diagonals = mixed_curvature_diagonals(window)
            diagonals_by_grid[grid_size] = diagonals
            statistics = curvature_triangulation_statistics(window)
            statistics_by_grid[grid_size] = statistics
            cpu_values = {
                key: value.detach().cpu().numpy()
                for key, value in statistics.items()
                if isinstance(value, torch.Tensor)
            }
            estimated_cpu = estimated_hurst.detach().cpu().numpy()
            for sample_id in range(batch_size):
                rows.append(
                    {
                        "hurst": hurst,
                        "sample_id": sample_id,
                        "grid_size": grid_size,
                        "periodic_resolution": periodic_resolution,
                        "estimated_hurst": float(estimated_cpu[sample_id]),
                        "vertex_count": int(statistics["vertex_count"]),
                        "edge_count": int(statistics["edge_count"]),
                        "odd_degree_vertex_count": int(
                            cpu_values["odd_degree_vertex_count"][sample_id]
                        ),
                        "odd_degree_vertex_fraction": float(
                            cpu_values[
                                "odd_degree_vertex_fraction"
                            ][sample_id]
                        ),
                        "chromatic_number": int(
                            cpu_values["chromatic_number"][sample_id]
                        ),
                        "main_diagonal_fraction": float(
                            cpu_values["main_diagonal_fraction"][sample_id]
                        ),
                        "neighbor_diagonal_sign_agreement": float(
                            cpu_values[
                                "neighbor_diagonal_sign_agreement"
                            ][sample_id]
                        ),
                        "mean_absolute_mixed_curvature": float(
                            cpu_values[
                                "mean_absolute_mixed_curvature"
                            ][sample_id]
                        ),
                    }
                )

        grid_sizes = sorted(diagonals_by_grid)
        for coarse_index, coarse_grid in enumerate(grid_sizes):
            for fine_grid in grid_sizes[coarse_index + 1 :]:
                mismatch = diagonal_majority_transport_mismatch(
                    diagonals_by_grid[coarse_grid],
                    diagonals_by_grid[fine_grid],
                ).detach().cpu().numpy()
                for sample_id, value in enumerate(mismatch):
                    transport_rows.append(
                        {
                            "hurst": hurst,
                            "sample_id": sample_id,
                            "coarse_grid_size": coarse_grid,
                            "fine_grid_size": fine_grid,
                            "mismatch_rate": float(value),
                        }
                    )

        if visual_state is None and hurst >= 0.5:
            visual_grid = grid_sizes[0]
            visual_state = (
                windows_by_grid[visual_grid][0].detach().cpu().numpy(),
                diagonals_by_grid[visual_grid][0].detach().cpu().numpy(),
                hurst,
                int(
                    statistics_by_grid[visual_grid]["chromatic_number"][0]
                ),
            )
        synchronize(device)
        analysis_seconds = time.perf_counter() - analysis_start
        total_seconds = time.perf_counter() - group_start
        timing_rows.append(
            {
                "hurst": hurst,
                "sample_count": batch_size,
                "generation_and_hurst_seconds": generation_seconds,
                "triangulation_and_transport_seconds": analysis_seconds,
                "total_seconds": total_seconds,
            }
        )
        print(
            f"completed H={hurst:g} generation={generation_seconds:.3f}s "
            f"analysis={analysis_seconds:.3f}s"
        )

    synchronize(device)
    elapsed = time.perf_counter() - start
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    fixtures = exact_fixture_rows()
    summaries = summarize(rows)
    transport_summaries = summarize_transport(transport_rows)
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
        "batch_size": batch_size,
        "periodic_resolution": periodic_resolution,
        "hurst_values": list(hurst_values),
        "grid_sizes": sorted({int(row["grid_size"]) for row in rows}),
        "peak_cuda_memory_bytes": peak_memory,
        "elapsed_seconds": elapsed,
        "continuum_infinite_coloring_constructed": False,
        "finite_planar_triangulations_certified": True,
    }
    write_csv(output_dir / "fractional_four_color_runs.csv", rows)
    write_csv(output_dir / "fractional_four_color_summary.csv", summaries)
    write_csv(
        output_dir / "fractional_four_color_transport_rows.csv",
        transport_rows,
    )
    write_csv(
        output_dir / "fractional_four_color_transport_summary.csv",
        transport_summaries,
    )
    write_csv(output_dir / "fractional_four_color_fixtures.csv", fixtures)
    write_csv(output_dir / "fractional_four_color_timing.csv", timing_rows)
    (output_dir / "device_report.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "device_report.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in audit.items()) + "\n",
        encoding="utf-8",
    )
    render_summary(summaries, transport_summaries, output_dir)
    if visual_state is None:
        raise AssertionError("no visualization state selected")
    visual_window, visual_diagonals, visual_hurst, visual_chromatic = (
        visual_state
    )
    adjacency = compactified_triangulation_adjacency(
        torch.from_numpy(visual_diagonals)
    )
    colors = dsatur_coloring(adjacency, 4)
    if not coloring_is_proper(adjacency, colors):
        raise AssertionError("explicit visualization coloring is invalid")
    render_sample(
        visual_window,
        visual_diagonals,
        colors,
        visual_chromatic,
        visual_hurst,
        output_dir,
    )
    (output_dir / "FRACTIONAL_FIELD_FOUR_COLOR_REPORT.md").write_text(
        make_report(
            summaries,
            transport_summaries,
            fixtures,
            timing_rows,
            audit,
        ),
        encoding="utf-8",
    )
    print(
        f"wrote {len(rows)} finite maps in {elapsed:.3f}s "
        f"(peak_cuda_memory={peak_memory})"
    )


if __name__ == "__main__":
    main()
