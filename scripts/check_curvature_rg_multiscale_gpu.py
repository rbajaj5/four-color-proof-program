"""Resolve the curvature RG experiment into macro/meso/micro spectral bands."""

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

from scripts.check_curvature_rg_knot_collapse_gpu import (
    FIXTURES,
    OUTPUT_DIR,
    select_device,
    synchronize,
    write_csv,
)
from src.curvature_rg_flow import (
    periodic_heat_coarse_grain,
    spectral_scale_bands,
)
from src.maxwell_knot_fields import magnetic_core_curve


COLLAPSE_FRACTION = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--tau-count", type=int, default=48)
    parser.add_argument("--tau-max", type=float, default=5.0)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def projection_fall_times(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    result: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row["below_minimal_crossing"].lower() == "true"
                and row["fixture_id"] not in result
            ):
                result[row["fixture_id"]] = float(row["tau"])
    return result


def collapse_interval(
    rows: list[dict[str, Any]],
    threshold: float,
) -> tuple[str, float | None]:
    collapsed = [
        row
        for row in rows
        if float(row["retained_bending_fraction"]) <= threshold
    ]
    if not collapsed:
        return f"not_seen_through_{float(rows[-1]['tau']):g}", None
    first = collapsed[0]
    index = int(first["tau_index"])
    lower = float(rows[max(0, index - 1)]["tau"])
    upper = float(first["tau"])
    return f"({lower:.6g}, {upper:.6g}]", upper


def render_scale_cascade(
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    scales = ("macroscopic", "mesoscopic", "microscopic")
    colors = {
        "hopf_core_unknot": "#00876c",
        "trefoil_2_3": "#0057b8",
        "cinquefoil_2_5": "#d81b60",
        "torus_knot_3_4": "#e67e22",
    }
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.2))
    for column, scale in enumerate(scales):
        retention_axis = axes[0, column]
        share_axis = axes[1, column]
        for fixture_id, _, _, _ in FIXTURES:
            selected = [
                row
                for row in rows
                if row["fixture_id"] == fixture_id and row["scale"] == scale
            ]
            tau = [max(float(row["tau"]), 1e-5) for row in selected]
            retention_axis.plot(
                tau,
                [
                    max(float(row["retained_bending_fraction"]), 1e-16)
                    for row in selected
                ],
                color=colors[fixture_id],
                label=fixture_id,
            )
            share_axis.plot(
                tau,
                [float(row["current_bending_fraction"]) for row in selected],
                color=colors[fixture_id],
                label=fixture_id,
            )
        retention_axis.axhline(
            COLLAPSE_FRACTION,
            color="#333333",
            linestyle="--",
            linewidth=1,
        )
        retention_axis.set_xscale("log")
        retention_axis.set_yscale("log")
        retention_axis.set_title(scale)
        retention_axis.grid(True, alpha=0.25)
        share_axis.set_xscale("log")
        share_axis.set_ylim(-0.03, 1.03)
        share_axis.set_xlabel("RG time tau")
        share_axis.grid(True, alpha=0.25)
    axes[0, 0].set_ylabel("absolute bending retention")
    axes[1, 0].set_ylabel("relative bending share")
    axes[0, 0].legend(fontsize=8)
    figure.suptitle(
        "Curvature RG Jenga cascade: depletion versus structural share"
    )
    figure.tight_layout()
    figure.savefig(output_dir / "curvature_rg_scale_cascade.png", dpi=180)
    plt.close(figure)


def format_measure(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def make_report(
    summaries: list[dict[str, Any]],
    audit: dict[str, Any],
) -> str:
    table_rows = []
    for summary in summaries:
        display = {
            **summary,
            "retained_at_projection_fall": format_measure(
                summary["retained_at_projection_fall"]
            ),
            "bending_share_at_projection_fall": format_measure(
                summary["bending_share_at_projection_fall"]
            ),
        }
        table_rows.append(
            "| {fixture_id} | {scale} | {mode_range} | "
            "{initial_bending_fraction:.4f} | "
            "{one_percent_collapse_interval} | "
            "{retained_at_projection_fall} | "
            "{bending_share_at_projection_fall} | "
            "{collapse_before_projection_fall} |".format(
                **display
            )
        )
    table = "\n".join(table_rows)
    return f"""# Curvature RG Multiscale Jenga Report

## Discretization

The 512-vertex curve is resolved into Fourier blocks and then grouped by
wavelength relative to one traversal of the closed curve:

- **macroscopic:** modes 1-2, controlling the global loop and terminal
  harmonic;
- **mesoscopic:** modes 3-8, controlling lobes, crossing corridors, and the
  recognizable knot diagram; and
- **microscopic:** modes 9 and above, controlling fine bending and local
  geometric roughness.

The boundaries are declared diagnostics, not universal physical constants.
For each band this run records spectral power, Dirichlet energy, and bending
energy. A band is called depleted when it retains at most
`{COLLAPSE_FRACTION:.0%}` of its initial bending energy.

## CUDA Audit

- Device: `{audit["actual_device"]}`
- GPU: `{audit["gpu_name"]}`
- CUDA used: `{audit["cuda_used"]}`
- Curves: `{audit["fixture_count"]}`
- Samples per curve: `{audit["samples"]}`
- RG scales: `{audit["tau_count"]}`
- Peak CUDA allocation: `{audit["peak_cuda_memory_bytes"]}` bytes
- Elapsed: `{audit["elapsed_seconds"]:.3f}` seconds

## Scale Cascade

| Fixture | Scale | Modes | Initial bending share | 99% absolute-depletion interval | Absolute retention at projection fall | Relative share at projection fall | Depleted before fall |
| --- | --- | --- | ---: | --- | --- | --- | --- |
{table}

`not_applicable` means the unknot fixture has no nonzero minimal-crossing
baseline. `not_seen` means the band did not cross the depletion threshold in
the sampled RG interval. Absolute retention follows the unrescaled heat flow;
relative share records which band controls the rescaled shape.

## Macroscopic Behavior

The low modes retain the gross occupied region while all shorter wavelengths
are suppressed. Eventually the first nonzero harmonic dominates: a rank-two
first mode gives an ellipse, while a higher first mode gives a multiply
covered ellipse. This level answers where the entire tower leans and what
shape survives after the fall. Its absolute energy still decays, but its
relative share approaches one.

## Mesoscopic Behavior

The middle modes encode the visible lobe and crossing architecture. Their
depletion is therefore compared directly with the first fixed-projection
crossing count below the knot's known minimum. This level is the load-bearing
arrangement of the tower: topology can persist after microscopic roughness is
gone, but it cannot be read from the macroscopic outline alone. Near a
multiple-cover limit, a band with tiny relative energy may still act as a
topologically decisive shim that keeps nearly coincident strands apart.

## Microscopic Behavior

High modes carry disproportionately large bending energy because mode `k`
is weighted by `k^4`. The heat multiplier removes them at rate `k^2`, so this
band is depleted first. In the analogy, individual high-curvature blocks fall
without necessarily changing the tower's global knot type.

## Interpretation Boundary

This is a spectral coarse-graining hierarchy. It does not model gravity,
friction, block contacts, or intrinsic elastic-rod dynamics. The scale bands
are useful observables, while the fixed-projection fall remains a one-sided
diagnostic rather than a complete knot classifier.
"""


def main() -> None:
    args = parse_args()
    if args.samples < 128 or args.tau_count < 12 or args.tau_max <= 0:
        raise ValueError("samples >= 128, tau-count >= 12, and tau-max > 0")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.allow_cpu)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    tau_values = np.concatenate(
        (
            np.array([0.0]),
            np.geomspace(1e-4, args.tau_max, args.tau_count - 1),
        )
    )
    fall_times = projection_fall_times(
        output_dir / "curvature_rg_flow_rows.csv"
    )
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    start = time.perf_counter()
    print(f"device={device} scales=3 tau_count={len(tau_values)}")

    for fixture_id, p, q, _ in FIXTURES:
        initial = magnetic_core_curve(
            p,
            q,
            args.samples,
            device=device,
            dtype=torch.float64,
        )
        initial_bands = {
            str(row["scale"]): row for row in spectral_scale_bands(initial)
        }
        initial_total_bending = sum(
            float(row["bending_energy"]) for row in initial_bands.values()
        )
        fixture_rows: list[dict[str, Any]] = []
        for tau_index, tau in enumerate(tau_values):
            curve = periodic_heat_coarse_grain(initial, float(tau))
            current_bands = spectral_scale_bands(curve)
            current_total_bending = sum(
                float(band["bending_energy"]) for band in current_bands
            )
            for band in current_bands:
                scale = str(band["scale"])
                initial_bending = float(
                    initial_bands[scale]["bending_energy"]
                )
                retained = (
                    float(band["bending_energy"]) / initial_bending
                    if initial_bending > 0
                    else 0.0
                )
                row = {
                    "fixture_id": fixture_id,
                    "p": p,
                    "q": q,
                    "tau_index": tau_index,
                    "tau": float(tau),
                    **band,
                    "initial_bending_energy": initial_bending,
                    "initial_bending_fraction": (
                        initial_bending / initial_total_bending
                    ),
                    "current_bending_fraction": (
                        float(band["bending_energy"])
                        / current_total_bending
                    ),
                    "retained_bending_fraction": retained,
                }
                rows.append(row)
                fixture_rows.append(row)

        for scale, initial_band in initial_bands.items():
            scale_rows = [
                row for row in fixture_rows if row["scale"] == scale
            ]
            interval, collapse_upper = collapse_interval(
                scale_rows,
                COLLAPSE_FRACTION,
            )
            fall_tau = fall_times.get(fixture_id)
            if fall_tau is None:
                retained_at_fall: float | str = "not_applicable"
                share_at_fall: float | str = "not_applicable"
                before_fall: bool | str = "not_applicable"
            else:
                fall_row = min(
                    scale_rows,
                    key=lambda row: abs(float(row["tau"]) - fall_tau),
                )
                retained_at_fall = float(
                    fall_row["retained_bending_fraction"]
                )
                share_at_fall = float(
                    fall_row["current_bending_fraction"]
                )
                before_fall = (
                    collapse_upper is not None and collapse_upper <= fall_tau
                )
            maximum_mode = int(initial_band["maximum_mode"])
            summaries.append(
                {
                    "fixture_id": fixture_id,
                    "scale": scale,
                    "mode_range": (
                        f"{int(initial_band['minimum_mode'])}+"
                        if maximum_mode == -1
                        else (
                            f"{int(initial_band['minimum_mode'])}-"
                            f"{maximum_mode}"
                        )
                    ),
                    "initial_bending_fraction": (
                        float(initial_band["bending_energy"])
                        / initial_total_bending
                    ),
                    "one_percent_collapse_interval": interval,
                    "collapse_tau_upper": (
                        collapse_upper if collapse_upper is not None else ""
                    ),
                    "projection_fall_tau": (
                        fall_tau if fall_tau is not None else ""
                    ),
                    "retained_at_projection_fall": retained_at_fall,
                    "bending_share_at_projection_fall": share_at_fall,
                    "collapse_before_projection_fall": before_fall,
                }
            )
        print(f"completed {fixture_id}")

    synchronize(device)
    elapsed = time.perf_counter() - start
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    audit = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "actual_device": str(device),
        "gpu_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "not used"
        ),
        "cuda_used": device.type == "cuda",
        "fixture_count": len(FIXTURES),
        "samples": args.samples,
        "tau_count": len(tau_values),
        "scale_count": 3,
        "depletion_fraction": COLLAPSE_FRACTION,
        "peak_cuda_memory_bytes": peak_memory,
        "elapsed_seconds": elapsed,
    }
    write_csv(output_dir / "curvature_rg_scale_rows.csv", rows)
    write_csv(output_dir / "curvature_rg_scale_summary.csv", summaries)
    (output_dir / "curvature_rg_multiscale_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )
    render_scale_cascade(rows, output_dir)
    (output_dir / "CURVATURE_RG_MULTISCALE_JENGA_REPORT.md").write_text(
        make_report(summaries, audit),
        encoding="utf-8",
    )
    print(
        f"wrote multiscale artifacts in {elapsed:.3f}s "
        f"(peak_cuda_memory={peak_memory})"
    )


if __name__ == "__main__":
    main()
