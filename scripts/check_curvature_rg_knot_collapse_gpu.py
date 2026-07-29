"""Run the curvature-driven Fourier coarse-graining experiment on CUDA."""

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

from scripts.check_maxwell_knot_fields_gpu import projection_crossings
from src.curvature_rg_flow import (
    effective_mode_count,
    first_nonzero_mode,
    geometric_curve_metrics,
    leading_mode_rank,
    normalized_nonlocal_clearance,
    periodic_heat_coarse_grain,
    spectral_energies,
)
from src.maxwell_knot_fields import magnetic_core_curve


OUTPUT_DIR = ROOT / "results" / "curvature_rg_knot_collapse"
FIXTURES = (
    ("hopf_core_unknot", 1, 1, 0),
    ("trefoil_2_3", 2, 3, 3),
    ("cinquefoil_2_5", 2, 5, 5),
    ("torus_knot_3_4", 3, 4, 8),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--tau-count", type=int, default=48)
    parser.add_argument("--tau-max", type=float, default=5.0)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
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
        raise ValueError(f"cannot write empty CSV {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_metric_figure(
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    colors = {
        "hopf_core_unknot": "#00876c",
        "trefoil_2_3": "#0057b8",
        "cinquefoil_2_5": "#d81b60",
        "torus_knot_3_4": "#e67e22",
    }
    for fixture_id, _, _, _ in FIXTURES:
        fixture_rows = [row for row in rows if row["fixture_id"] == fixture_id]
        tau = [max(float(row["tau"]), 1e-5) for row in fixture_rows]
        axes[0].step(
            tau,
            [int(row["projection_crossings"]) for row in fixture_rows],
            where="post",
            label=fixture_id,
            color=colors[fixture_id],
        )
        axes[1].plot(
            tau,
            [float(row["normalized_clearance"]) for row in fixture_rows],
            color=colors[fixture_id],
        )
        axes[2].step(
            tau,
            [int(row["effective_mode_count"]) for row in fixture_rows],
            where="post",
            color=colors[fixture_id],
        )
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel("RG time tau")
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("fixed-projection crossings")
    axes[1].set_ylabel("nonlocal clearance / median edge")
    axes[2].set_ylabel("effective positive Fourier modes")
    axes[0].legend(fontsize=8)
    figure.suptitle("Curvature RG: high-frequency blocks disappear before topology")
    figure.tight_layout()
    figure.savefig(output_dir / "curvature_rg_metrics.png", dpi=180)
    plt.close(figure)


def render_snapshots(
    fixture_id: str,
    initial: torch.Tensor,
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    fixture_rows = [row for row in rows if row["fixture_id"] == fixture_id]
    fall_rows = [row for row in fixture_rows if row["below_minimal_crossing"]]
    fall_index = (
        int(fall_rows[0]["tau_index"])
        if fall_rows
        else len(fixture_rows) - 1
    )
    indices = sorted(
        {
            0,
            max(0, fall_index - 1),
            fall_index,
            len(fixture_rows) - 1,
        }
    )
    while len(indices) < 4:
        indices.append(len(fixture_rows) - 1)
    indices = indices[:4]

    figure = plt.figure(figsize=(14, 3.8))
    for plot_index, row_index in enumerate(indices, start=1):
        row = fixture_rows[row_index]
        curve = periodic_heat_coarse_grain(
            initial,
            float(row["tau"]),
            normalize=True,
        )
        points = curve.detach().cpu().numpy()
        axis = figure.add_subplot(1, 4, plot_index, projection="3d")
        axis.plot(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color="#0057b8",
            linewidth=1.6,
        )
        center = points.mean(axis=0)
        radius = max(0.55 * float(np.max(np.ptp(points, axis=0))), 1e-3)
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.set_box_aspect((1, 1, 1))
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_zticks([])
        axis.set_title(
            "tau={:.3g}\ncrossings={}, modes={}".format(
                float(row["tau"]),
                int(row["projection_crossings"]),
                int(row["effective_mode_count"]),
            ),
            fontsize=9,
        )
    figure.suptitle(f"{fixture_id}: curvature-governed tower collapse")
    figure.tight_layout()
    figure.savefig(output_dir / f"{fixture_id}_rg_snapshots.png", dpi=180)
    plt.close(figure)


def make_report(
    summaries: list[dict[str, Any]],
    audit: dict[str, Any],
) -> str:
    table = "\n".join(
        "| {fixture_id} | {first_nonzero_mode} | {leading_mode_rank} | "
        "{minimal_crossing_number} | {fall_tau_interval} | "
        "{minimum_clearance:.4g} |".format(**summary)
        for summary in summaries
    )
    return f"""# Curvature RG Knot Collapse Report

## Model

The knot is treated as a periodic stack of Fourier blocks. Coarse-graining is
the heat semigroup

```text
partial_tau gamma = partial_s^2 gamma,
gamma_hat_k(tau) = exp(-tau k^2) gamma_hat_k(0).
```

For arclength parameter `s`, `partial_s^2 gamma` is the curvature vector.
The implementation retains the periodic parameter inherited from the Bateman
core sampler, so it is exactly a parameter-space Fourier heat flow rather
than an intrinsic curve-shortening solver. Its spectral bending energy weights
mode `k` by `k^4`, making the removal of fine, curvature-producing modes
explicit. Curves are centered and rescaled only for visualization and
scale-free diagnostics.

This is an abstract geometry flow, not a mechanical model of elastic string
or a literal Jenga tower.

## Exact Collapse Proposition

Let `m` be the first nonzero Fourier mode of a nonconstant smooth closed
curve. After subtracting the centroid and multiplying by `exp(m^2 tau)`, the
heat flow converges in every fixed `C^r` norm to its `m`-th harmonic.

The proof is termwise: every mode `k > m` is suppressed relative to mode `m`
by `exp(-(k^2-m^2)tau)`. When `m=1` and the leading sine/cosine vectors have
rank two, the limit is a planar ellipse. A nontrivial initial knot in this
case cannot remain embedded for every finite RG time: eventual `C^1`
closeness to the ellipse would place it in the ellipse's tubular neighborhood
and hence in the unknot isotopy class.

When `m>1`, the limit is an `m`-fold cover of an ellipse and is not embedded.
Convergence to that limiting multiple cover alone does **not** prove that
self-contact occurs at a finite RG time. The finite-time fall intervals below
are numerical projection diagnostics, not consequences of the exact
proposition. These Fourier statements are elementary and are not claimed as
new literature theorems.

## CUDA Run

- Device: `{audit["actual_device"]}`
- GPU: `{audit["gpu_name"]}`
- CUDA used: `{audit["cuda_used"]}`
- Samples per curve: `{audit["samples"]}`
- RG scales per curve: `{audit["tau_count"]}`
- Peak CUDA allocation: `{audit["peak_cuda_memory_bytes"]}` bytes
- Elapsed: `{audit["elapsed_seconds"]:.3f}` seconds

## Results

| Fixture | First mode | Leading rank | Minimal crossing baseline | First below-baseline interval | Minimum clearance |
| --- | ---: | ---: | ---: | --- | ---: |
{table}

The fixed generic projection supplies a one-sided fall certificate: once its
crossing count is below the known minimal crossing number of the initial
torus knot, the curve cannot still be a generic diagram of that knot. A
crossing-count change by itself is not used as a complete knot classifier.
The nonlocal-clearance trace shows where the sampled tower approaches
self-contact.

The cinquefoil has first surviving mode `2`, so its terminal object is a
double-covered ellipse rather than an embedded unknot. This is why its
clearance becomes extremely small before the projection finally degenerates.

## Interpretation

The useful RG state variables are:

- effective Fourier-block count;
- spectral bending energy;
- geometric curvature;
- nonlocal clearance;
- projection transversality; and
- the minimal-crossing lower bound of the starting knot.

This upgrades the earlier visual analogy into a controlled statement:
curvature determines which scales disappear, while loss of embedding marks
the fall. It does not yet supply a topology-preserving simplifier. Adding
self-repulsion or a hard tube-thickness constraint is the next mathematically
meaningful branch.

## Claim Boundary

- New knot theorem: `false`
- New Maxwell solution: `false`
- Exact abstract collapse proposition recorded: `true`
- Numerical topology classification claimed: `false`
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

    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    initial_curves: dict[str, torch.Tensor] = {}
    start = time.perf_counter()
    print(f"device={device} tau_count={len(tau_values)}")

    for fixture_id, p, q, minimal_crossing in FIXTURES:
        initial = magnetic_core_curve(
            p,
            q,
            args.samples,
            device=device,
            dtype=torch.float64,
        )
        initial_curves[fixture_id] = initial
        first_mode = first_nonzero_mode(initial)
        mode_rank = leading_mode_rank(initial, first_mode)
        fixture_rows: list[dict[str, Any]] = []
        for tau_index, tau in enumerate(tau_values):
            raw = periodic_heat_coarse_grain(initial, float(tau))
            normalized = periodic_heat_coarse_grain(
                initial,
                float(tau),
                normalize=True,
            )
            spectral = spectral_energies(raw)
            geometric = geometric_curve_metrics(normalized)
            clearance = normalized_nonlocal_clearance(normalized)
            synchronize(device)
            transfer_start = time.perf_counter()
            points_cpu = normalized.detach().cpu().numpy()
            transfer_seconds = time.perf_counter() - transfer_start
            crossings = len(projection_crossings(points_cpu))
            row = {
                "fixture_id": fixture_id,
                "p": p,
                "q": q,
                "tau_index": tau_index,
                "tau": float(tau),
                "minimal_crossing_number": minimal_crossing,
                "projection_crossings": crossings,
                "below_minimal_crossing": (
                    minimal_crossing > 0 and crossings < minimal_crossing
                ),
                "first_nonzero_mode": first_mode,
                "leading_mode_rank": mode_rank,
                "effective_mode_count": effective_mode_count(normalized),
                "normalized_clearance": clearance,
                **spectral,
                **geometric,
                "cpu_transfer_seconds": transfer_seconds,
            }
            rows.append(row)
            fixture_rows.append(row)

        fall_rows = [
            row for row in fixture_rows if row["below_minimal_crossing"]
        ]
        if fall_rows:
            first_fall = fall_rows[0]
            first_index = int(first_fall["tau_index"])
            previous_tau = float(fixture_rows[first_index - 1]["tau"])
            fall_interval = (
                f"({previous_tau:.6g}, {float(first_fall['tau']):.6g}]"
            )
        elif minimal_crossing == 0:
            fall_interval = "not_applicable"
        else:
            fall_interval = f"not_seen_through_{args.tau_max:g}"
        summary = {
            "fixture_id": fixture_id,
            "p": p,
            "q": q,
            "first_nonzero_mode": first_mode,
            "leading_mode_rank": mode_rank,
            "minimal_crossing_number": minimal_crossing,
            "fall_tau_interval": fall_interval,
            "fall_detected": bool(fall_rows),
            "minimum_clearance": min(
                float(row["normalized_clearance"]) for row in fixture_rows
            ),
            "initial_effective_mode_count": int(
                fixture_rows[0]["effective_mode_count"]
            ),
            "final_effective_mode_count": int(
                fixture_rows[-1]["effective_mode_count"]
            ),
            "initial_spectral_bending_energy": float(
                fixture_rows[0]["spectral_bending_energy"]
            ),
            "final_spectral_bending_energy": float(
                fixture_rows[-1]["spectral_bending_energy"]
            ),
        }
        summaries.append(summary)
        print(
            f"completed {fixture_id}: first_mode={first_mode} "
            f"fall={fall_interval}"
        )

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
        "samples": args.samples,
        "tau_count": len(tau_values),
        "tau_max": args.tau_max,
        "peak_cuda_memory_bytes": peak_memory,
        "elapsed_seconds": elapsed,
        "all_nontrivial_projection_falls_detected": all(
            summary["fall_detected"]
            for summary in summaries
            if int(summary["minimal_crossing_number"]) > 0
        ),
        "new_knot_theorem_claimed": False,
        "new_maxwell_solution_claimed": False,
    }
    write_csv(output_dir / "curvature_rg_flow_rows.csv", rows)
    write_csv(output_dir / "curvature_rg_summary.csv", summaries)
    (output_dir / "curvature_rg_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )
    render_metric_figure(rows, output_dir)
    for fixture_id, _, _, _ in FIXTURES:
        render_snapshots(
            fixture_id,
            initial_curves[fixture_id],
            rows,
            output_dir,
        )
    report = make_report(summaries, audit)
    (output_dir / "CURVATURE_RG_KNOT_COLLAPSE_REPORT.md").write_text(
        report,
        encoding="utf-8",
    )
    print(
        f"wrote {output_dir} in {elapsed:.3f}s "
        f"(peak_cuda_memory={peak_memory})"
    )


if __name__ == "__main__":
    main()
