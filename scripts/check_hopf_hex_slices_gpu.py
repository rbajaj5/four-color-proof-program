"""CUDA Hex-crossing experiment on planar pullbacks of the Hopf map."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hex_correlated_fields import (  # noqa: E402
    axial_neighbor_agreement,
    boolean_boards_to_bitboards,
)
from src.hex_crossing import crossing_outcomes  # noqa: E402
from src.hopf_hex_slices import sample_hopf_hex_window  # noqa: E402


OUTPUT_DIR = ROOT / "results" / "hopf_hex_slices_gpu"
CHANNELS = ("h_x", "h_y", "h_z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--chunk-size", type=int, default=2_500)
    parser.add_argument("--center-extent", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260802)
    return parser.parse_args()


def select_device(allow_cpu: bool) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if allow_cpu:
        return torch.device("cpu")
    raise RuntimeError("CUDA unavailable; refusing silent CPU fallback")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def decreasing_level(
    thresholds: list[float],
    crossings: list[float],
    target: float,
) -> float:
    ordered = sorted(zip(thresholds, crossings, strict=True))
    monotone = []
    running = 1.0
    for threshold, crossing in ordered:
        running = min(running, crossing)
        monotone.append((threshold, running))
    for (left_t, left_v), (right_t, right_v) in zip(
        monotone,
        monotone[1:],
        strict=True,
    ):
        if left_v >= target >= right_v:
            if left_v == right_v:
                return 0.5 * (left_t + right_t)
            fraction = (left_v - target) / (left_v - right_v)
            return left_t + fraction * (right_t - left_t)
    raise ValueError("target is outside the sampled threshold curve")


def run_family(
    size: int,
    field_of_view: float,
    thresholds: torch.Tensor,
    samples: int,
    *,
    chunk_size: int,
    center_extent: float,
    device: torch.device,
    seed: int,
) -> tuple[list[dict[str, object]], float, float]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    threshold_count = len(thresholds)
    totals = torch.zeros(
        (len(CHANNELS), threshold_count, 7),
        dtype=torch.float64,
        device=device,
    )
    max_phase_error = 0.0
    synchronize(device)
    started = time.perf_counter()
    for start in range(0, samples, chunk_size):
        current = min(chunk_size, samples - start)
        image, rotated, phase_error = sample_hopf_hex_window(
            size,
            field_of_view,
            current,
            center_extent=center_extent,
            device=device,
            generator=generator,
        )
        max_phase_error = max(max_phase_error, phase_error)
        for channel_index in range(len(CHANNELS)):
            values = image[..., channel_index]
            rotated_values = rotated[..., channel_index]
            boards = values[:, None, :, :] >= thresholds[None, :, None, None]
            rotated_boards = (
                rotated_values[:, None, :, :]
                >= thresholds[None, :, None, None]
            )
            flattened = boards.reshape(
                current * threshold_count,
                size,
                size,
            )
            states = boolean_boards_to_bitboards(flattened)
            blue, yellow = crossing_outcomes(states, size)
            blue = blue.reshape(current, threshold_count)
            yellow = yellow.reshape(current, threshold_count)
            agreement = axial_neighbor_agreement(flattened).reshape(
                current,
                threshold_count,
            )
            totals[channel_index, :, 0] += boards.sum(dim=(0, 2, 3))
            totals[channel_index, :, 1] += agreement.sum(dim=0)
            totals[channel_index, :, 2] += blue.sum(dim=0)
            totals[channel_index, :, 3] += yellow.sum(dim=0)
            totals[channel_index, :, 4] += (blue & yellow).sum(dim=0)
            totals[channel_index, :, 5] += (~(blue | yellow)).sum(dim=0)
            totals[channel_index, :, 6] += (
                boards != rotated_boards
            ).sum(dim=(0, 2, 3))
    synchronize(device)
    elapsed = time.perf_counter() - started
    values = totals.cpu()
    rows = []
    for channel_index, channel in enumerate(CHANNELS):
        for threshold_index, threshold in enumerate(thresholds.cpu().tolist()):
            selected = values[channel_index, threshold_index]
            rows.append(
                {
                    "size": size,
                    "field_of_view": field_of_view,
                    "center_extent": center_extent,
                    "channel": channel,
                    "threshold": threshold,
                    "samples": samples,
                    "realized_blue_fraction": float(
                        selected[0] / (samples * size * size)
                    ),
                    "mean_neighbor_agreement": float(selected[1] / samples),
                    "blue_crossing_probability": float(selected[2] / samples),
                    "yellow_crossing_probability": float(selected[3] / samples),
                    "both_winners": int(selected[4]),
                    "neither_winners": int(selected[5]),
                    "u1_board_mismatch_count": int(selected[6]),
                }
            )
    return rows, elapsed, max_phase_error


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, float, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            int(row["size"]),
            float(row["field_of_view"]),
            str(row["channel"]),
        )
        grouped.setdefault(key, []).append(row)
    summaries = []
    for (size, field_of_view, channel), selected in sorted(grouped.items()):
        selected.sort(key=lambda row: float(row["threshold"]))
        thresholds = [float(row["threshold"]) for row in selected]
        crossings = [float(row["blue_crossing_probability"]) for row in selected]
        level_75 = decreasing_level(thresholds, crossings, 0.75)
        level_25 = decreasing_level(thresholds, crossings, 0.25)
        zero = min(selected, key=lambda row: abs(float(row["threshold"])))
        summaries.append(
            {
                "size": size,
                "field_of_view": field_of_view,
                "channel": channel,
                "samples_per_threshold": int(zero["samples"]),
                "blue_fraction_at_zero": float(zero["realized_blue_fraction"]),
                "neighbor_agreement_at_zero": float(
                    zero["mean_neighbor_agreement"]
                ),
                "crossing_probability_at_zero": float(
                    zero["blue_crossing_probability"]
                ),
                "threshold_at_crossing_75": level_75,
                "threshold_at_crossing_25": level_25,
                "transition_width": level_25 - level_75,
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(rows: list[dict[str, object]]) -> None:
    sizes = sorted({int(row["size"]) for row in rows})
    figure, axes = plt.subplots(1, len(sizes), figsize=(6.3 * len(sizes), 4.8))
    if len(sizes) == 1:
        axes = [axes]
    colors = {"h_x": "#0072B2", "h_y": "#D55E00", "h_z": "#009E73"}
    styles = {1.0: ":", 2.0: "--", 4.0: "-"}
    for axis, size in zip(axes, sizes, strict=True):
        for field_of_view in sorted(
            {float(row["field_of_view"]) for row in rows}
        ):
            for channel in CHANNELS:
                selected = [
                    row
                    for row in rows
                    if int(row["size"]) == size
                    and float(row["field_of_view"]) == field_of_view
                    and str(row["channel"]) == channel
                ]
                if not selected:
                    continue
                selected.sort(key=lambda row: float(row["threshold"]))
                axis.plot(
                    [float(row["threshold"]) for row in selected],
                    [float(row["blue_crossing_probability"]) for row in selected],
                    color=colors[channel],
                    linestyle=styles[field_of_view],
                    marker="o",
                    markersize=3,
                    linewidth=1.5,
                    label=f"{channel}, FOV={field_of_view:g}",
                )
        axis.axhline(0.5, color="#666666", linewidth=0.8)
        axis.axvline(0.0, color="#666666", linewidth=0.8)
        axis.set_title(f"Hopf pullback on {size} x {size} Hex")
        axis.set_xlabel("Hopf-component threshold")
        axis.set_ylabel("Blue crossing probability")
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "hopf_hex_crossing_curves.png", dpi=180)
    plt.close(figure)


def nvidia_smi() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unavailable"


def main() -> None:
    args = parse_args()
    device = select_device(args.allow_cpu)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sizes = (6,) if args.smoke else (6, 7)
    fields_of_view = (2.0,) if args.smoke else (1.0, 2.0, 4.0)
    threshold_values = (-0.6, -0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4, 0.6)
    thresholds = torch.tensor(
        threshold_values,
        dtype=torch.float64,
        device=device,
    )
    samples = min(args.samples, 5_000) if args.smoke else args.samples
    print(
        f"device={device} gpu="
        f"{torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}"
    )
    rows: list[dict[str, object]] = []
    timing_rows = []
    max_phase_error = 0.0
    config_index = 0
    for size in sizes:
        for field_of_view in fields_of_view:
            selected_rows, elapsed, phase_error = run_family(
                size,
                field_of_view,
                thresholds,
                samples,
                chunk_size=args.chunk_size,
                center_extent=args.center_extent,
                device=device,
                seed=args.seed + 10_000_019 * config_index,
            )
            rows.extend(selected_rows)
            max_phase_error = max(max_phase_error, phase_error)
            timing_rows.append(
                {
                    "size": size,
                    "field_of_view": field_of_view,
                    "samples": samples,
                    "board_evaluations": (
                        samples * len(CHANNELS) * len(threshold_values)
                    ),
                    "elapsed_seconds": elapsed,
                }
            )
            config_index += 1
            zero_rows = [
                row for row in selected_rows if float(row["threshold"]) == 0.0
            ]
            summary_text = ", ".join(
                f"{row['channel']}={row['blue_crossing_probability']:.3f}"
                for row in zero_rows
            )
            print(
                f"H{size} FOV={field_of_view:g}: {summary_text} "
                f"phase_error={phase_error:.2e}"
            )
    summaries = summarize(rows)
    write_csv(OUTPUT_DIR / "hopf_hex_slice_rows.csv", rows)
    write_csv(OUTPUT_DIR / "hopf_hex_slice_summary.csv", summaries)
    write_csv(OUTPUT_DIR / "hopf_hex_slice_timing.csv", timing_rows)
    plot_rows(rows)
    device_report = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "selected_device": str(device),
        "gpu_name": (
            torch.cuda.get_device_name(0) if device.type == "cuda" else None
        ),
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else 0
        ),
        "seed": args.seed,
        "nvidia_smi": nvidia_smi(),
    }
    (OUTPUT_DIR / "device_report.json").write_text(
        json.dumps(device_report, indent=2) + "\n",
        encoding="utf-8",
    )
    audit = {
        "cuda_used": device.type == "cuda",
        "field_slices_total": sum(
            int(row["samples"]) for row in timing_rows
        ),
        "hex_board_evaluations_total": sum(
            int(row["board_evaluations"]) for row in timing_rows
        ),
        "both_or_neither_winner_count": sum(
            int(row["both_winners"]) + int(row["neither_winners"])
            for row in rows
        ),
        "u1_board_mismatch_count": sum(
            int(row["u1_board_mismatch_count"]) for row in rows
        ),
        "hopf_map_u1_max_error": max_phase_error,
        "imports_tuft_physical_claims": False,
        "three_dimensional_knot_certificate": False,
    }
    (OUTPUT_DIR / "hopf_hex_slice_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )
    table = "\n".join(
        "| {size} | {field_of_view} | {channel} | "
        "{blue_fraction_at_zero:.4f} | {neighbor_agreement_at_zero:.4f} | "
        "{crossing_probability_at_zero:.4f} | {transition_width:.4f} |".format(
            **row
        )
        for row in summaries
    )
    report = f"""# Hopf-to-Hex CUDA Slice Report

## Experiment

The classical Hopf map is evaluated on Bateman `S^3` coordinates over
randomly centered and oriented planar Hex windows. Each component
`h_x, h_y, h_z` is thresholded into a completed two-color board. The full
run uses **{samples:,}** windows per size/FOV configuration.

| size | FOV | channel | blue fraction at 0 | neighbor agreement | crossing at 0 | threshold width |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
{table}

![Hopf Hex crossing curves](hopf_hex_crossing_curves.png)

## Gauge audit

Every `S^3` coordinate pair was also multiplied by a random common `U(1)`
phase before applying the Hopf map.

- Maximum Hopf-map phase residual: **{max_phase_error:.3e}**.
- Thresholded-board mismatches: **{audit['u1_board_mismatch_count']}**.
- Both/neither Hex winners: **{audit['both_or_neither_winner_count']}**.

Thus the planar observable depends on the projective Hopf point, not the
arbitrary fiber-phase representative.

## Interpretation

The transition width and neighbor agreement measure how the pulled-back Hopf
texture alters global planar connectivity. Differences among components or
fields of view are geometric finite-window effects. They are not evidence
that Hex detects a three-dimensional knot.

## Surrogate-model decision

No learned surrogate is fitted. The Hopf map and Hex winner test are evaluated
directly, and the remaining threshold response is one-dimensional and
monotone. Following the compute-matched cautions in the KAN critical
assessment (arXiv:2407.11075), a KAN would enter only a later
symbolic-regression comparison against simpler spline, logistic, and matched
MLP baselines. Predictive fit would remain conjecture-generating evidence, not
a topology proof.

## Boundaries

- This experiment uses only the classical Hopf bundle and the existing exact
  Bateman coordinate formulas.
- It does not import the TUFT manuscript's gauge-gravity or particle claims.
- Random planar slices discard three-dimensional information and cannot
  certify linking, helicity, or knot type.
- The curves are finite Monte Carlo measurements, not new Hex theorems.
"""
    (OUTPUT_DIR / "HOPF_HEX_SLICE_GPU_REPORT.md").write_text(
        report,
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
