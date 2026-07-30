"""CUDA comparison of i.i.d. and correlated threshold fields on Hex boards."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import platform
from statistics import NormalDist
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

from src.fractional_field_four_color import fractional_gaussian_surfaces  # noqa: E402
from src.hex_correlated_fields import (  # noqa: E402
    axial_neighbor_agreement,
    boolean_boards_to_bitboards,
    interpolate_probability_level,
)
from src.hex_crossing import crossing_outcomes  # noqa: E402


OUTPUT_DIR = ROOT / "results" / "hex_correlated_fields_gpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--samples", type=int, default=200_000)
    parser.add_argument("--chunk-size", type=int, default=25_000)
    parser.add_argument("--seed", type=int, default=20260731)
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


def sample_boards(
    family: str,
    size: int,
    probability: float,
    sample_count: int,
    *,
    device: torch.device,
    seed: int,
    hurst: float | None,
) -> torch.Tensor:
    if family == "iid":
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        return torch.rand(
            (sample_count, size, size),
            device=device,
            generator=generator,
        ) < probability
    if family != "fractional_gaussian" or hurst is None:
        raise ValueError("unknown field family")
    surfaces = fractional_gaussian_surfaces(
        sample_count,
        8,
        hurst,
        seed=seed,
        device=device,
        dtype=torch.float32,
    )
    threshold = NormalDist().inv_cdf(1.0 - probability)
    return surfaces[:, :size, :size] >= threshold


def run_configuration(
    family: str,
    size: int,
    probability: float,
    samples: int,
    *,
    device: torch.device,
    chunk_size: int,
    seed: int,
    hurst: float | None,
) -> dict[str, object]:
    blue_total = 0
    yellow_total = 0
    both_total = 0
    neither_total = 0
    occupied_total = 0
    agreement_total = 0.0
    synchronize(device)
    started = time.perf_counter()
    for chunk_index, start in enumerate(range(0, samples, chunk_size)):
        current = min(chunk_size, samples - start)
        boards = sample_boards(
            family,
            size,
            probability,
            current,
            device=device,
            seed=seed + 1_000_003 * chunk_index,
            hurst=hurst,
        )
        states = boolean_boards_to_bitboards(boards)
        blue, yellow = crossing_outcomes(states, size)
        blue_total += int(blue.sum().item())
        yellow_total += int(yellow.sum().item())
        both_total += int((blue & yellow).sum().item())
        neither_total += int((~(blue | yellow)).sum().item())
        occupied_total += int(boards.sum().item())
        agreement_total += float(axial_neighbor_agreement(boards).sum().item())
    synchronize(device)
    elapsed = time.perf_counter() - started
    estimate = blue_total / samples
    standard_error = math.sqrt(estimate * (1.0 - estimate) / samples)
    return {
        "family": family,
        "hurst": "" if hurst is None else hurst,
        "size": size,
        "nominal_probability": probability,
        "samples": samples,
        "realized_blue_fraction": occupied_total / (samples * size * size),
        "mean_neighbor_agreement": agreement_total / samples,
        "blue_crossing_count": blue_total,
        "yellow_crossing_count": yellow_total,
        "both_winners": both_total,
        "neither_winners": neither_total,
        "blue_crossing_probability": estimate,
        "ci95_low": max(0.0, estimate - 1.96 * standard_error),
        "ci95_high": min(1.0, estimate + 1.96 * standard_error),
        "elapsed_seconds": elapsed,
        "samples_per_second": samples / elapsed,
    }


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["family"]), str(row["hurst"]), int(row["size"]))
        grouped.setdefault(key, []).append(row)
    output = []
    for (family, hurst, size), selected in sorted(grouped.items()):
        selected.sort(key=lambda row: float(row["nominal_probability"]))
        probabilities = [float(row["nominal_probability"]) for row in selected]
        crossings = [float(row["blue_crossing_probability"]) for row in selected]
        p25 = interpolate_probability_level(probabilities, crossings, 0.25)
        p75 = interpolate_probability_level(probabilities, crossings, 0.75)
        half = min(
            selected,
            key=lambda row: abs(float(row["nominal_probability"]) - 0.5),
        )
        output.append(
            {
                "family": family,
                "hurst": hurst,
                "size": size,
                "samples_per_probability": int(half["samples"]),
                "crossing_probability_at_half": float(
                    half["blue_crossing_probability"]
                ),
                "realized_blue_fraction_at_half": float(
                    half["realized_blue_fraction"]
                ),
                "neighbor_agreement_at_half": float(
                    half["mean_neighbor_agreement"]
                ),
                "p25": p25,
                "p75": p75,
                "transition_width": p75 - p25,
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(rows: list[dict[str, object]]) -> None:
    sizes = sorted({int(row["size"]) for row in rows})
    figure, axes = plt.subplots(1, len(sizes), figsize=(6 * len(sizes), 4.5))
    if len(sizes) == 1:
        axes = [axes]
    for axis, size in zip(axes, sizes, strict=True):
        selected_size = [row for row in rows if int(row["size"]) == size]
        keys = sorted(
            {(str(row["family"]), str(row["hurst"])) for row in selected_size}
        )
        for family, hurst in keys:
            selected = [
                row
                for row in selected_size
                if str(row["family"]) == family and str(row["hurst"]) == hurst
            ]
            selected.sort(key=lambda row: float(row["nominal_probability"]))
            label = "i.i.d." if family == "iid" else f"FGF H={hurst}"
            axis.plot(
                [float(row["nominal_probability"]) for row in selected],
                [float(row["blue_crossing_probability"]) for row in selected],
                marker="o",
                linewidth=1.8,
                markersize=4,
                label=label,
            )
        axis.axhline(0.5, color="#666666", linewidth=0.8)
        axis.axvline(0.5, color="#666666", linewidth=0.8)
        axis.set_title(f"{size} x {size} Hex")
        axis.set_xlabel("Nominal blue probability")
        axis.set_ylabel("Blue crossing probability")
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.2)
        axis.legend()
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "hex_correlated_crossing_curves.png",
        dpi=180,
    )
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
    probabilities = (
        (0.35, 0.50, 0.65)
        if args.smoke
        else (0.25, 0.30, 0.35, 0.40, 0.45, 0.475, 0.50, 0.525, 0.55, 0.60, 0.65, 0.70, 0.75)
    )
    configurations = (("iid", None), ("fractional_gaussian", 0.2)) if args.smoke else (
        ("iid", None),
        ("fractional_gaussian", 0.2),
        ("fractional_gaussian", 0.5),
        ("fractional_gaussian", 0.8),
    )
    samples = min(args.samples, 10_000) if args.smoke else args.samples
    print(
        f"device={device} gpu="
        f"{torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}"
    )
    rows: list[dict[str, object]] = []
    config_index = 0
    for size in sizes:
        for family, hurst in configurations:
            for probability in probabilities:
                seed = args.seed + 10_000_019 * config_index
                row = run_configuration(
                    family,
                    size,
                    probability,
                    samples,
                    device=device,
                    chunk_size=args.chunk_size,
                    seed=seed,
                    hurst=hurst,
                )
                rows.append(row)
                config_index += 1
                print(
                    f"H{size} {family} H={hurst} p={probability:.3f}: "
                    f"density={row['realized_blue_fraction']:.4f} "
                    f"crossing={row['blue_crossing_probability']:.4f} "
                    f"agreement={row['mean_neighbor_agreement']:.4f}"
                )
    summaries = summarize(rows)
    write_csv(OUTPUT_DIR / "hex_correlated_field_rows.csv", rows)
    write_csv(OUTPUT_DIR / "hex_correlated_field_summary.csv", summaries)
    plot_curves(rows)
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
        "gpu_capability": (
            torch.cuda.get_device_capability(0) if device.type == "cuda" else None
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
        "samples_total": sum(int(row["samples"]) for row in rows),
        "both_or_neither_winner_count": sum(
            int(row["both_winners"]) + int(row["neither_winners"])
            for row in rows
        ),
        "families": sorted({str(row["family"]) for row in rows}),
        "hurst_values": sorted(
            {float(row["hurst"]) for row in rows if row["hurst"] != ""}
        ),
        "continuum_fractional_field_claim": False,
        "new_hex_theorem": False,
    }
    (OUTPUT_DIR / "hex_correlated_field_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )
    table = "\n".join(
        "| {family} | {hurst} | {size} | {crossing_probability_at_half:.4f} | "
        "{realized_blue_fraction_at_half:.4f} | "
        "{neighbor_agreement_at_half:.4f} | {transition_width:.4f} |".format(**row)
        for row in summaries
    )
    report = f"""# Correlated-Field Hex CUDA Report

## Experiment

Thresholded `6 x 6` and `7 x 7` Hex boards were generated from either
independent uniforms or periodic cutoff fractional-Gaussian surfaces. Each
curve uses **{samples:,}** samples per nominal probability and seed
`{args.seed}`.

| family | H | size | crossing at 1/2 | realized blue | neighbor agreement | p75-p25 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

![Crossing curves](hex_correlated_crossing_curves.png)

## Reading the comparison

Neighbor agreement measures the local clustering introduced by the field.
The crossing curve measures its global consequence. Comparing transition
widths at fixed board size separates a density effect from a correlation
effect: a family can have realized blue fraction near one half while having
a substantially different crossing response away from one half.

## Boundaries

- These are finite periodic cutoff Gaussian fields at resolution eight, not
  continuum fractional Gaussian fields.
- Per-surface centering and variance normalization alter finite-sample
  marginal tails; the realized blue fraction is therefore reported.
- Correlated samples do not inherit Russo's independent-site pivotal
  identity.
- The Hex winner theorem remains exact for every completed board, but the
  sampled probability curves are empirical.
- This experiment diagnoses planar threshold slices; it does not preserve or
  certify three-dimensional knot type.
"""
    (OUTPUT_DIR / "HEX_CORRELATED_FIELD_GPU_REPORT.md").write_text(
        report,
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
