"""Exact and sampled CUDA experiments for rhombic Hex crossing events."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
from pathlib import Path
import subprocess
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hex_crossing import (  # noqa: E402
    crossing_outcomes,
    inverse_reliability,
    popcount,
    reliability_derivative,
    reliability_probability,
)


OUTPUT_DIR = ROOT / "results" / "hex_crossing_gpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--exact-chunk-size", type=int, default=1_048_576)
    parser.add_argument("--sample-chunk-size", type=int, default=100_000)
    parser.add_argument("--samples", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=20260730)
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


def exact_census(
    size: int,
    device: torch.device,
    chunk_size: int,
) -> tuple[dict[str, object], list[int]]:
    count = size * size
    total = 1 << count
    coefficients = torch.zeros(count + 1, dtype=torch.int64, device=device)
    outcome_counts = torch.zeros(5, dtype=torch.int64, device=device)
    synchronize(device)
    started = time.perf_counter()
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        states = torch.arange(start, stop, dtype=torch.int64, device=device)
        blue, yellow = crossing_outcomes(states, size)
        weights = popcount(states, count)
        coefficients += torch.bincount(weights[blue], minlength=count + 1)
        outcome_counts += torch.stack(
            (
                blue.sum(),
                yellow.sum(),
                (blue & yellow).sum(),
                (~(blue | yellow)).sum(),
                torch.tensor(stop - start, dtype=torch.int64, device=device),
            )
        )
    synchronize(device)
    elapsed = time.perf_counter() - started
    values = coefficients.cpu().tolist()
    outcomes = outcome_counts.cpu().tolist()
    low = inverse_reliability(values, 0.25)
    high = inverse_reliability(values, 0.75)
    return (
        {
            "size": size,
            "cells": count,
            "states": total,
            "blue_winners": outcomes[0],
            "yellow_winners": outcomes[1],
            "both_winners": outcomes[2],
            "neither_winners": outcomes[3],
            "processed_states": outcomes[4],
            "probability_at_half": reliability_probability(values, 0.5),
            "slope_at_half": reliability_derivative(values, 0.5),
            "p25": low,
            "p75": high,
            "critical_window_width": high - low,
            "elapsed_seconds": elapsed,
            "states_per_second": total / elapsed,
        },
        values,
    )


def random_bitboards(
    sample_count: int,
    size: int,
    probability: float,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    count = size * size
    uniforms = torch.rand(
        (sample_count, count),
        device=device,
        generator=generator,
    )
    occupied = uniforms < probability
    states = torch.zeros(sample_count, dtype=torch.int64, device=device)
    for index in range(count):
        states |= occupied[:, index].to(torch.int64) << index
    return states


def sampled_crossing(
    size: int,
    probability: float,
    samples: int,
    *,
    device: torch.device,
    chunk_size: int,
    generator: torch.Generator,
) -> dict[str, object]:
    blue_total = 0
    yellow_total = 0
    both_total = 0
    neither_total = 0
    synchronize(device)
    started = time.perf_counter()
    for start in range(0, samples, chunk_size):
        current = min(chunk_size, samples - start)
        states = random_bitboards(
            current,
            size,
            probability,
            device=device,
            generator=generator,
        )
        blue, yellow = crossing_outcomes(states, size)
        blue_total += int(blue.sum().item())
        yellow_total += int(yellow.sum().item())
        both_total += int((blue & yellow).sum().item())
        neither_total += int((~(blue | yellow)).sum().item())
    synchronize(device)
    elapsed = time.perf_counter() - started
    estimate = blue_total / samples
    standard_error = math.sqrt(estimate * (1.0 - estimate) / samples)
    return {
        "size": size,
        "probability": probability,
        "samples": samples,
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


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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


def fit_slope_exponent(rows: list[dict[str, object]]) -> dict[str, float]:
    usable = [row for row in rows if int(row["size"]) >= 2]
    x = torch.tensor(
        [math.log(float(row["size"])) for row in usable],
        dtype=torch.float64,
    )
    y = torch.tensor(
        [math.log(float(row["slope_at_half"])) for row in usable],
        dtype=torch.float64,
    )
    design = torch.stack((torch.ones_like(x), x), dim=1)
    solution = torch.linalg.lstsq(design, y).solution
    prediction = design @ solution
    residual = float(torch.sum((y - prediction) ** 2))
    return {
        "intercept": float(solution[0]),
        "finite_size_exponent": float(solution[1]),
        "residual_sum_squares": residual,
    }


def main() -> None:
    args = parse_args()
    device = select_device(args.allow_cpu)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    exact_sizes = range(1, 4) if args.smoke else range(1, 6)
    sample_sizes = (4,) if args.smoke else (6, 7)
    probabilities = (
        (0.4, 0.5, 0.6)
        if args.smoke
        else (0.35, 0.40, 0.425, 0.45, 0.475, 0.50, 0.525, 0.55, 0.575, 0.60, 0.65)
    )
    samples = min(args.samples, 20_000) if args.smoke else args.samples
    print(
        f"device={device} gpu="
        f"{torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}"
    )

    exact_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    for size in exact_sizes:
        summary, coefficients = exact_census(
            size,
            device,
            args.exact_chunk_size,
        )
        exact_rows.append(summary)
        for blue_cells, coefficient in enumerate(coefficients):
            coefficient_rows.append(
                {
                    "size": size,
                    "cells": size * size,
                    "blue_cells": blue_cells,
                    "blue_winner_count": coefficient,
                    "all_colorings_at_weight": math.comb(size * size, blue_cells),
                }
            )
        print(
            f"exact H{size}: states={summary['states']:,} "
            f"slope={summary['slope_at_half']:.6f} "
            f"mismatch={summary['both_winners'] + summary['neither_winners']} "
            f"seconds={summary['elapsed_seconds']:.4f}"
        )

    sample_rows: list[dict[str, object]] = []
    for size in sample_sizes:
        for probability in probabilities:
            row = sampled_crossing(
                size,
                probability,
                samples,
                device=device,
                chunk_size=args.sample_chunk_size,
                generator=generator,
            )
            sample_rows.append(row)
            print(
                f"sample H{size} p={probability:.3f}: "
                f"P={row['blue_crossing_probability']:.6f} "
                f"seconds={row['elapsed_seconds']:.4f}"
            )

    write_csv(OUTPUT_DIR / "hex_exact_summary.csv", exact_rows)
    write_csv(OUTPUT_DIR / "hex_reliability_coefficients.csv", coefficient_rows)
    write_csv(OUTPUT_DIR / "hex_monte_carlo_crossings.csv", sample_rows)
    exponent_fit = fit_slope_exponent(exact_rows)
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
        "largest_exact_size": max(exact_sizes),
        "exact_states_total": sum(int(row["states"]) for row in exact_rows),
        "exactly_one_winner_all_exact_states": all(
            int(row["both_winners"]) == 0 and int(row["neither_winners"]) == 0
            for row in exact_rows
        ),
        "half_probability_is_one_half": all(
            abs(float(row["probability_at_half"]) - 0.5) < 1e-12
            for row in exact_rows
        ),
        "coefficient_duality": all(
            int(row["blue_winner_count"])
            + next(
                int(other["blue_winner_count"])
                for other in coefficient_rows
                if other["size"] == row["size"]
                and other["blue_cells"] == int(row["cells"]) - int(row["blue_cells"])
            )
            == int(row["all_colorings_at_weight"])
            for row in coefficient_rows
        ),
        "monte_carlo_both_or_neither": sum(
            int(row["both_winners"]) + int(row["neither_winners"])
            for row in sample_rows
        ),
        "finite_size_fit": exponent_fit,
        "new_hex_theorem": False,
    }
    (OUTPUT_DIR / "hex_crossing_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )

    exact_table = "\n".join(
        "| {size} | {cells} | {states} | {blue_winners} | {yellow_winners} | "
        "{slope_at_half:.6f} | {critical_window_width:.6f} | "
        "{elapsed_seconds:.4f} |".format(**row)
        for row in exact_rows
    )
    sampled_half = [
        row for row in sample_rows if abs(float(row["probability"]) - 0.5) < 1e-12
    ]
    sample_table = "\n".join(
        "| {size} | {samples} | {blue_crossing_probability:.6f} | "
        "{ci95_low:.6f} | {ci95_high:.6f} | {elapsed_seconds:.4f} |".format(**row)
        for row in sampled_half
    )
    report = f"""# Hex Crossing CUDA Report

## Exact reliability census

Blue connects the two `q` sides of an `n x n` rhombic Hex board; yellow
connects the two `r` sides. Every coloring through side length
**{max(exact_sizes)}** was enumerated and tested directly.

| n | cells | states | blue wins | yellow wins | slope at 1/2 | p75-p25 | seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{exact_table}

The coefficient file records the exact reliability polynomial

```text
P_n(p) = sum_k B[n,k] p^k (1-p)^(n^2-k),
```

where `B[n,k]` is the number of blue-winning boards with exactly `k` blue
cells. Color/axis duality gives `P_n(1/2)=1/2`. The derivative
`P_n'(1/2)` is the expected number of pivotal cells under unbiased
independent coloring by Russo's formula.

The log-log slope fit over the very small exact sizes `n=2,...,{max(exact_sizes)}`
is **{exponent_fit['finite_size_exponent']:.6f}**. This is a finite-size
diagnostic only, not an estimate of a critical exponent.

## Larger-board sampling at p=1/2

| n | samples | blue crossing estimate | 95% low | 95% high | seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
{sample_table}

The full Monte Carlo file contains the crossing curve across the declared
probability grid. All random runs use seed `{args.seed}`.

## Interpretation

This turns the qualitative Hex interface theorem into two quantitative
objects: an exact small-board reliability polynomial and a sampled
large-board transition curve. The slope and critical-window width measure
how sensitively global connectivity responds to local color perturbations.
That makes them suitable diagnostics for thresholded planar slices of
curvature, flux, or other fields.

## Boundaries

- The exactly-one-winner statement is the standard Hex theorem, not new.
- The exact census ends at `n={max(exact_sizes)}`; larger rows are Monte Carlo.
- The finite-size exponent fit is not an asymptotic percolation claim.
- Crossing connectivity does not certify knot type, linking, helicity, or
  a three-dimensional Four Color analogue.
"""
    (OUTPUT_DIR / "HEX_CROSSING_GPU_REPORT.md").write_text(
        report,
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
