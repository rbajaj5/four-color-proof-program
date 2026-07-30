"""Exhaustive CUDA audit of topology-preserving Y-board majority reduction."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path
import subprocess
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.y_majority_coarse_grain import triangular_coordinates  # noqa: E402


OUTPUT_DIR = ROOT / "results" / "y_majority_coarse_grain_gpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=262_144)
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


def bit_mask(size: int, predicate) -> int:
    return sum(
        1 << (q * size + r)
        for q, r in triangular_coordinates(size)
        if predicate(q, r)
    )


def valid_mask(size: int) -> int:
    return bit_mask(size, lambda _q, _r: True)


def neighbor_shift_specs(size: int) -> tuple[tuple[int, int], ...]:
    directions = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1))
    specs = []
    coordinates = set(triangular_coordinates(size))
    for dq, dr in directions:
        source_mask = 0
        for q, r in coordinates:
            if (q + dq, r + dr) in coordinates:
                source_mask |= 1 << (q * size + r)
        specs.append((source_mask, dq * size + dr))
    return tuple(specs)


def expand_neighbors(
    reach: torch.Tensor,
    specs: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    expanded = reach
    for source_mask, shift in specs:
        source = reach & source_mask
        expanded = (
            expanded | (source << shift)
            if shift >= 0
            else expanded | (source >> -shift)
        )
    return expanded


def flood(
    active: torch.Tensor,
    seed_mask: int,
    *,
    size: int,
    specs: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    reach = active & seed_mask
    for _ in range(len(triangular_coordinates(size))):
        reach = expand_neighbors(reach, specs) & active
    return reach


def has_y(
    active: torch.Tensor,
    *,
    size: int,
    specs: tuple[tuple[int, int], ...],
    side_r0: int,
    side_diag: int,
) -> torch.Tensor:
    """Test whether one component, not a union of components, meets all sides."""

    winner = torch.zeros_like(active, dtype=torch.bool)
    for r in range(size):
        reach = flood(active, 1 << r, size=size, specs=specs)
        winner |= (reach & side_r0).ne(0) & (reach & side_diag).ne(0)
    return winner


def compact_to_grid(
    compact_states: torch.Tensor,
    size: int,
) -> torch.Tensor:
    board = torch.zeros_like(compact_states)
    for compact_index, (q, r) in enumerate(triangular_coordinates(size)):
        board |= ((compact_states >> compact_index) & 1) << (q * size + r)
    return board


def majority_reduce_grid(board: torch.Tensor, size: int) -> torch.Tensor:
    reduced = torch.zeros_like(board)
    for q, r in triangular_coordinates(size - 1):
        tips = (
            (q + 1) * size + r,
            q * size + r,
            q * size + r + 1,
        )
        majority = sum((board >> index) & 1 for index in tips) >= 2
        reduced |= majority.to(torch.int64) << (q * (size - 1) + r)
    return reduced


def reduce_to_single(board: torch.Tensor, size: int) -> torch.Tensor:
    current = board
    for current_size in range(size, 1, -1):
        current = majority_reduce_grid(current, current_size)
    return (current & 1).bool()


def popcount_compact(states: torch.Tensor, cell_count: int) -> torch.Tensor:
    count = torch.zeros_like(states)
    for index in range(cell_count):
        count += (states >> index) & 1
    return count


def census(size: int, device: torch.device, chunk_size: int) -> dict[str, object]:
    cell_count = len(triangular_coordinates(size))
    state_count = 1 << cell_count
    specs = neighbor_shift_specs(size)
    full = valid_mask(size)
    side_r0 = bit_mask(size, lambda _q, r: r == 0)
    side_diag = bit_mask(size, lambda q, r: q + r == size - 1)
    counts = torch.zeros(6, device=device, dtype=torch.int64)
    hist = torch.zeros(
        (2, cell_count + 1),
        device=device,
        dtype=torch.int64,
    )

    synchronize(device)
    started = time.perf_counter()
    for start in range(0, state_count, chunk_size):
        stop = min(start + chunk_size, state_count)
        compact = torch.arange(start, stop, device=device, dtype=torch.int64)
        blue = compact_to_grid(compact, size)
        yellow = blue ^ full
        blue_y = has_y(
            blue,
            size=size,
            specs=specs,
            side_r0=side_r0,
            side_diag=side_diag,
        )
        yellow_y = has_y(
            yellow,
            size=size,
            specs=specs,
            side_r0=side_r0,
            side_diag=side_diag,
        )
        reduced_blue = reduce_to_single(blue, size)
        mismatch = reduced_blue ^ blue_y
        both = blue_y & yellow_y
        neither = ~(blue_y | yellow_y)
        counts += torch.stack(
            (
                blue_y.sum(),
                yellow_y.sum(),
                both.sum(),
                neither.sum(),
                mismatch.sum(),
                torch.tensor(stop - start, device=device),
            )
        )
        weights = popcount_compact(compact, cell_count)
        hist[0] += torch.bincount(
            weights[blue_y],
            minlength=cell_count + 1,
        )
        hist[1] += torch.bincount(
            weights[yellow_y],
            minlength=cell_count + 1,
        )
    synchronize(device)
    elapsed = time.perf_counter() - started
    counts_cpu = counts.cpu().tolist()
    hist_cpu = hist.cpu().tolist()
    return {
        "size": size,
        "cell_count": cell_count,
        "state_count": state_count,
        "blue_y_count": counts_cpu[0],
        "yellow_y_count": counts_cpu[1],
        "both_count": counts_cpu[2],
        "neither_count": counts_cpu[3],
        "reduction_mismatch_count": counts_cpu[4],
        "processed_count": counts_cpu[5],
        "elapsed_seconds": elapsed,
        "states_per_second": state_count / elapsed,
        "histogram": hist_cpu,
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


def main() -> None:
    args = parse_args()
    device = select_device(args.allow_cpu)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sizes = range(1, 5) if args.smoke else range(1, 7)
    summaries = []
    histogram_rows: list[dict[str, object]] = []
    print(
        f"device={device} gpu="
        f"{torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}"
    )
    for size in sizes:
        summary = census(size, device, args.chunk_size)
        summaries.append({key: value for key, value in summary.items() if key != "histogram"})
        for blue_cells in range(summary["cell_count"] + 1):
            histogram_rows.append(
                {
                    "size": size,
                    "blue_cell_count": blue_cells,
                    "blue_y_count": summary["histogram"][0][blue_cells],
                    "yellow_y_count": summary["histogram"][1][blue_cells],
                }
            )
        print(
            f"Y{size}: states={summary['state_count']:,} "
            f"mismatch={summary['reduction_mismatch_count']} "
            f"seconds={summary['elapsed_seconds']:.4f}"
        )

    write_csv(OUTPUT_DIR / "y_census_summary.csv", summaries)
    write_csv(OUTPUT_DIR / "y_census_by_blue_count.csv", histogram_rows)
    peak_memory = (
        torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0
    )
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
        "peak_cuda_memory_bytes": peak_memory,
        "nvidia_smi": nvidia_smi(),
    }
    (OUTPUT_DIR / "device_report.json").write_text(
        json.dumps(device_report, indent=2) + "\n",
        encoding="utf-8",
    )

    audit = {
        "cuda_used": device.type == "cuda",
        "largest_board_size": max(sizes),
        "total_states": sum(int(row["state_count"]) for row in summaries),
        "all_states_processed": all(
            row["state_count"] == row["processed_count"] for row in summaries
        ),
        "no_both_winners": all(row["both_count"] == 0 for row in summaries),
        "no_neither_winners": all(row["neither_count"] == 0 for row in summaries),
        "majority_reduction_preserves_y": all(
            row["reduction_mismatch_count"] == 0 for row in summaries
        ),
        "new_hex_theorem": False,
        "three_dimensional_extension": False,
    }
    (OUTPUT_DIR / "y_majority_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )

    table = "\n".join(
        "| {size} | {cell_count} | {state_count} | {blue_y_count} | "
        "{yellow_y_count} | {both_count} | {neither_count} | "
        "{reduction_mismatch_count} | {elapsed_seconds:.4f} |".format(**row)
        for row in summaries
    )
    report = f"""# Y-Majority Coarse-Graining CUDA Report

## Exact finite audit

The triangular board is

```text
{{(q,r): q>=0, r>=0, q+r<n}}.
```

One reduction step replaces the triangle

```text
(q+1,r), (q,r), (q,r+1)
```

by its majority color. The GPU exhausts every coloring through side length
**{max(sizes)}** and compares the recursively reduced one-cell color with a
direct three-boundary connectivity test.

| n | cells | states | blue Y | yellow Y | both | neither | mismatch | seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

All tested boards have exactly one monochromatic Y, and majority reduction
preserves its color.

## Coarse-graining interpretation

This supplies an exact planar renormalization primitive. Threshold a scalar
field on a triangular hexagonal slice into high/low cells. With the declared
three-side boundary geometry, recursive majority reduction preserves which
phase has a three-arm connection. For a two-terminal Hex geometry, the same
interface argument certifies a high-phase barrier or a low-phase dual
corridor.

That is useful for curvature or flux-tube slices: it preserves a declared
connectivity event while reducing resolution. It is substantially stronger
than replacing blocks by their average and hoping topology survives.

## Boundaries

- This is the Karlin-Peres Hex/Y theorem made executable, not a new theorem.
- The certificate is planar. In three dimensions, both phases may percolate,
  and there is no analogous exclusivity without additional hypotheses.
- It preserves the Y-connectivity predicate, not knot type, linking number,
  magnetic helicity, or the Four Color Theorem.
- A physical application must preregister the slice, threshold, boundary
  arcs, and refinement rule.

## Source

Anna R. Karlin and Yuval Peres, *Game Theory, Alive*, Sections 1.2.1-1.2.3,
American Mathematical Society. The supplied excerpt contains the oriented
Hex-interface proof and the majority-triangle reduction from Y boards of
side `n` to side `n-1`.
"""
    (OUTPUT_DIR / "Y_MAJORITY_COARSE_GRAIN_GPU_REPORT.md").write_text(
        report,
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
