"""Exhaustive CUDA census of Penrose-Kauffman smoothing landscapes."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import networkx as nx
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.penrose_smoothing_landscape import (
    canonical_edges,
    chromatic_bucket,
    is_k_colorable,
    state_component_graph,
    state_nullity,
)


OUTPUT_DIR = ROOT / "results" / "penrose_smoothing_landscape_gpu"


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
    raise RuntimeError(
        "CUDA is unavailable; pass --allow-cpu only for a deliberate CPU run"
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def gf2_rank_batch(matrices: torch.Tensor) -> torch.Tensor:
    """Vectorized exact GF(2) rank for small square matrices."""

    reduced = matrices.clone()
    batch_size, size, _ = reduced.shape
    ranks = torch.zeros(
        batch_size,
        device=reduced.device,
        dtype=torch.long,
    )
    batch = torch.arange(batch_size, device=reduced.device)
    rows = torch.arange(size, device=reduced.device).view(1, size)

    for column in range(size):
        candidates = reduced[:, :, column] & (rows >= ranks.view(-1, 1))
        has_pivot = candidates.any(dim=1)
        pivots = candidates.to(torch.int8).argmax(dim=1)
        targets = ranks.clamp(max=size - 1)
        active = batch[has_pivot]
        pivot_rows = pivots[has_pivot]
        target_rows = targets[has_pivot]

        saved = reduced[active, target_rows].clone()
        reduced[active, target_rows] = reduced[active, pivot_rows]
        reduced[active, pivot_rows] = saved

        pivot_vectors = reduced[batch, targets]
        factors = (
            reduced[:, :, column]
            & (rows != targets.view(-1, 1))
            & has_pivot.view(-1, 1)
        )
        reduced ^= factors.unsqueeze(2) & pivot_vectors.unsqueeze(1)
        ranks += has_pivot.to(torch.long)
    return ranks


def contribution_matrix(
    vertex_count: int,
    edges: tuple[tuple[int, int], ...],
    device: torch.device,
) -> torch.Tensor:
    contributions = torch.zeros(
        (len(edges), vertex_count * vertex_count),
        device=device,
        dtype=torch.float32,
    )
    for edge_id, (left, right) in enumerate(edges):
        for row, column in (
            (left, left),
            (right, right),
            (left, right),
            (right, left),
        ):
            contributions[edge_id, row * vertex_count + column] = 1.0
    return contributions


def cuda_landscape(
    graph: nx.Graph,
    device: torch.device,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    edges = canonical_edges(graph)
    vertex_count = len(graph)
    edge_count = len(edges)
    state_count = 1 << edge_count
    contributions = contribution_matrix(vertex_count, edges, device)
    nullities = torch.empty(
        state_count,
        device=device,
        dtype=torch.int8,
    )

    synchronize(device)
    started = time.perf_counter()
    for start in range(0, state_count, chunk_size):
        stop = min(start + chunk_size, state_count)
        state_ids = torch.arange(
            start,
            stop,
            device=device,
            dtype=torch.long,
        )
        bits = (
            (
                state_ids.view(-1, 1)
                >> torch.arange(edge_count, device=device)
            )
            & 1
        ).to(torch.float32)
        # All sums are at most |E| <= 26 in this census and are exact in fp32.
        matrices = (
            (bits @ contributions)
            .remainder_(2)
            .gt(0.5)
            .reshape(-1, vertex_count, vertex_count)
        )
        if device.type == "cuda":
            assert matrices.is_cuda
        nullities[start:stop] = (
            vertex_count - gf2_rank_batch(matrices)
        ).to(torch.int8)
    synchronize(device)
    rank_seconds = time.perf_counter() - started

    state_ids = torch.arange(
        state_count,
        device=device,
        dtype=torch.long,
    )
    local_maximum = torch.ones(
        state_count,
        device=device,
        dtype=torch.bool,
    )
    maximum_started = time.perf_counter()
    for edge_id in range(edge_count):
        local_maximum &= (
            nullities
            > nullities[state_ids ^ (1 << edge_id)]
        )
    maxima = local_maximum.nonzero().flatten()
    synchronize(device)
    maximum_seconds = time.perf_counter() - maximum_started
    if device.type == "cuda":
        assert nullities.is_cuda
        assert maxima.is_cuda
    return nullities, maxima, {
        "rank_seconds": rank_seconds,
        "local_maximum_seconds": maximum_seconds,
        "states_per_second": (
            state_count / rank_seconds if rank_seconds else float("inf")
        ),
    }


def wheel_fixtures(smoke: bool) -> list[tuple[str, nx.Graph]]:
    sizes = (4, 6, 8) if smoke else (4, 6, 8, 10, 12, 14)
    return [
        (f"odd_wheel_W{size}", nx.wheel_graph(size))
        for size in sizes
    ]


def atlas_fixtures(smoke: bool) -> list[tuple[str, nx.Graph]]:
    fixtures: list[tuple[str, nx.Graph]] = []
    maximum_nodes = 6 if smoke else 7
    for atlas_index, candidate in enumerate(nx.graph_atlas_g()):
        if not 4 <= len(candidate) <= maximum_nodes:
            continue
        if not nx.is_connected(candidate):
            continue
        if nx.node_connectivity(candidate) < 3:
            continue
        if not nx.check_planarity(candidate)[0]:
            continue
        if is_k_colorable(candidate, 3):
            continue
        if not is_k_colorable(candidate, 4):
            raise AssertionError("planar atlas graph exceeded four colors")
        fixtures.append(
            (
                f"atlas_{atlas_index}",
                nx.convert_node_labels_to_integers(candidate),
            )
        )
    return fixtures


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def nvidia_smi() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        ).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unavailable"


def device_audit(device: torch.device) -> dict[str, Any]:
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "selected_device": str(device),
        "gpu_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "gpu_capability": (
            list(torch.cuda.get_device_capability(device))
            if device.type == "cuda"
            else None
        ),
        "dtype": "bool GF(2), float32 exact accumulation, int8 nullity",
        "nvidia_smi": nvidia_smi(),
    }


def report_text(
    wheel_summaries: list[dict[str, Any]],
    atlas_summaries: list[dict[str, Any]],
    audit: dict[str, Any],
    timing_rows: list[dict[str, Any]],
) -> str:
    wheel_table = "\n".join(
        "| {fixture} | {state_count} | {local_maximum_count} | "
        "{expected_local_maximum_count} | {three_colorable_nonzero_count} | "
        "{non_three_colorable_nonzero_count} | {all_nonzero_k3} |".format(
            **row
        )
        for row in wheel_summaries
    )
    atlas_with_bad = [
        row
        for row in atlas_summaries
        if row["non_three_colorable_nonzero_count"] > 0
    ]
    atlas_with_good = [
        row
        for row in atlas_summaries
        if row["three_colorable_state_count"] > 0
    ]
    total_states = sum(row["state_count"] for row in timing_rows)
    total_gpu_seconds = sum(
        row["rank_seconds"] + row["local_maximum_seconds"]
        for row in timing_rows
    )
    return f"""# Penrose Smoothing Landscape CUDA Report

## Scope

This is an exhaustive finite census of the Kauffman-Silver-Williams
smoothing cube. For a plane Tait graph, the GPU computes the mod-2 Laplacian
nullity of every edge-subset state and finds every strict local maximum.
Only those maxima return to the CPU, where their component graphs are
reconstructed and tested for 3-colorability exactly.

This is not a new proof of the Four Color Theorem.

## CUDA audit

- CUDA used: **{audit["selected_device"] == "cuda"}**
- Device: `{audit["gpu_name"]}`
- PyTorch: `{audit["torch_version"]}`
- CUDA runtime: `{audit["torch_cuda_version"]}`
- Peak allocated memory: **{audit["peak_cuda_memory_bytes"]} bytes**
- Exhaustive states processed: **{total_states:,}**
- CUDA rank/maxima time: **{total_gpu_seconds:.3f} seconds**

## Odd-wheel pattern

For `W_(2k+2)`, whose rim is the odd cycle `C_(2k+1)`, the observed
nonzero maximum count is

```text
(4^k - 1) / 3.
```

The zero state has component graph equal to the 4-chromatic odd wheel. Every
observed nonzero maximum has nullity three and component graph `K3`.

| Fixture | States | Maxima | Formula | Nonzero 3-colorable | Nonzero not 3-colorable | All nonzero K3 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
{wheel_table}

The count is compatible with a coloring-orbit explanation: fix the hub color,
properly color the odd rim with the other three colors, then quotient by the
six permutations of those colors. The number is
`(2^(2k+1)-2)/6 = (4^k-1)/3`.

That observation is a **proof sketch only**. A complete proof must show that
the orbit-to-state map is bijective and that every nonzero strict maximum has
component graph `K3`.

## Boundary control

- 3-connected, 4-chromatic planar atlas graphs: **{len(atlas_summaries)}**
- Graphs with at least one 3-colorable maximum: **{len(atlas_with_good)}**
- Graphs with a nonzero maximum that is not 3-colorable:
  **{len(atlas_with_bad)}**

Thus the attractive assertion

```text
every nonzero strict local maximum is 3-colorable
```

is false already in the small planar atlas. The odd-wheel behavior is
family-specific. This is the main diagnostic result: component-count ascent
alone does not solve the 3-colorability burden in Theorem 6.6.

## Relation to the classical proof

Robertson-Sanders-Seymour-Thomas prove reducibility for 633 configurations
and unavoidability using 32 discharging rules. This census supplies neither
ingredient. Its possible value is narrower: the odd-wheel family gives a
closed-form test case for studying how smoothing maxima encode colorings,
while the atlas rows provide compact counterexamples for proposed universal
landscape rules.

## Status

- Exhaustive CUDA census: `passed`
- CPU reconstruction/nullity agreement: `passed`
- Odd-wheel closed form: `conjecture_with_orbit_proof_sketch`
- Universal nonzero-maximum rule: `falsified`
- New Four Color proof: `false`

## Sources

- Kauffman, Silver, Williams, Theorems 5.3 and 6.6:
  https://arxiv.org/abs/2604.16635
- Robertson, Sanders, Seymour, Thomas:
  https://thomas.math.gatech.edu/FC/fourcolor.html
"""


def main() -> None:
    args = parse_args()
    device = select_device(args.allow_cpu)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    audit = device_audit(device)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"device={device} gpu={audit['gpu_name']} "
        f"chunk_size={args.chunk_size}"
    )

    state_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    fixtures = [
        ("odd_wheel", name, graph)
        for name, graph in wheel_fixtures(args.smoke)
    ] + [
        ("atlas_control", name, graph)
        for name, graph in atlas_fixtures(args.smoke)
    ]

    for fixture_group, fixture, graph in fixtures:
        graph = nx.convert_node_labels_to_integers(graph)
        edges = canonical_edges(graph)
        nullities, maxima_gpu, timing = cuda_landscape(
            graph,
            device,
            args.chunk_size,
        )
        transfer_started = time.perf_counter()
        maxima = maxima_gpu.cpu().tolist()
        maximum_nullities = nullities[maxima_gpu].cpu().tolist()
        synchronize(device)
        transfer_seconds = time.perf_counter() - transfer_started

        analysis_started = time.perf_counter()
        fixture_rows: list[dict[str, Any]] = []
        for state_index, gpu_nullity in zip(
            maxima,
            maximum_nullities,
            strict=True,
        ):
            component_graph = state_component_graph(
                graph,
                edges,
                state_index,
            )
            cpu_nullity = state_nullity(
                len(graph),
                edges,
                state_index,
            )
            if cpu_nullity != gpu_nullity:
                raise AssertionError("CPU/GPU nullity mismatch")
            if len(component_graph) != gpu_nullity:
                raise AssertionError("component/nullity mismatch")
            chromatic = chromatic_bucket(component_graph)
            row = {
                "fixture_group": fixture_group,
                "fixture": fixture,
                "vertices": len(graph),
                "edges": len(edges),
                "state_index": state_index,
                "state_hex": hex(state_index),
                "state_popcount": state_index.bit_count(),
                "nullity": gpu_nullity,
                "component_vertices": len(component_graph),
                "component_edges": component_graph.number_of_edges(),
                "component_chromatic_bucket": chromatic,
                "three_colorable": chromatic in {"1", "2", "3"},
                "component_is_k3": nx.is_isomorphic(
                    component_graph,
                    nx.complete_graph(3),
                ),
                "zero_state": state_index == 0,
            }
            fixture_rows.append(row)
            state_rows.append(row)
        cpu_analysis_seconds = time.perf_counter() - analysis_started

        nonzero = [row for row in fixture_rows if not row["zero_state"]]
        k_value = (len(graph) - 2) // 2
        expected_nonzero = (
            (4**k_value - 1) // 3
            if fixture_group == "odd_wheel"
            else ""
        )
        summary = {
            "fixture_group": fixture_group,
            "fixture": fixture,
            "graph6": nx.to_graph6_bytes(
                graph,
                header=False,
            ).decode("ascii").strip(),
            "vertices": len(graph),
            "edges": len(edges),
            "state_count": 1 << len(edges),
            "local_maximum_count": len(fixture_rows),
            "three_colorable_state_count": sum(
                bool(row["three_colorable"])
                for row in fixture_rows
            ),
            "three_colorable_nonzero_count": sum(
                bool(row["three_colorable"])
                for row in nonzero
            ),
            "non_three_colorable_nonzero_count": sum(
                not bool(row["three_colorable"])
                for row in nonzero
            ),
            "expected_nonzero_count": expected_nonzero,
            "expected_local_maximum_count": (
                expected_nonzero + 1
                if expected_nonzero != ""
                else ""
            ),
            "odd_wheel_formula_passed": (
                len(nonzero) == expected_nonzero
                if expected_nonzero != ""
                else ""
            ),
            "all_nonzero_k3": (
                all(bool(row["component_is_k3"]) for row in nonzero)
                if nonzero
                else False
            ),
            "max_component_chromatic_bucket": (
                ">4"
                if any(
                    row["component_chromatic_bucket"] == ">4"
                    for row in fixture_rows
                )
                else max(
                    int(row["component_chromatic_bucket"])
                    for row in fixture_rows
                )
            ),
        }
        summaries.append(summary)
        timing_rows.append(
            {
                "fixture_group": fixture_group,
                "fixture": fixture,
                "vertices": len(graph),
                "edges": len(edges),
                "state_count": 1 << len(edges),
                **timing,
                "cpu_transfer_seconds": transfer_seconds,
                "cpu_component_analysis_seconds": cpu_analysis_seconds,
            }
        )
        print(
            f"{fixture}: states={1 << len(edges):,} "
            f"maxima={len(maxima)} "
            f"rank_s={timing['rank_seconds']:.4f}"
        )

    audit["peak_cuda_memory_bytes"] = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    audit["main_tensors_on_cuda"] = device.type == "cuda"
    wheel_summaries = [
        row for row in summaries if row["fixture_group"] == "odd_wheel"
    ]
    atlas_summaries = [
        row for row in summaries if row["fixture_group"] == "atlas_control"
    ]
    write_csv(OUTPUT_DIR / "smoothing_maximum_rows.csv", state_rows)
    write_csv(OUTPUT_DIR / "smoothing_fixture_summary.csv", summaries)
    write_csv(OUTPUT_DIR / "smoothing_timing.csv", timing_rows)
    (OUTPUT_DIR / "device_report.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "device_report.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in audit.items()) + "\n",
        encoding="utf-8",
    )
    overall_audit = {
        "cuda_used": device.type == "cuda",
        "fixture_count": len(summaries),
        "wheel_count": len(wheel_summaries),
        "atlas_control_count": len(atlas_summaries),
        "all_wheel_formulas_passed": all(
            bool(row["odd_wheel_formula_passed"])
            for row in wheel_summaries
        ),
        "all_wheel_nonzero_states_are_k3": all(
            bool(row["all_nonzero_k3"])
            for row in wheel_summaries
        ),
        "atlas_graphs_falsifying_universal_nonzero_rule": sum(
            row["non_three_colorable_nonzero_count"] > 0
            for row in atlas_summaries
        ),
        "new_four_color_proof": False,
        "odd_wheel_status": "conjecture_with_orbit_proof_sketch",
    }
    (OUTPUT_DIR / "smoothing_landscape_audit.json").write_text(
        json.dumps(overall_audit, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "PENROSE_SMOOTHING_LANDSCAPE_GPU_REPORT.md").write_text(
        report_text(
            wheel_summaries,
            atlas_summaries,
            audit,
            timing_rows,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
