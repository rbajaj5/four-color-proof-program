"""Run CUDA diagnostics for exact Bateman Maxwell knot fields.

This experiment samples known analytic torus-knot core lines.  It does not
search unrestricted Maxwell initial data and does not claim discovery of new
knot types.
"""

from __future__ import annotations

import argparse
import csv
from fractions import Fraction
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
import networkx as nx
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.goemans_conflict_fixture import (
    DETOUR_FRACTIONS,
    edge_relaxation_is_feasible,
    fractional_direct_cost,
    integral_rows,
    minimum_integral_cost,
    missing_triangle_inequality_slack,
)
from src.maxwell_knot_fields import (
    all_magnetic_core_curves,
    core_tangent_alignment,
    electric_magnetic_fields,
    expected_core_component_count,
    maxwell_residuals,
)


OUTPUT_DIR = ROOT / "results" / "maxwell_knot_fields"
FIXTURES = (
    ("hopfion_1_1", 1, 1, "two linked core rings"),
    ("linked_rings_1_2", 1, 2, "two linked core rings"),
    ("trefoil_2_3", 2, 3, "two linked trefoil core knots"),
    ("cinquefoil_2_5", 2, 5, "two linked cinquefoil core knots"),
    ("four_component_2_2", 2, 2, "four linked core rings"),
    ("torus_knot_3_4", 3, 4, "two linked T(3,4) core knots"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--residual-points", type=int, default=2048)
    parser.add_argument("--finite-difference-step", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def selected_device(allow_cpu: bool) -> torch.device:
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


def rotation_matrix() -> np.ndarray:
    azimuth = math.radians(23.0)
    elevation = math.radians(31.0)
    rotate_z = np.array(
        [
            [math.cos(azimuth), -math.sin(azimuth), 0.0],
            [math.sin(azimuth), math.cos(azimuth), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotate_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(elevation), -math.sin(elevation)],
            [0.0, math.sin(elevation), math.cos(elevation)],
        ]
    )
    return rotate_x @ rotate_z


def cross_2d(left: np.ndarray, right: np.ndarray) -> float:
    return float(left[0] * right[1] - left[1] * right[0])


def projection_crossings(curve: np.ndarray) -> list[dict[str, float | int]]:
    """Find transverse self-crossings in a fixed generic projection."""

    unique = curve[:-1] @ rotation_matrix().T
    count = len(unique)
    rows: list[dict[str, float | int]] = []
    for first in range(count):
        first_next = (first + 1) % count
        p0 = unique[first, :2]
        p1 = unique[first_next, :2]
        first_vector = p1 - p0
        for second in range(first + 1, count):
            second_next = (second + 1) % count
            cyclic_distance = min(
                (second - first) % count,
                (first - second) % count,
            )
            if cyclic_distance <= 2:
                continue
            q0 = unique[second, :2]
            q1 = unique[second_next, :2]
            second_vector = q1 - q0
            denominator = cross_2d(first_vector, second_vector)
            if abs(denominator) < 1e-12:
                continue
            displacement = q0 - p0
            first_parameter = cross_2d(displacement, second_vector) / denominator
            second_parameter = cross_2d(displacement, first_vector) / denominator
            endpoint_margin = 1e-7
            if not (
                endpoint_margin < first_parameter < 1.0 - endpoint_margin
                and endpoint_margin < second_parameter < 1.0 - endpoint_margin
            ):
                continue
            first_z = (
                unique[first, 2]
                + first_parameter
                * (unique[first_next, 2] - unique[first, 2])
            )
            second_z = (
                unique[second, 2]
                + second_parameter
                * (unique[second_next, 2] - unique[second, 2])
            )
            angle_sine = abs(denominator) / (
                np.linalg.norm(first_vector)
                * np.linalg.norm(second_vector)
            )
            rows.append(
                {
                    "first_segment": first,
                    "second_segment": second,
                    "angle_sine": float(angle_sine),
                    "depth_gap": float(abs(first_z - second_z)),
                }
            )
    return rows


def nonlocal_self_distance(curve: np.ndarray) -> float:
    unique = curve[:-1]
    count = len(unique)
    minimum = math.inf
    for index in range(count):
        differences = unique - unique[index]
        distances = np.linalg.norm(differences, axis=1)
        cyclic = np.minimum(
            (np.arange(count) - index) % count,
            (index - np.arange(count)) % count,
        )
        distances[cyclic <= 3] = math.inf
        minimum = min(minimum, float(np.min(distances)))
    return minimum


def crossing_conflict_graph(
    crossings: list[dict[str, float | int]],
    segment_count: int,
) -> nx.Graph:
    """Build a local-overlap graph for projection diagnostic work.

    Two crossing checks conflict when their one-segment neighborhoods overlap.
    This graph schedules numerical checks; it is not a knot invariant.
    """

    graph = nx.Graph()
    neighborhoods: list[set[int]] = []
    for crossing_id, crossing in enumerate(crossings):
        graph.add_node(crossing_id)
        affected: set[int] = set()
        for key in ("first_segment", "second_segment"):
            segment = int(crossing[key])
            affected.update(
                {
                    (segment - 1) % segment_count,
                    segment,
                    (segment + 1) % segment_count,
                }
            )
        neighborhoods.append(affected)
    for left in range(len(crossings)):
        for right in range(left + 1, len(crossings)):
            if neighborhoods[left] & neighborhoods[right]:
                graph.add_edge(left, right)
    return graph


def chromatic_number_up_to_four(graph: nx.Graph) -> int | None:
    if graph.number_of_nodes() == 0:
        return 0
    order = sorted(graph.nodes(), key=graph.degree, reverse=True)
    assignments: dict[int, int] = {}

    def can_color(position: int, color_count: int) -> bool:
        if position == len(order):
            return True
        vertex = order[position]
        forbidden = {
            assignments[neighbor]
            for neighbor in graph.neighbors(vertex)
            if neighbor in assignments
        }
        for color in range(color_count):
            if color in forbidden:
                continue
            assignments[vertex] = color
            if can_color(position + 1, color_count):
                return True
            del assignments[vertex]
        return False

    for color_count in range(1, 5):
        assignments.clear()
        if can_color(0, color_count):
            return color_count
    return None


def render_fixture(
    fixture_id: str,
    p: int,
    q: int,
    curves: tuple[torch.Tensor, ...],
    output_dir: Path,
) -> str:
    figure = plt.figure(figsize=(7.2, 6.4))
    axis = figure.add_subplot(111, projection="3d")
    palette = ("#0057b8", "#d81b60", "#00876c", "#e67e22", "#6f42c1", "#444444")
    all_points = []
    for index, curve in enumerate(curves):
        points = curve.detach().cpu().numpy()
        all_points.append(points)
        axis.plot(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color=palette[index % len(palette)],
            linewidth=1.8,
        )
    combined = np.concatenate(all_points, axis=0)
    center = combined.mean(axis=0)
    radius = 0.55 * float(np.max(np.ptp(combined, axis=0)))
    radius = max(radius, 1e-3)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    axis.set_title(f"Bateman magnetic core lines: T({p},{q})")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    filename = f"{fixture_id}_core_lines.png"
    figure.savefig(output_dir / filename, dpi=180)
    plt.close(figure)
    return filename


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_report(
    *,
    device: torch.device,
    elapsed_seconds: float,
    fixture_rows: list[dict[str, Any]],
    residual_rows: list[dict[str, Any]],
    audit: dict[str, Any],
) -> str:
    fixture_table = "\n".join(
        "| {fixture_id} | ({p},{q}) | {expected_core_components} | "
        "{projection_crossings} | {minimum_tangent_alignment:.8f} | "
        "{projection_status} |".format(**row)
        for row in fixture_rows
    )
    residual_table = "\n".join(
        "| {fixture_id} | {relative_divergence_magnetic_mean:.3e} | "
        "{relative_faraday_mean:.3e} | {relative_ampere_mean:.3e} | "
        "{null_dot_mean_abs:.3e} |".format(**row)
        for row in residual_rows
    )
    return f"""# Maxwell Knot Field Experiment

## Scope

This is a finite numerical audit of a published analytic family. It does not
claim a new Maxwell solution, a new knot type, or a new Four Color proof.

Kedia et al. start with Bateman potentials

```text
alpha = (r^2 - t^2 - 1 + 2 i z) / (r^2 - (t-i)^2)
beta  = 2(x - i y) / (r^2 - (t-i)^2)
F     = E + iB = grad(alpha^p) x grad(beta^q).
```

For positive integers `(p,q)`, the magnetic core lines are known `(p,q)`
torus knots or links. The script evaluates these equations directly and
checks them numerically on `{device}`.

## CUDA Audit

- Actual device: `{device}`
- CUDA used: `{device.type == "cuda"}`
- GPU: `{audit["gpu_name"]}`
- Torch: `{audit["torch_version"]}`
- Dtype: `float64 / complex128`
- Peak CUDA allocation: `{audit["peak_cuda_memory_bytes"]}` bytes
- Total elapsed: `{elapsed_seconds:.3f}` seconds

## Geometric Fixtures

| Fixture | `(p,q)` | Core components | Projected crossings | Min `|cos(B,tangent)|` | Projection diagnostic |
| --- | --- | ---: | ---: | ---: | --- |
{fixture_table}

The fixed projection is used only to expose numerical degeneracies. Crossing
count in this table is not asserted to be minimal. The local crossing-conflict
graph colors overlapping diagnostic tasks; its color count is not a knot
invariant.

## Maxwell Residuals

| Fixture | `div B` relative mean | Faraday relative mean | Ampere relative mean | `E dot B` mean |
| --- | ---: | ---: | ---: | ---: |
{residual_table}

These are centered finite-difference residuals. The equations themselves are
analytic consequences of Bateman's construction.

## Static Permanent-Magnet Boundary

McDonald's note distinguishes a knotted material magnet from an arbitrary
knotted field in its exterior. In a simply connected, current-free, static
region, `curl B = 0` and `div B = 0`, hence `B = grad Phi` for harmonic
`Phi`. Along a nonstationary field line,

```text
d Phi / ds = |grad Phi|^2 > 0.
```

Such a gradient line cannot be a closed orbit. Closed knotted magnetic lines
therefore require a source/current region, nontrivial domain topology,
time-dependent null fields such as the Bateman family, or passage through
field zeros where reconnection can occur. This is why blindly varying a
static dipole parameter is the wrong search space.

## Sanders-Style Diagnostic Analogy

The Robertson-Sanders-Seymour-Thomas proof has two logically separate jobs:
its configurations are reducible, and the collection is unavoidable. Here we
borrow only that audit architecture. The finite bad-configuration catalog is:

1. near-zero field magnitude;
2. failure of core-line closure;
3. loss of field/tangent alignment;
4. near-tangent projected crossings;
5. nearly equal crossing depths; and
6. a nonplanar overlap graph for local diagnostic tasks.

Passing this catalog is a fixture check, not an unavoidability theorem for all
Maxwell fields. The 633 Four Color configurations are not being transferred
to electromagnetism.

## Goemans Conflict-Core Test

The exact abstraction uses a triangle of pairwise-incompatible detours. The
fractional point `(1/3, 2/5, 1/3)` satisfies every edge inequality and has
cost `58`. Every integral selection uses at most one detour and costs at least
`60`. The triangle is planar and 3-colorable.

This gives a clean negative lesson for the field-line analogy: coloring a
planar conflict graph can schedule whole-route checks, but does not make the
edge relaxation integral or preserve fractional cost. The missing inequality
is the odd-cycle constraint `z1 + z2 + z3 <= 1`, violated by `1/15`.

The repository verifies only this abstract conflict core. The recently
reported directed-path counterexample to the cost-strengthened Goemans
conjecture is not yet treated here as peer-reviewed mathematical fact.

## What Changed in the Proof Program

- **Upgraded to exact fixture:** Bateman `(p,q)` values provide a canonical
  Maxwell-constrained geometry generator, replacing arbitrary 3D knot
  drawings.
- **Falsified as a general bridge:** Four-colorability alone cannot certify a
  cost-preserving whole-field-line selection.
- **Still open:** whether a useful unavoidable family of local geometric
  pathologies can support a reduction theorem for a physically constrained
  class of knotted fields.

## Sources

- K. T. McDonald, *Can the Field Lines of a Permanent Magnet Be Tied in
  Knots?*: https://kirkmcd.princeton.edu/examples/knot.pdf
- H. Kedia et al., *Tying knots in light fields*:
  https://arxiv.org/abs/1302.0342
- H. Kedia, D. Peralta-Salas, and W. T. M. Irvine, *When do knots in light
  stay knotted?*: https://arxiv.org/abs/1706.06175
- N. Robertson, D. Sanders, P. Seymour, and R. Thomas, Four Color proof
  materials: https://thomas.math.gatech.edu/FC/fourcolor.html
- Y. Dinitz, N. Garg, and M. X. Goemans, *On the single-source unsplittable
  flow problem*: https://doi.org/10.1007/s004930050043
- Current planar SSUF formulation and Goemans Conjecture 1.3:
  https://doi.org/10.1007/s10107-026-02365-x
"""


def main() -> None:
    args = parse_args()
    if args.samples < 64 or args.residual_points < 16:
        raise ValueError("samples >= 64 and residual-points >= 16 are required")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = selected_device(args.allow_cpu)
    print(f"device={device} torch={torch.__version__}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    start = time.perf_counter()
    fixture_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    pathology_rows: list[dict[str, Any]] = []

    for fixture_id, p, q, description in FIXTURES:
        group_start = time.perf_counter()
        curves = all_magnetic_core_curves(
            p,
            q,
            args.samples,
            device=device,
            dtype=torch.float64,
        )
        synchronize(device)
        primary = curves[0]
        alignment = core_tangent_alignment(primary, p, q)
        _, magnetic = electric_magnetic_fields(primary[:-1], p=p, q=q)
        closure_errors = [
            float(torch.linalg.vector_norm(curve[-1] - curve[0]).item())
            for curve in curves
        ]
        minimum_field_norm = float(
            torch.linalg.vector_norm(magnetic, dim=-1).min().item()
        )
        primary_numpy = primary.detach().cpu().numpy()
        crossings = projection_crossings(primary_numpy)
        conflict_graph = crossing_conflict_graph(
            crossings,
            args.samples,
        )
        planar, _ = nx.check_planarity(conflict_graph)
        color_count = (
            chromatic_number_up_to_four(conflict_graph) if planar else None
        )
        near_tangent_count = sum(
            float(row["angle_sine"]) < 0.05 for row in crossings
        )
        ambiguous_depth_count = sum(
            float(row["depth_gap"]) < 1e-3 for row in crossings
        )
        minimum_angle_sine = min(
            (float(row["angle_sine"]) for row in crossings),
            default=1.0,
        )
        minimum_depth_gap = min(
            (float(row["depth_gap"]) for row in crossings),
            default=math.inf,
        )
        projection_status = (
            "pass"
            if (
                max(closure_errors) < 1e-9
                and float(alignment.min().item()) > 0.99
                and minimum_field_norm > 1e-10
                and near_tangent_count == 0
                and ambiguous_depth_count == 0
                and planar
                and color_count is not None
            )
            else "warning"
        )
        figure_file = render_fixture(
            fixture_id,
            p,
            q,
            curves,
            output_dir,
        )

        random_points = (
            3.0
            * torch.rand(
                args.residual_points,
                3,
                generator=generator,
                device=device,
                dtype=torch.float64,
            )
            - 1.5
        )
        residual = maxwell_residuals(
            random_points,
            p,
            q,
            step=args.finite_difference_step,
        )
        synchronize(device)
        fixture_rows.append(
            {
                "fixture_id": fixture_id,
                "p": p,
                "q": q,
                "description": description,
                "expected_core_components": expected_core_component_count(p, q),
                "sampled_core_components": len(curves),
                "samples_per_component": args.samples,
                "maximum_closure_error": max(closure_errors),
                "mean_tangent_alignment": float(alignment.mean().item()),
                "minimum_tangent_alignment": float(alignment.min().item()),
                "minimum_core_field_norm": minimum_field_norm,
                "minimum_nonlocal_self_distance": nonlocal_self_distance(
                    primary_numpy
                ),
                "projection_crossings": len(crossings),
                "near_tangent_crossings": near_tangent_count,
                "ambiguous_depth_crossings": ambiguous_depth_count,
                "minimum_crossing_angle_sine": minimum_angle_sine,
                "minimum_crossing_depth_gap": (
                    minimum_depth_gap if math.isfinite(minimum_depth_gap) else ""
                ),
                "diagnostic_conflict_graph_nodes": conflict_graph.number_of_nodes(),
                "diagnostic_conflict_graph_edges": conflict_graph.number_of_edges(),
                "diagnostic_conflict_graph_planar": planar,
                "diagnostic_chromatic_number_up_to_4": (
                    color_count if color_count is not None else ""
                ),
                "projection_status": projection_status,
                "figure_file": figure_file,
                "elapsed_seconds": time.perf_counter() - group_start,
            }
        )
        residual_rows.append(
            {
                "fixture_id": fixture_id,
                "p": p,
                "q": q,
                "finite_difference_step": args.finite_difference_step,
                "residual_points": args.residual_points,
                **residual,
            }
        )
        pathology_rows.append(
            {
                "fixture_id": fixture_id,
                "near_zero_field": minimum_field_norm <= 1e-10,
                "closure_failure": max(closure_errors) >= 1e-9,
                "alignment_failure": float(alignment.min().item()) <= 0.99,
                "near_tangent_projection": near_tangent_count > 0,
                "ambiguous_over_under_depth": ambiguous_depth_count > 0,
                "nonplanar_diagnostic_overlap_graph": not planar,
                "catalog_passed": projection_status == "pass",
            }
        )
        print(
            f"completed {fixture_id}: crossings={len(crossings)} "
            f"status={projection_status}"
        )

    goemans_rows: list[dict[str, Any]] = [
        {
            "row_type": "fractional",
            "detour_1": str(DETOUR_FRACTIONS[0]),
            "detour_2": str(DETOUR_FRACTIONS[1]),
            "detour_3": str(DETOUR_FRACTIONS[2]),
            "edge_constraints_feasible": edge_relaxation_is_feasible(),
            "whole_route_feasible": "",
            "direct_cost": str(fractional_direct_cost()),
        }
    ]
    for row in integral_rows():
        goemans_rows.append(
            {
                "row_type": "integral",
                "detour_1": row["detour_1"],
                "detour_2": row["detour_2"],
                "detour_3": row["detour_3"],
                "edge_constraints_feasible": row["feasible"],
                "whole_route_feasible": row["feasible"],
                "direct_cost": row["direct_cost"],
            }
        )

    synchronize(device)
    elapsed_seconds = time.perf_counter() - start
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    audit = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "actual_device": str(device),
        "gpu_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "not used"
        ),
        "dtype": "float64 / complex128",
        "peak_cuda_memory_bytes": peak_memory,
        "fixture_count": len(fixture_rows),
        "all_expected_components_sampled": all(
            row["expected_core_components"] == row["sampled_core_components"]
            for row in fixture_rows
        ),
        "all_core_lines_closed": all(
            float(row["maximum_closure_error"]) < 1e-9
            for row in fixture_rows
        ),
        "all_core_tangencies_verified": all(
            float(row["minimum_tangent_alignment"]) > 0.99
            for row in fixture_rows
        ),
        "goemans_fractional_cost": str(fractional_direct_cost()),
        "goemans_minimum_integral_cost": minimum_integral_cost(),
        "goemans_missing_triangle_slack": str(
            missing_triangle_inequality_slack()
        ),
        "goemans_directed_path_counterexample_verified": False,
        "new_maxwell_solution_claimed": False,
        "new_knot_type_claimed": False,
        "new_four_color_proof_claimed": False,
        "elapsed_seconds": elapsed_seconds,
    }

    write_csv(output_dir / "maxwell_knot_fixture_rows.csv", fixture_rows)
    write_csv(output_dir / "maxwell_knot_residuals.csv", residual_rows)
    write_csv(output_dir / "maxwell_projection_pathologies.csv", pathology_rows)
    write_csv(output_dir / "goemans_conflict_fixture.csv", goemans_rows)
    (output_dir / "maxwell_knot_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )
    report = make_report(
        device=device,
        elapsed_seconds=elapsed_seconds,
        fixture_rows=fixture_rows,
        residual_rows=residual_rows,
        audit=audit,
    )
    (output_dir / "MAXWELL_KNOT_FIELD_REPORT.md").write_text(
        report,
        encoding="utf-8",
    )
    print(
        f"wrote {output_dir} in {elapsed_seconds:.3f}s "
        f"(peak_cuda_memory={peak_memory})"
    )


if __name__ == "__main__":
    main()
