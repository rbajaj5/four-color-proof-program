"""Run quasiperiodic convex-cell coarse graining on the knot fixtures."""

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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_curvature_rg_knot_collapse_gpu import (
    FIXTURES,
    projection_crossings,
    select_device,
    synchronize,
    write_csv,
)
from src.curvature_rg_flow import periodic_heat_coarse_grain
from src.maxwell_knot_fields import magnetic_core_curve
from src.quasiperiodic_polyhedral_rg import (
    PHI,
    curve_point_curvatures,
    defect_library,
    itinerary_metrics,
    polyhedral_force_susceptibility,
    projected_winding_numbers,
    quasiperiodic_tetrahedral_itinerary,
    space_form_coupling_force,
    tangent_holonomy,
)


OUTPUT_DIR = ROOT / "results" / "quasiperiodic_polyhedral_rg"
INFLATION_LEVELS = (-2, -1, 0, 1, 2)
BASE_TAU_VALUES = (0.0, 0.02, 0.1, 0.5, 2.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--tile-count", type=int, default=55)
    parser.add_argument("--base-cell-scale", type=float, default=0.18)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def load_projection_fall_times(path: Path) -> dict[str, float]:
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


def unique_first_indices(values: torch.Tensor) -> torch.Tensor:
    seen: set[int] = set()
    indices = []
    for index, value in enumerate(values.detach().cpu().tolist()):
        integer = int(value)
        if integer not in seen:
            seen.add(integer)
            indices.append(index)
    return torch.tensor(indices, device=values.device, dtype=torch.long)


def render_concept_schematic(output_dir: Path) -> None:
    figure = plt.figure(figsize=(15, 4.6))

    index_axis = figure.add_subplot(1, 3, 1)
    theta = np.linspace(0.0, 2.0 * np.pi, 300)
    radius = 0.9 + 0.15 * np.cos(3.0 * theta)
    index_axis.plot(
        radius * np.cos(theta),
        radius * np.sin(theta),
        color="#0057b8",
        linewidth=2,
    )
    index_axis.plot([-1.35, 1.35], [-0.45, 0.55], color="#333333")
    index_axis.scatter([0.0], [0.0], color="#d81b60", s=35, zorder=4)
    index_axis.text(-0.58, -0.15, "+1", fontsize=11)
    index_axis.text(0.52, 0.20, "-1", fontsize=11)
    index_axis.text(0.05, -0.08, r"$v$", fontsize=12)
    index_axis.set_title("Signed crossings and vertex index")
    index_axis.set_aspect("equal")
    index_axis.axis("off")

    development_axis = figure.add_subplot(1, 3, 2)
    centers = [(0.0, 0.0), (0.8, 0.15), (1.55, -0.03), (2.3, 0.18)]
    for index, (center_x, center_y) in enumerate(centers):
        polygon = np.array(
            (
                (-0.42, -0.28),
                (0.35, -0.34),
                (0.48, 0.22),
                (-0.15, 0.38),
            )
        )
        polygon[:, 0] += center_x
        polygon[:, 1] += center_y
        development_axis.plot(
            *np.vstack((polygon, polygon[:1])).T,
            color="#555555",
            linewidth=1.2,
        )
        development_axis.text(
            center_x - 0.08,
            center_y + 0.03,
            rf"$\Gamma_{index}$",
            fontsize=11,
        )
    development_axis.plot(
        [-0.35, 2.65],
        [-0.18, 0.35],
        color="#d81b60",
        linewidth=2.2,
    )
    development_axis.scatter(
        [-0.35, 2.65],
        [-0.18, 0.35],
        color="#111111",
        s=22,
    )
    development_axis.set_title("Face-by-face tangent development")
    development_axis.set_aspect("equal")
    development_axis.axis("off")

    hierarchy_axis = figure.add_subplot(1, 3, 3)
    colors = {"L": "#00876c", "S": "#e67e22"}
    words = ("LSLLSLSL", "LSLLS", "LSL")
    y_values = (0.8, 0.0, -0.8)
    labels = ("micro cells", "meso supercells", "macro itinerary")
    for word, y_value, label in zip(words, y_values, labels, strict=True):
        cursor = 0.0
        for letter in word:
            width = PHI if letter == "L" else 1.0
            hierarchy_axis.add_patch(
                plt.Rectangle(
                    (cursor, y_value - 0.22),
                    width,
                    0.44,
                    facecolor=colors[letter],
                    edgecolor="white",
                    linewidth=0.8,
                    alpha=0.9,
                )
            )
            cursor += width
        hierarchy_axis.text(cursor + 0.2, y_value, label, va="center")
    hierarchy_axis.annotate(
        "",
        xy=(3.8, -0.52),
        xytext=(5.8, 0.55),
        arrowprops={"arrowstyle": "->", "linewidth": 1.8},
    )
    hierarchy_axis.set_xlim(-0.2, 13.5)
    hierarchy_axis.set_ylim(-1.2, 1.2)
    hierarchy_axis.set_title("Quasiperiodic inflation RG")
    hierarchy_axis.axis("off")

    figure.suptitle(
        "Polyhedral RG state: index, tangent development, and inflation"
    )
    figure.tight_layout()
    figure.savefig(
        output_dir / "polyhedral_rg_concept_schematic.png",
        dpi=190,
    )
    plt.close(figure)


def tetrahedron_faces(vertices: np.ndarray) -> list[np.ndarray]:
    return [
        vertices[[0, 1, 2]],
        vertices[[0, 1, 3]],
        vertices[[0, 2, 3]],
        vertices[[1, 2, 3]],
    ]


def render_occupied_tetrahedra(
    curve: torch.Tensor,
    itinerary: dict[str, Any],
    output_dir: Path,
) -> None:
    identifiers = itinerary["tetrahedron_ids"]
    first_indices = unique_first_indices(identifiers)
    vertices = itinerary["sample_tetrahedron_vertices"][first_indices]
    type_codes = itinerary["tile_type_codes"][first_indices]
    vertices_cpu = vertices.detach().cpu().numpy()
    type_cpu = type_codes.detach().cpu().numpy()
    points = curve.detach().cpu().numpy()
    palette = (
        "#0057b8",
        "#00876c",
        "#d81b60",
        "#e67e22",
        "#7a5195",
        "#ef5675",
        "#2f4b7c",
        "#ffa600",
    )
    figure = plt.figure(figsize=(9, 7.5))
    axis = figure.add_subplot(111, projection="3d")
    for tetrahedron, tile_type in zip(
        vertices_cpu,
        type_cpu,
        strict=True,
    ):
        collection = Poly3DCollection(
            tetrahedron_faces(tetrahedron),
            facecolor=palette[int(tile_type)],
            edgecolor="#666666",
            linewidth=0.25,
            alpha=0.055,
        )
        axis.add_collection3d(collection)
    axis.plot(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        color="#111111",
        linewidth=2.3,
    )
    all_points = np.vstack((points, vertices_cpu.reshape(-1, 3)))
    center = all_points.mean(axis=0)
    radius = 0.55 * float(np.max(np.ptp(all_points, axis=0)))
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_zticks([])
    axis.set_title(
        "Trefoil itinerary through sheared Fibonacci tetrahedra\n"
        "(occupied cells only; inflation level 0)"
    )
    figure.tight_layout()
    figure.savefig(
        output_dir / "trefoil_occupied_quasiperiodic_tetrahedra.png",
        dpi=190,
    )
    plt.close(figure)


def render_complexity_plot(
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(1, 4, figsize=(19, 4.8))
    colors = {
        "hopf_core_unknot": "#00876c",
        "trefoil_2_3": "#0057b8",
        "cinquefoil_2_5": "#d81b60",
        "torus_knot_3_4": "#e67e22",
    }
    initial_rows = [row for row in rows if float(row["tau"]) == 0.0]
    for fixture_id, _, _, _ in FIXTURES:
        selected = [
            row for row in initial_rows if row["fixture_id"] == fixture_id
        ]
        selected.sort(key=lambda row: int(row["inflation_level"]))
        level = [int(row["inflation_level"]) for row in selected]
        axes[0].plot(
            level,
            [int(row["occupied_tetrahedra"]) for row in selected],
            marker="o",
            color=colors[fixture_id],
            label=fixture_id,
        )
        axes[1].plot(
            level,
            [float(row["occupancy_entropy"]) for row in selected],
            marker="o",
            color=colors[fixture_id],
        )
        axes[2].plot(
            level,
            [
                float(row["maximum_cell_curvature_fraction"])
                for row in selected
            ],
            marker="o",
            color=colors[fixture_id],
        )
        axes[3].plot(
            level,
            [
                float(row["positive_space_form_force_mean_per_mv2"])
                for row in selected
            ],
            marker="o",
            color=colors[fixture_id],
        )
    titles = (
        "occupied tetrahedra",
        "itinerary entropy",
        "max cell curvature share",
        "mean spherical response per $m v^2$",
    )
    for axis, title in zip(axes, titles, strict=True):
        axis.set_xlabel("Fibonacci inflation level")
        axis.set_title(title)
        axis.grid(True, alpha=0.25)
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)
    figure.suptitle("Quasiperiodic polyhedral coarse-graining at tau=0")
    figure.tight_layout()
    figure.savefig(
        output_dir / "quasiperiodic_rg_complexity.png",
        dpi=190,
    )
    plt.close(figure)


def make_report(
    summaries: list[dict[str, Any]],
    defect_rows: list[dict[str, Any]],
    force_rows: list[dict[str, Any]],
    audit: dict[str, Any],
) -> str:
    table = "\n".join(
        "| {fixture_id} | {tau_label} | {projection_crossings} | "
        "{fine_occupied_tetrahedra} | {coarse_occupied_tetrahedra} | "
        "{fine_to_coarse_contraction:.4f} | "
        "{fine_nonzero_winding_fraction:.4f} | "
        "{tangent_holonomy_angle:.4g} |".format(**summary)
        for summary in summaries
    )
    minimum_residual = min(
        float(row["bounded_relation_residual"]) for row in defect_rows
    )
    zero_residual_count = sum(
        float(row["bounded_relation_residual"]) < 1e-12
        for row in defect_rows
    )
    return f"""# Quasiperiodic Polyhedral Knot RG Report

## Replacement for the Jenga Model

This experiment separates four coupled channels:

1. **metric flow:** Fourier heat time `tau` smooths the curve;
2. **spatial inflation:** a Fibonacci substitution tiling is inflated through
   five declared levels; and
3. **connection compression:** total turning and tangent holonomy are retained
   even when the detailed cell itinerary is contracted; and
4. **force susceptibility:** each occupied cell carries a normalized
   positive- and negative-curvature response envelope.

The 3D patch is the Cartesian product of three Fibonacci interval tilings,
globally transformed by a golden-ratio affine shear and subdivided by a
face-compatible Freudenthal triangulation. Every cell is a convex tetrahedron,
the patch is face-to-face, and inflation preserves quasiperiodic order.

This is an exact product-Fibonacci prototype, **not** an
Ammann-Kramer-Neri icosahedral tiling.

## Galperin Figure Interface

Figures 6-8 in Galperin's *Convex Polyhedra Without Simple Closed Geodesics*
provide the discrete differential-geometric interface:

- signed intersections define differences of vertex indices;
- stereographic projection turns those indices into planar winding numbers;
- successive faces are developed into a plane so a surface geodesic becomes
  straight; and
- the polyhedral Gauss-Bonnet ledger is
  `delta_L + sum_v Delta_v ind_L(v) = 2 pi k`.

The original schematic in this folder adapts those operations to the RG
pipeline. Source: https://www.ux1.eiu.edu/~ggalperin/papers/GeodesRCD.pdf

## Space-Form Force Channel

Coulton and Galperin compute the exact coupling force needed to keep two
constant-speed paths separated by distance `d` in a smooth two-dimensional
space form:

```text
K > 0:  F =  2 m v^2 sqrt(K) tan(sqrt(K) d / 2)
K < 0:  F = -2 m v^2 sqrt(-K) tanh(sqrt(-K) d / 2).
```

For small `d`, both have the signed approximation
`F = m v^2 K d + O(d^3)`. The implementation verifies this limit and records
a normalized local-chart susceptibility. Each sampled tetrahedron is assigned
a chart radius equal to its diameter and a virtual separation equal to one
half that diameter. Inflation therefore renormalizes the response scale along
with occupancy and curvature load. The `{len(force_rows)}` exact calibration
states are written to `space_form_force_response.csv`.

This channel is a geometric response model. The ambient tiling remains
Euclidean and does not physically exert the reported forces.

## CUDA Audit

- Device: `{audit["actual_device"]}`
- GPU: `{audit["gpu_name"]}`
- CUDA used: `{audit["cuda_used"]}`
- Curves: `{audit["fixture_count"]}`
- Curve samples: `{audit["samples"]}`
- Metric times: `{audit["tau_state_count"]}`
- Inflation levels: `{audit["inflation_level_count"]}`
- State rows: `{audit["state_row_count"]}`
- Peak CUDA allocation: `{audit["peak_cuda_memory_bytes"]}` bytes
- Elapsed: `{audit["elapsed_seconds"]:.3f}` seconds

## Results

| Fixture | State | Crossings | Fine cells | Coarse cells | Coarse/fine | Fine nonzero-winding fraction | Tangent holonomy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

The occupied-cell count and itinerary entropy contract under inflation, but
the tangent holonomy is retained as separate connection data. Curvature load
is also aggregated by cell rather than discarded. This is the central
improvement over the Fourier tower: coarse geometric appearance is no longer
the entire state. The space-form response decreases as cell diameter grows,
while the spherical response is slightly stronger than the equal-magnitude
hyperbolic response because `tan(x) > tanh(x)` for positive `x`.

## Defect Spectrum

The eight Fibonacci box types and six Freudenthal simplices produce 48 local
tetrahedral prototypes. Every defect vector sums to `4 pi`, as required by
Descartes' formula. A bounded integer search with coefficients in `[-3,3]`
found minimum normalized residual `{minimum_residual:.4g}` and
`{zero_residual_count}` exact-within-tolerance prototype relations.

This finite search does not prove rational independence. In particular,
symmetry-induced equal defects can create exact small relations, so Galperin's
generic-tetrahedron obstruction cannot be assigned automatically to every
cell.

## Scale Semantics

- **Microscopic:** individual tetrahedra, face transitions, local curvature
  loads, and local defect spectra.
- **Mesoscopic:** recurrent cell words, transition entropy, projected winding
  proxies, and regions where curvature concentrates.
- **Macroscopic:** contracted dual itinerary, total turning, tangent holonomy,
  and the knot's independent projection diagnostics.

## Claim Boundary

The knot passes through cell interiors in this first prototype. Therefore the
reported winding values around cell centers are ambient projected winding
proxies, not Galperin vertex indices of a broken line constrained to one
polyhedron surface. Local tetrahedral defects are likewise not summed as if
interior tiling vertices carried ambient curvature; face-to-face Euclidean
cells cancel that curvature globally.

The force susceptibility is not inferred from those discrete defects. The
Coulton-Galperin theorem assumes smooth constant sectional curvature and
equidistant particle paths; neither is supplied by an ambient Euclidean
tetrahedral itinerary. The recorded force is a declared chart envelope only.

No knot classification, geodesic-existence theorem, or new quasiperiodic
tiling theorem is claimed. The next rigorous step is to route each
entry-to-exit segment along the corresponding convex cell boundary, unfold
the traversed faces, and then evaluate the actual defect-index identity.
"""


def main() -> None:
    args = parse_args()
    if args.samples < 128 or args.tile_count < 24:
        raise ValueError("samples >= 128 and tile-count >= 24 are required")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.allow_cpu)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    fall_times = load_projection_fall_times(
        ROOT
        / "results"
        / "curvature_rg_knot_collapse"
        / "curvature_rg_flow_rows.csv"
    )
    library = defect_library(coefficient_bound=3)
    defect_rows = []
    for code, record in library.items():
        defects = np.asarray(record["defects"])
        defect_rows.append(
            {
                "type_simplex_code": code,
                "tile_type": record["tile_type"],
                "simplex_code": record["simplex_code"],
                "defect_0": defects[0],
                "defect_1": defects[1],
                "defect_2": defects[2],
                "defect_3": defects[3],
                "defect_sum": record["defect_sum"],
                "defect_sum_error_from_4pi": abs(
                    float(record["defect_sum"]) - 4.0 * math.pi
                ),
                "bounded_relation_residual": record[
                    "bounded_relation_residual"
                ],
            }
        )

    force_rows = []
    for curvature in (-4.0, -1.0, -0.25, 0.0, 0.25, 1.0, 4.0):
        for separation in (0.05, 0.25, 0.5):
            force = float(
                space_form_coupling_force(
                    curvature,
                    separation,
                ).item()
            )
            linear = curvature * separation
            force_rows.append(
                {
                    "sectional_curvature": curvature,
                    "separation": separation,
                    "mass": 1.0,
                    "speed": 1.0,
                    "exact_coupling_force": force,
                    "linear_small_distance_approximation": linear,
                    "absolute_approximation_error": abs(force - linear),
                }
            )

    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    visual_curve: torch.Tensor | None = None
    visual_itinerary: dict[str, Any] | None = None
    start = time.perf_counter()
    print(
        f"device={device} inflation_levels={len(INFLATION_LEVELS)} "
        f"tile_count={args.tile_count}"
    )

    for fixture_id, p, q, _ in FIXTURES:
        tau_values = set(BASE_TAU_VALUES)
        if fixture_id in fall_times:
            tau_values.add(fall_times[fixture_id])
        sorted_tau = sorted(tau_values)
        initial = magnetic_core_curve(
            p,
            q,
            args.samples,
            device=device,
            dtype=torch.float64,
        )
        fixture_tau_rows: dict[float, list[dict[str, Any]]] = {}
        for tau in sorted_tau:
            curve = periodic_heat_coarse_grain(
                initial,
                tau,
                normalize=True,
            )
            curvatures = curve_point_curvatures(curve)
            holonomy = tangent_holonomy(curve)
            points_cpu = curve.detach().cpu().numpy()
            crossing_count = len(projection_crossings(points_cpu))
            tau_rows = []
            for level in INFLATION_LEVELS:
                itinerary = quasiperiodic_tetrahedral_itinerary(
                    curve,
                    inflation_level=level,
                    base_scale=args.base_cell_scale,
                    tile_count=args.tile_count,
                )
                metrics = itinerary_metrics(itinerary, curvatures)
                force_metrics = polyhedral_force_susceptibility(itinerary)
                first_indices = unique_first_indices(
                    itinerary["tetrahedron_ids"]
                )
                centers = itinerary["cell_centers"][first_indices]
                winding = projected_winding_numbers(curve, centers)
                unique_type_codes = torch.unique(
                    itinerary["type_simplex_codes"]
                ).detach().cpu().tolist()
                relation_residuals = [
                    float(library[int(code)]["bounded_relation_residual"])
                    for code in unique_type_codes
                ]
                row = {
                    "fixture_id": fixture_id,
                    "p": p,
                    "q": q,
                    "tau": tau,
                    "is_projection_fall_state": (
                        fixture_id in fall_times
                        and abs(tau - fall_times[fixture_id]) < 1e-12
                    ),
                    "projection_crossings": crossing_count,
                    "inflation_level": level,
                    "scale_regime": (
                        "microscopic"
                        if level < 0
                        else "mesoscopic" if level == 0 else "macroscopic"
                    ),
                    "cell_scale": itinerary["cell_scale"],
                    **metrics,
                    **force_metrics,
                    "visited_defect_class_count": len(unique_type_codes),
                    "minimum_visited_bounded_relation_residual": min(
                        relation_residuals
                    ),
                    "median_visited_bounded_relation_residual": float(
                        np.median(relation_residuals)
                    ),
                    "nonzero_projected_winding_count": int(
                        torch.sum(winding != 0).item()
                    ),
                    "nonzero_projected_winding_fraction": float(
                        torch.mean((winding != 0).double()).item()
                    ),
                    "maximum_absolute_projected_winding": int(
                        torch.max(torch.abs(winding)).item()
                    ),
                    **holonomy,
                }
                rows.append(row)
                tau_rows.append(row)
                if (
                    fixture_id == "trefoil_2_3"
                    and tau == 0.0
                    and level == 0
                ):
                    visual_curve = curve
                    visual_itinerary = itinerary
            fixture_tau_rows[tau] = tau_rows
        selected_tau = [0.0]
        if fixture_id in fall_times:
            selected_tau.append(fall_times[fixture_id])
        for tau in selected_tau:
            state_rows = fixture_tau_rows[tau]
            by_level = {
                int(row["inflation_level"]): row for row in state_rows
            }
            fine = by_level[min(INFLATION_LEVELS)]
            coarse = by_level[max(INFLATION_LEVELS)]
            summaries.append(
                {
                    "fixture_id": fixture_id,
                    "tau": tau,
                    "tau_label": (
                        "initial"
                        if tau == 0.0
                        else f"projection_fall_tau={tau:.4g}"
                    ),
                    "projection_crossings": fine["projection_crossings"],
                    "fine_occupied_tetrahedra": fine[
                        "occupied_tetrahedra"
                    ],
                    "coarse_occupied_tetrahedra": coarse[
                        "occupied_tetrahedra"
                    ],
                    "fine_to_coarse_contraction": (
                        float(coarse["occupied_tetrahedra"])
                        / float(fine["occupied_tetrahedra"])
                    ),
                    "fine_nonzero_winding_fraction": fine[
                        "nonzero_projected_winding_fraction"
                    ],
                    "tangent_holonomy_angle": fine[
                        "tangent_holonomy_angle"
                    ],
                    "total_polygonal_turning": fine[
                        "total_polygonal_turning"
                    ],
                }
            )
        print(f"completed {fixture_id}: tau_states={len(sorted_tau)}")

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
        "tile_count_per_axis": args.tile_count,
        "base_cell_scale": args.base_cell_scale,
        "tau_state_count": len(
            {(row["fixture_id"], row["tau"]) for row in rows}
        ),
        "inflation_level_count": len(INFLATION_LEVELS),
        "state_row_count": len(rows),
        "defect_prototype_count": len(defect_rows),
        "force_calibration_row_count": len(force_rows),
        "force_model": (
            "Coulton-Galperin smooth constant-curvature response envelope"
        ),
        "force_model_is_physical_tiling_force": False,
        "peak_cuda_memory_bytes": peak_memory,
        "elapsed_seconds": elapsed,
    }
    write_csv(output_dir / "quasiperiodic_polyhedral_rg_rows.csv", rows)
    write_csv(
        output_dir / "quasiperiodic_polyhedral_rg_summary.csv",
        summaries,
    )
    write_csv(
        output_dir / "quasiperiodic_tetrahedron_defects.csv",
        defect_rows,
    )
    write_csv(
        output_dir / "space_form_force_response.csv",
        force_rows,
    )
    (output_dir / "quasiperiodic_polyhedral_rg_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )
    render_concept_schematic(output_dir)
    render_complexity_plot(rows, output_dir)
    if visual_curve is None or visual_itinerary is None:
        raise AssertionError("trefoil visualization state was not captured")
    render_occupied_tetrahedra(
        visual_curve,
        visual_itinerary,
        output_dir,
    )
    (output_dir / "QUASIPERIODIC_POLYHEDRAL_RG_REPORT.md").write_text(
        make_report(summaries, defect_rows, force_rows, audit),
        encoding="utf-8",
    )
    print(
        f"wrote {len(rows)} states in {elapsed:.3f}s "
        f"(peak_cuda_memory={peak_memory})"
    )


if __name__ == "__main__":
    main()
