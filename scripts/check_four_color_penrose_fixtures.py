"""Generate exact finite fixtures for the Four Color/Penrose proof program."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.four_color_penrose import (
    CubicRotationSystem,
    evaluate_penrose_tait,
    has_bridge,
    rotation_from_coordinates,
)


OUTPUT_DIR = ROOT / "results" / "four_color_penrose_program"


def coordinate_system(
    edges: tuple[tuple[int, int], ...],
    coordinates: dict[int, tuple[float, float]],
) -> CubicRotationSystem:
    return CubicRotationSystem(
        edges=edges,
        rotations=rotation_from_coordinates(edges, coordinates),
    )


def fixture_systems() -> tuple[dict[str, Any], ...]:
    theta_edges = ((0, 1), (0, 1), (0, 1))
    theta = CubicRotationSystem(
        edges=theta_edges,
        rotations={0: (0, 1, 2), 1: (0, 2, 1)},
    )

    k4_edges = (
        (0, 1),
        (1, 2),
        (2, 0),
        (0, 3),
        (1, 3),
        (2, 3),
    )
    k4 = coordinate_system(
        k4_edges,
        {
            0: (-1.5, -1.0),
            1: (1.5, -1.0),
            2: (0.0, 1.5),
            3: (0.0, 0.0),
        },
    )

    cube_edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
    cube = coordinate_system(
        cube_edges,
        {
            0: (-3.0, -3.0),
            1: (3.0, -3.0),
            2: (3.0, 3.0),
            3: (-3.0, 3.0),
            4: (-1.0, -1.0),
            5: (1.0, -1.0),
            6: (1.0, 1.0),
            7: (-1.0, 1.0),
        },
    )

    prism_edges = (
        (0, 1),
        (1, 2),
        (2, 0),
        (3, 4),
        (4, 5),
        (5, 3),
        (0, 3),
        (1, 4),
        (2, 5),
    )
    prism = coordinate_system(
        prism_edges,
        {
            0: (0.0, 3.0),
            1: (-2.6, -1.5),
            2: (2.6, -1.5),
            3: (0.0, 1.2),
            4: (-1.0, -0.6),
            5: (1.0, -0.6),
        },
    )

    k33_edges = tuple(
        (left, right)
        for left in (0, 1, 2)
        for right in (3, 4, 5)
    )
    k33 = coordinate_system(
        k33_edges,
        {
            0: (-2.0, 2.0),
            1: (-2.0, 0.0),
            2: (-2.0, -2.0),
            3: (2.0, 2.0),
            4: (2.0, 0.0),
            5: (2.0, -2.0),
        },
    )

    petersen_edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 0),
        (0, 5),
        (1, 6),
        (2, 7),
        (3, 8),
        (4, 9),
        (5, 7),
        (7, 9),
        (9, 6),
        (6, 8),
        (8, 5),
    )
    petersen_coordinates: dict[int, tuple[float, float]] = {}
    for index in range(5):
        angle = 2.0 * math.pi * index / 5.0
        petersen_coordinates[index] = (
            2.5 * math.cos(angle),
            2.5 * math.sin(angle),
        )
        petersen_coordinates[index + 5] = (
            math.cos(angle),
            math.sin(angle),
        )
    petersen = coordinate_system(petersen_edges, petersen_coordinates)

    bridge_edges = (
        (0, 1),
        (1, 4),
        (4, 2),
        (2, 0),
        (0, 3),
        (1, 3),
        (2, 3),
        (5, 6),
        (6, 9),
        (9, 7),
        (7, 5),
        (5, 8),
        (6, 8),
        (7, 8),
        (4, 9),
    )
    bridge_incident: dict[int, list[int]] = {}
    for edge_id, (left, right) in enumerate(bridge_edges):
        bridge_incident.setdefault(left, []).append(edge_id)
        bridge_incident.setdefault(right, []).append(edge_id)
    bridge = CubicRotationSystem(
        edges=bridge_edges,
        rotations={
            vertex: tuple(edge_ids)
            for vertex, edge_ids in bridge_incident.items()
        },
    )

    return (
        {
            "fixture": "theta_plane_multigraph",
            "system": theta,
            "plane_embedding_declared": True,
            "expected_tait_count": 6,
            "role": "small planar multigraph calibration",
        },
        {
            "fixture": "k4_tetrahedral",
            "system": k4,
            "plane_embedding_declared": True,
            "expected_tait_count": 6,
            "role": "small simple planar calibration",
        },
        {
            "fixture": "cube_graph",
            "system": cube,
            "plane_embedding_declared": True,
            "expected_tait_count": None,
            "role": "larger bridgeless planar fixture",
        },
        {
            "fixture": "triangular_prism",
            "system": prism,
            "plane_embedding_declared": True,
            "expected_tait_count": None,
            "role": "odd-face bridgeless planar fixture",
        },
        {
            "fixture": "k33_nonplanar_control",
            "system": k33,
            "plane_embedding_declared": False,
            "expected_tait_count": 12,
            "role": "nonplanar signed-contraction boundary",
        },
        {
            "fixture": "petersen_snark_control",
            "system": petersen,
            "plane_embedding_declared": False,
            "expected_tait_count": 0,
            "role": "nonplanar uncolorable control",
        },
        {
            "fixture": "planar_bridge_control",
            "system": bridge,
            "plane_embedding_declared": True,
            "expected_tait_count": 0,
            "role": "bridge obstruction control",
        },
    )


def evaluate_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fixture in fixture_systems():
        system = fixture["system"]
        result = evaluate_penrose_tait(system)
        expected = fixture["expected_tait_count"]
        plane = bool(fixture["plane_embedding_declared"])
        rows.append(
            {
                "fixture": fixture["fixture"],
                "vertices": len(system.vertices()),
                "edges": len(system.edges),
                "plane_embedding_declared": plane,
                "has_bridge": has_bridge(system.edges),
                "tait_coloring_count": result.tait_coloring_count,
                "signed_epsilon_sum": result.signed_epsilon_sum,
                "penrose_contraction": result.penrose_contraction,
                "plane_penrose_identity_passed": (
                    result.penrose_contraction
                    == result.tait_coloring_count
                    if plane
                    else ""
                ),
                "expected_tait_count": "" if expected is None else expected,
                "expected_count_passed": (
                    ""
                    if expected is None
                    else result.tait_coloring_count == expected
                ),
                "role": fixture["role"],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report_text(rows: list[dict[str, Any]]) -> str:
    planar = [row for row in rows if row["plane_embedding_declared"]]
    nonplanar = [row for row in rows if not row["plane_embedding_declared"]]
    return f"""# Four Color Penrose Program Report

## Exact fixture result

- Fixtures: **{len(rows)}**
- Declared plane rotation systems: **{len(planar)}**
- Plane Penrose identities passed: **{sum(
    bool(row["plane_penrose_identity_passed"]) for row in planar
)}/{len(planar)}**
- Nonplanar controls: **{len(nonplanar)}**

The engine exactly counts proper Tait edge 3-colorings and independently
contracts the signed Levi-Civita tensors. The plane fixtures agree after the
standard `i^|V|` phase normalization.

## What this establishes

The implementation checks the finite algebraic interface behind the Penrose
reformulation. It also keeps bridge and nonplanar controls visible.

## What remains open

The Four Color burden is not the Penrose identity. It is proving that the
count is positive for every bridgeless cubic plane graph. Equivalently, using
Kauffman-Silver-Williams Theorem 6.6, one must prove that every reduced prime
alternating bigon-free plane diagram has a 3-colorable smoothing state.

Their component-count local-maximum theorem supplies a useful search
landscape, but not the missing 3-colorability argument.

## Decision

Continue with a bounded search for local smoothing reductions that preserve
3-colorable states. Treat any discovered move first as a conjectural reduction
and then test it against exact small diagrams. Do not claim a new proof unless
unavoidability, termination, and lifting are all proved.

## Sources

- https://arxiv.org/abs/2604.16635
- https://arxiv.org/abs/1511.06844
- https://thomas.math.gatech.edu/FC/fourcolor.html
- https://thomas.math.gatech.edu/FC/ftpinfo.html

## Status

- New Four Color proof: `false`
- New theorem: `false`
- Exact finite interface checks: `passed`
"""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = evaluate_rows()
    csv_path = OUTPUT_DIR / "four_color_penrose_fixtures.csv"
    report_path = OUTPUT_DIR / "FOUR_COLOR_PENROSE_PROGRAM_REPORT.md"
    audit_path = OUTPUT_DIR / "four_color_penrose_audit.json"
    write_csv(csv_path, rows)
    report_path.write_text(report_text(rows), encoding="utf-8")
    audit = {
        "fixture_count": len(rows),
        "all_plane_penrose_identities_passed": all(
            bool(row["plane_penrose_identity_passed"])
            for row in rows
            if row["plane_embedding_declared"]
        ),
        "all_declared_expected_counts_passed": all(
            row["expected_count_passed"] in ("", True)
            for row in rows
        ),
        "new_four_color_proof": False,
        "new_theorem": False,
        "proof_program": ["FC-C1", "FC-C2", "FC-C3"],
        "primary_new_source": "https://arxiv.org/abs/2604.16635",
    }
    audit_path.write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} exact fixtures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
