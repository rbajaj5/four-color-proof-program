"""Generate exact Clifton-Salia plane-saturation gate artifacts."""

from __future__ import annotations

import csv
from fractions import Fraction
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plane_saturation_gate import (  # noqa: E402
    compensation_grid,
    construction_one_rows,
    coarse_grain_scope_rows,
    twin_fixture_rows,
)


OUTPUT_DIR = ROOT / "results" / "plane_saturation_gate"


def scalar_text(value: object) -> object:
    if isinstance(value, Fraction):
        return str(value)
    return value


def serializable(row: dict[str, object]) -> dict[str, object]:
    return {key: scalar_text(value) for key, value in row.items()}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    cooked = [serializable(row) for row in rows]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cooked[0]))
        writer.writeheader()
        writer.writerows(cooked)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    construction_rows = construction_one_rows()
    compensation_rows = compensation_grid(64)
    twin_rows = twin_fixture_rows()
    scope_rows = coarse_grain_scope_rows()

    write_csv(
        OUTPUT_DIR / "construction_one_exact_counts.csv",
        construction_rows,
    )
    write_csv(
        OUTPUT_DIR / "r2_r3_compensation_grid.csv",
        compensation_rows,
    )
    write_csv(
        OUTPUT_DIR / "low_degree_twin_fixture_rows.csv",
        twin_rows,
    )
    write_csv(
        OUTPUT_DIR / "coarse_grain_scope_matrix.csv",
        scope_rows,
    )

    all_construction_checks = all(
        row["same_vertex_count"]
        and row["claim_2_1_inequality_holds"]
        for row in construction_rows
    )
    all_compensation_checks = all(
        row["an_applicable_case_proves_1_16"]
        for row in compensation_rows
    )
    audit = {
        "source": {
            "title": "Saturated Partial Embeddings of Planar Graphs",
            "authors": ["Alexander Clifton", "Nika Salia"],
            "arxiv": "2403.02458",
            "version_reviewed": "v1",
            "submitted": "2024-03-04",
            "provided_text_sha256": (
                "891d3b7dd3684ed30463a187a2d1a1f63c1688b0351194c7d328dd0566b56898"
            ),
        },
        "construction_one_exact_count_gate": all_construction_checks,
        "r2_r3_compensation_grid_gate": all_compensation_checks,
        "compensation_pair_count": len(compensation_rows),
        "theorem_1_3_bound": "strictly_greater_than_1/16",
        "conjecture_4_2_bound": "strictly_greater_than_1/9",
        "conjecture_promoted_to_theorem": False,
        "four_color_improvement_claimed": False,
        "knot_invariant_claimed": False,
        "simulation_used": False,
        "gpu_used": False,
        "artifact_role": (
            "exact transcription gate and coarse-graining scope diagnostic"
        ),
    }
    (OUTPUT_DIR / "plane_saturation_gate_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )

    report = f"""# Plane-Saturation Coarse-Graining Gate

## Verdict

The Clifton-Salia paper supplies a useful **embedding-efficiency diagnostic**
for this repository, not an improvement to the Four Color Theorem.

The exact Construction 1 count gate
**{"passes" if all_construction_checks else "fails"}**. The two-case
`r2/r3` arithmetic gate behind the twin-free `1/16` lower bound
**{"passes" if all_compensation_checks else "fails"}** across all
**{len(compensation_rows)}** nonzero integer pairs with coordinates at most
`64`.

This finite enumeration checks transcription and boundary arithmetic. The
paper's structural graph argument remains the proof.

## Published Results Recorded

For a plane-saturated subgraph `H` of a planar graph `G`, the paper defines

```text
psr(G) = min e(H) / e(G).
```

It proves:

- every planar graph with no degree-1 or degree-2 twins has
  `psr(G) > 1/16`;
- twin-free examples approach `1/16`;
- for the stated parameter regime,
  `psr(G) > 1 / (9 + k1 + 6 k2)`.

The extension to all nonnegative `(k1,k2)` and the minimum-degree-three
bound `psr(G) > 1/9` remain Conjectures 4.1 and 4.2.

## Construction 1 Audit

For `m >= 7`, the published construction has

```text
v(G) = 7m + 8       e(G) = 16m + 8
v(H) = 7m + 8       e(H) = m + 34.
```

The checker verifies exactly that

```text
(m + 34)/(16m + 8) < 1/16 + 3/m
```

for the recorded fixtures. The distance above `1/16` decreases to zero.

## Why The Lower Bound Needs Two Cases

One skeleton estimate leaves an `r2` term with ratio `1/17`. Grouping it
with `r3` repairs the estimate precisely when

```text
(r2 + r3)/(17r2 + 12r3) >= 1/16
iff 4r3 >= r2.
```

The alternate estimate repairs its deficient `r3` term when

```text
(r2 + r3)/(15r2 + 18r3) >= 1/16
iff r2 >= 2r3.
```

Every nonzero nonnegative pair satisfies at least one of these conditions.
The code checks both boundary equalities exactly with `Fraction` arithmetic.

## Upgrade To The Coarse-Graining Program

A planar coarse-graining record should now include:

1. host and retained edge counts;
2. the skeleton after deleting isolated vertices;
3. isolated-vertex count and face placement;
4. maximum degree-1 and degree-2 open-neighborhood twin multiplicities;
5. whether saturation was actually certified;
6. whether the object is a simple graph or a multigraph.

This catches a failure mode that color counts miss: repeated low-degree
neighborhoods can make the retained embedding ratio arbitrarily small.

## Knot-Projection Boundary

A simple four-regular planar projection graph automatically has no
degree-1 or degree-2 twins, so the paper's `1/16` hypothesis applies.
Generic knot projections, however, can be planar multigraphs with loops or
parallel edges. Passing to the underlying simple graph can change the host
subgraph relation and the saturation property. Therefore the theorem is
not applied to those diagrams until a preservation lemma or a multigraph
extension is supplied.

Plane saturation and Four Color also answer different questions:
saturation measures maximality of a partial embedding, while coloring
measures compatibility of region or vertex labels.

## Scope

- No Four Color bound is improved.
- Conjectures 4.1 and 4.2 remain conjectures.
- No knot equivalence or knot invariant follows.
- No simulation or GPU calculation was needed.
- The paper's full structural proof was not re-proved.

## Source

Alexander Clifton and Nika Salia, *Saturated Partial Embeddings of Planar
Graphs*, arXiv:2403.02458v1 (2024):
https://arxiv.org/abs/2403.02458
"""
    (OUTPUT_DIR / "PLANE_SATURATION_GATE_REPORT.md").write_text(
        report,
        encoding="utf-8",
    )
    print(
        f"wrote plane-saturation gate to {OUTPUT_DIR}; "
        f"construction={all_construction_checks}; "
        f"compensation={all_compensation_checks}"
    )


if __name__ == "__main__":
    main()
