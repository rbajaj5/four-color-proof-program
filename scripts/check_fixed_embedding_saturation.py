"""Generate exact fixed-embedding saturation certificates."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fixed_embedding_saturation import (  # noqa: E402
    fixed_embedding_fixture_rows,
)


OUTPUT_DIR = ROOT / "results" / "fixed_embedding_saturation"


def csv_value(value: object) -> object:
    if isinstance(value, tuple):
        return ";".join(
            "-".join(map(str, item)) if isinstance(item, tuple) else str(item)
            for item in value
        )
    return value


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    cooked = [
        {key: csv_value(value) for key, value in row.items()}
        for row in rows
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cooked[0]))
        writer.writeheader()
        writer.writerows(cooked)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = fixed_embedding_fixture_rows()
    write_csv(
        OUTPUT_DIR / "fixed_embedding_saturation_fixtures.csv",
        rows,
    )

    passed = all(row["matches_expected"] for row in rows)
    audit = {
        "fixture_count": len(rows),
        "all_fixtures_match_expected": passed,
        "checker_scope": (
            "connected spanning simple subgraph, fixed labels, fixed "
            "sphere rotation system, existing-vertex edge additions only"
        ),
        "clifton_salia_unlabeled_saturation_implemented": False,
        "new_vertex_additions_implemented": False,
        "disconnected_component_nesting_implemented": False,
        "multigraphs_implemented": False,
        "four_color_result_inferred": False,
        "knot_result_inferred": False,
        "simulation_used": False,
        "gpu_used": False,
    }
    (OUTPUT_DIR / "fixed_embedding_saturation_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )

    report = f"""# Fixed-Embedding Saturation Checker

## Result

All **{len(rows)}** exact fixtures
**{"pass" if passed else "do not pass"}**.

For a connected spanning simple subgraph `H` with a declared sphere rotation
system, every face boundary is enumerated exactly as a dart orbit. A missing
host edge can be inserted without crossing precisely when its endpoints are
cofacial. The checker therefore certifies whether `H` is edge-saturated for
that fixed labeled embedding.

## Separator Fixture

The nontrivial fixture places one leaf inside a separating triangle and a
second leaf outside it. The abstract host adds an edge between the leaves.
That edge is not cofacial in the declared embedding, so it is blocked. If the
host instead adds an edge from the interior triangle vertex to the exterior
leaf, the checker finds the endpoints on a common face and rejects
saturation.

This makes the embedding dependence concrete: abstract planarity of the host
does not mean every host edge can be inserted into an arbitrary partial
embedding.

## Strict Scope Boundary

This checker does **not** implement the full Clifton-Salia definition.
Their vertices are unlabeled under the subgraph embedding, and an added edge
may introduce new vertices. Their proof also handles disconnected skeletons
and the placement of isolated vertices among faces.

Accordingly, every output keeps
`clifton_salia_plane_saturation_certified = False`. The current certificate
is useful as a necessary implementation layer and debugging oracle, not as a
replacement for the paper's saturation test.

## Next Exact Step

The next extension should enumerate label-preserving graph isomorphisms for
small hosts and explicitly encode component nesting and isolated-vertex face
assignments. Only then can finite fixtures approach the paper's unlabeled
notion. Multigraph knot projections remain a separate extension.

## Scope

- No Four Color result is inferred.
- No knot invariant or knot-equivalence claim is made.
- No simulation or GPU computation was used.
"""
    (OUTPUT_DIR / "FIXED_EMBEDDING_SATURATION_REPORT.md").write_text(
        report,
        encoding="utf-8",
    )
    print(
        f"wrote fixed-embedding saturation artifacts to {OUTPUT_DIR}; "
        f"passed={passed}"
    )


if __name__ == "__main__":
    main()
