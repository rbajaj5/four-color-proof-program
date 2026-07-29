"""Run exact local edge-smoothing and coloring-lift experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import networkx as nx


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cubic_edge_smoothing import (
    InvalidSmoothing,
    SmoothingChild,
    enumerate_tait_colorings,
    liftable_parent_colorings,
    replacement_edges_are_adjacent,
    replacement_edges_are_bichromatic,
    smoothing_partition,
)
from src.four_color_penrose import has_bridge


OUTPUT_DIR = ROOT / "results" / "cubic_edge_smoothing"


def canonical_edges(graph: nx.Graph) -> tuple[tuple[int, int], ...]:
    edges = (
        (min(int(left), int(right)), max(int(left), int(right)))
        for left, right in graph.edges()
    )
    return tuple(sorted(edges))


def fixture_graphs() -> tuple[tuple[str, nx.Graph], ...]:
    return (
        ("k4_tetrahedral", nx.tetrahedral_graph()),
        ("triangular_prism", nx.circular_ladder_graph(3)),
        ("cube_graph", nx.cubical_graph()),
        ("k33_nonplanar", nx.complete_bipartite_graph(3, 3)),
        ("petersen_snark", nx.petersen_graph()),
        ("frucht_graph", nx.frucht_graph()),
        ("truncated_tetrahedron", nx.truncated_tetrahedron_graph()),
        ("heawood_graph", nx.heawood_graph()),
    )


def child_is_planar(child: SmoothingChild) -> bool:
    graph = nx.MultiGraph()
    graph.add_edges_from(child.edges)
    return bool(nx.check_planarity(graph)[0])


def evaluate_experiment() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for fixture, graph in fixture_graphs():
        if not nx.is_connected(graph):
            raise ValueError(f"{fixture} must be connected")
        if set(dict(graph.degree()).values()) != {3}:
            raise ValueError(f"{fixture} must be cubic")

        edges = canonical_edges(graph)
        original_colorings = enumerate_tait_colorings(edges)
        original_coloring_set = set(original_colorings)
        graph_rows: list[dict[str, Any]] = []
        edge_identity_results: list[bool] = []

        for central_edge_id, central_edge in enumerate(edges):
            branches, combined_lifted = smoothing_partition(
                edges,
                central_edge_id,
            )
            combined_lifted_set = set(combined_lifted)
            partition_passed = (
                len(combined_lifted) == len(combined_lifted_set)
                and combined_lifted_set == original_coloring_set
            )
            edge_identity_results.append(partition_passed)

            for branch in branches:
                base_row: dict[str, Any] = {
                    "fixture": fixture,
                    "vertices": graph.number_of_nodes(),
                    "edges": graph.number_of_edges(),
                    "parent_planar": bool(nx.check_planarity(graph)[0]),
                    "parent_tait_count": len(original_colorings),
                    "central_edge_id": central_edge_id,
                    "central_edge": repr(central_edge),
                    "pairing_index": branch.pairing_index,
                    "partition_identity_passed": partition_passed,
                }
                if isinstance(branch, InvalidSmoothing):
                    base_row.update(
                        {
                            "valid_child": False,
                            "invalid_reason": branch.reason,
                            "child_planar": "",
                            "child_has_bridge": "",
                            "replacement_edges_adjacent": "",
                            "child_tait_count": 0,
                            "replacement_monochromatic_count": 0,
                            "replacement_bichromatic_count": 0,
                            "liftable_parent_count": 0,
                            "lift_rate": "",
                            "colorable_but_nonliftable": False,
                        }
                    )
                else:
                    child_colorings = enumerate_tait_colorings(branch.edges)
                    lifted = liftable_parent_colorings(branch)
                    replacement_adjacent = replacement_edges_are_adjacent(
                        branch
                    )
                    bichromatic_count = sum(
                        replacement_edges_are_bichromatic(branch, coloring)
                        for coloring in child_colorings
                    )
                    if bichromatic_count != len(lifted):
                        raise AssertionError(
                            "bichromatic boundary count must equal lift count"
                        )
                    if (
                        replacement_adjacent
                        and bichromatic_count != len(child_colorings)
                    ):
                        raise AssertionError(
                            "adjacent replacement edges must be bichromatic"
                        )
                    base_row.update(
                        {
                            "valid_child": True,
                            "invalid_reason": "",
                            "child_planar": child_is_planar(branch),
                            "child_has_bridge": has_bridge(branch.edges),
                            "replacement_edges_adjacent": replacement_adjacent,
                            "child_tait_count": len(child_colorings),
                            "replacement_monochromatic_count": (
                                len(child_colorings) - bichromatic_count
                            ),
                            "replacement_bichromatic_count": bichromatic_count,
                            "liftable_parent_count": len(lifted),
                            "lift_rate": (
                                len(lifted) / len(child_colorings)
                                if child_colorings
                                else ""
                            ),
                            "colorable_but_nonliftable": (
                                bool(child_colorings) and not lifted
                            ),
                        }
                    )
                rows.append(base_row)
                graph_rows.append(base_row)

        valid_rows = [row for row in graph_rows if row["valid_child"]]
        colorable_rows = [
            row for row in valid_rows if row["child_tait_count"] > 0
        ]
        liftable_rows = [
            row for row in valid_rows
            if row["liftable_parent_count"] > 0
        ]
        lift_rates = [
            float(row["lift_rate"])
            for row in colorable_rows
            if row["lift_rate"] != ""
        ]
        summaries.append(
            {
                "fixture": fixture,
                "vertices": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "parent_planar": bool(nx.check_planarity(graph)[0]),
                "parent_tait_count": len(original_colorings),
                "edges_checked": len(edges),
                "partition_identity_passed_all_edges": all(
                    edge_identity_results
                ),
                "valid_children": len(valid_rows),
                "invalid_children": len(graph_rows) - len(valid_rows),
                "colorable_children": len(colorable_rows),
                "liftable_children": len(liftable_rows),
                "colorable_but_nonliftable_children": sum(
                    bool(row["colorable_but_nonliftable"])
                    for row in valid_rows
                ),
                "adjacent_replacement_children": sum(
                    row["replacement_edges_adjacent"] is True
                    for row in valid_rows
                ),
                "adjacent_replacement_full_lift_passed": all(
                    row["replacement_bichromatic_count"]
                    == row["child_tait_count"]
                    for row in valid_rows
                    if row["replacement_edges_adjacent"] is True
                ),
                "mean_lift_rate_among_colorable_children": (
                    statistics.fmean(lift_rates) if lift_rates else ""
                ),
            }
        )
    return rows, summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report_text(
    rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
) -> str:
    obstruction_rows = [
        row for row in rows if row["colorable_but_nonliftable"]
    ]
    planar_parent_obstructions = [
        row for row in obstruction_rows if row["parent_planar"]
    ]
    identity_passed = all(
        row["partition_identity_passed_all_edges"]
        for row in summaries
    )
    obstruction_examples = "\n".join(
        f"- `{row['fixture']}`, edge `{row['central_edge']}`, "
        f"pairing {row['pairing_index']}: "
        f"{row['child_tait_count']} child colorings, 0 lifts"
        for row in obstruction_rows[:12]
    )
    if not obstruction_examples:
        obstruction_examples = "- None in this bounded catalog."

    summary_table = "\n".join(
        "| {fixture} | {parent_planar} | {parent_tait_count} | "
        "{colorable_children} | {liftable_children} | "
        "{colorable_but_nonliftable_children} |".format(**row)
        for row in summaries
    )
    return f"""# Cubic Edge-Smoothing Lift Report

## Question

For an edge of a loopless cubic graph, delete its endpoints and reconnect the
four dangling halfedges in the two possible cross pairings. Does a Tait
3-edge-coloring of a smoothed child imply a coloring of the parent?

## Exact result

- Parent fixtures: **{len(summaries)}**
- Parent edges checked: **{sum(row["edges_checked"] for row in summaries)}**
- Smoothing branches recorded: **{len(rows)}**
- Exact coloring-partition identity passed on every edge: **{identity_passed}**
- Colorable children with no liftable coloring: **{len(obstruction_rows)}**
- Such children from planar parents: **{len(planar_parent_obstructions)}**

For every tested edge, the parent's labeled Tait colorings partition exactly
between the two smoothing pairings:

```text
Col_3(G) = Lift_0(G,e) disjoint-union Lift_1(G,e).
```

This identity follows directly from the two noncentral colors at the endpoints
of a colored edge: exactly one pairing joins equal colors. The program checks
the stronger labeled-set equality, not just equality of cardinalities.

## Fixture summary

| Fixture | Planar parent | Parent colorings | Colorable children | Liftable children | Colorable but nonliftable |
| --- | --- | ---: | ---: | ---: | ---: |
{summary_table}

## Obstruction to a naive reduction

{obstruction_examples}

A child being 3-edge-colorable is therefore not, by itself, a
positivity-preserving reduction certificate. A valid FC-C1 move needs either a
liftable child coloring or a structural condition forcing positive lift rate.

The exact boundary criterion is especially simple. A proper child coloring
lifts if and only if the two replacement edges are bichromatic. Consequently,

```text
|Col_3(G)| = Bichrom_0(G,e) + Bichrom_1(G,e),
```

while the monochromatic sectors contribute no parent colorings. This
two-terminal signature is the useful quantity for subsequent reductions.

There is also a proved local sufficient condition: if the two replacement
edges meet at a child vertex, properness forces distinct colors, and every
child coloring lifts. All
**{sum(row["adjacent_replacement_children"] for row in summaries)}** branches
with adjacent replacement edges passed this implication. What remains missing
is an
unavoidability argument that supplies a complexity-reducing, colorable branch
with a positive bichromatic sector in every bridgeless cubic plane graph.

## Scope

This is an exact finite experiment and an elementary local identity. It is not
a new proof of the Four Color Theorem and is not presented as a novel theorem.
The nonplanar controls diagnose the lifting logic; they are not evidence about
plane unavoidability.

## Decision

Retain the exact partition identity as infrastructure. Search next for
checkable local conditions that force a nonzero liftable sector, while testing
every proposed condition against snarks and nonplanar controls before any
planar proof claim.
"""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, summaries = evaluate_experiment()
    write_csv(OUTPUT_DIR / "edge_smoothing_rows.csv", rows)
    write_csv(OUTPUT_DIR / "edge_smoothing_graph_summary.csv", summaries)
    audit = {
        "fixtures": len(summaries),
        "parent_edges_checked": sum(
            row["edges_checked"] for row in summaries
        ),
        "smoothing_branches": len(rows),
        "all_partition_identities_passed": all(
            row["partition_identity_passed_all_edges"]
            for row in summaries
        ),
        "colorable_but_nonliftable_children": sum(
            row["colorable_but_nonliftable_children"]
            for row in summaries
        ),
        "planar_parent_colorable_but_nonliftable_children": sum(
            row["parent_planar"] and row["colorable_but_nonliftable"]
            for row in rows
        ),
        "adjacent_replacement_children": sum(
            row["replacement_edges_adjacent"] is True
            for row in rows
        ),
        "adjacent_replacement_full_lift_passed": all(
            row["replacement_bichromatic_count"] == row["child_tait_count"]
            for row in rows
            if row["replacement_edges_adjacent"] is True
        ),
        "new_four_color_proof": False,
        "new_theorem_claim": False,
    }
    (OUTPUT_DIR / "edge_smoothing_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "CUBIC_EDGE_SMOOTHING_REPORT.md").write_text(
        report_text(rows, summaries),
        encoding="utf-8",
    )
    print(
        f"checked {audit['parent_edges_checked']} parent edges; "
        f"wrote {len(rows)} smoothing rows to {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
