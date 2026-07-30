"""Exact Walsh-Fourier and game diagnostics for smoothing pictures."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.penrose_smoothing_landscape import (  # noqa: E402
    canonical_edges,
    chromatic_bucket,
    state_component_graph,
    state_nullity,
)
from src.picture_fourier_smoothing import (  # noqa: E402
    strict_local_maximum_indicator,
    strict_nash_indicator,
    walsh_hadamard,
)


OUTPUT_DIR = ROOT / "results" / "picture_fourier_smoothing"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def graph_nullities(graph: nx.Graph) -> tuple[tuple[tuple[int, int], ...], list[int]]:
    edges = canonical_edges(graph)
    return edges, [
        state_nullity(len(graph), edges, state)
        for state in range(1 << len(edges))
    ]


def spectrum_row(
    fixture: str,
    vertex_count: int,
    edge_count: int,
    observable: str,
    values: list[int],
) -> dict[str, object]:
    spectrum = walsh_hadamard(values)
    return {
        "fixture": fixture,
        "vertex_count": vertex_count,
        "edge_count": edge_count,
        "observable": observable,
        "state_count": len(values),
        "spectrum_minimum": min(spectrum),
        "spectrum_maximum": max(spectrum),
        "negative_coefficient_count": sum(
            value < 0 for value in spectrum
        ),
        "zero_coefficient_count": sum(value == 0 for value in spectrum),
        "translation_kernel_psd": min(spectrum) >= 0,
    }


def selected_fixture_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    game_rows: list[dict[str, object]] = []
    fixtures = (
        ("K4", nx.complete_graph(4)),
        ("graph_atlas_44", nx.graph_atlas(44)),
        ("graph_atlas_48", nx.graph_atlas(48)),
        ("graph_atlas_51", nx.graph_atlas(51)),
    )
    for name, raw_graph in fixtures:
        graph = nx.convert_node_labels_to_integers(raw_graph)
        edges, nullities = graph_nullities(graph)
        strict = strict_local_maximum_indicator(nullities, len(edges))
        observables = {
            "2^nullity": [2**value for value in nullities],
            "3^nullity": [3**value for value in nullities],
            "strict_maximum_indicator": strict,
        }
        if name in {"K4", "graph_atlas_51"}:
            colorable = [
                int(
                    chromatic_bucket(
                        state_component_graph(graph, edges, state)
                    )
                    in {"1", "2", "3"}
                )
                for state in range(1 << len(edges))
            ]
            observables["3colorable_indicator"] = colorable
            observables["colorable_strict_indicator"] = [
                left * right
                for left, right in zip(strict, colorable, strict=True)
            ]
        for observable, values in observables.items():
            rows.append(
                spectrum_row(
                    name,
                    len(graph),
                    len(edges),
                    observable,
                    values,
                )
            )
        nash = strict_nash_indicator(nullities, len(edges))
        game_rows.append(
            {
                "fixture": name,
                "players": len(edges),
                "states": len(nullities),
                "strict_local_maxima": sum(strict),
                "strict_nash_equilibria": sum(nash),
                "sets_identical": strict == nash,
            }
        )
    return rows, game_rows


def atlas_census() -> list[dict[str, object]]:
    rows = []
    for atlas_index, raw_graph in enumerate(nx.graph_atlas_g()):
        if not raw_graph or not nx.is_connected(raw_graph):
            continue
        if raw_graph.number_of_edges() > 10:
            continue
        if not nx.check_planarity(raw_graph)[0]:
            continue
        graph = nx.convert_node_labels_to_integers(raw_graph)
        edges, nullities = graph_nullities(graph)
        observables = {
            "2^nullity": [2**value for value in nullities],
            "3^nullity": [3**value for value in nullities],
            "strict_maximum_indicator": strict_local_maximum_indicator(
                nullities,
                len(edges),
            ),
        }
        for observable, values in observables.items():
            spectrum = walsh_hadamard(values)
            rows.append(
                {
                    "atlas_index": atlas_index,
                    "vertex_count": len(graph),
                    "edge_count": len(edges),
                    "observable": observable,
                    "spectrum_minimum": min(spectrum),
                    "negative_coefficient_count": sum(
                        value < 0 for value in spectrum
                    ),
                    "translation_kernel_psd": min(spectrum) >= 0,
                }
            )
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fixture_rows, game_rows = selected_fixture_rows()
    census_rows = atlas_census()
    write_csv(OUTPUT_DIR / "selected_smoothing_spectra.csv", fixture_rows)
    write_csv(OUTPUT_DIR / "smoothing_game_equilibria.csv", game_rows)
    write_csv(OUTPUT_DIR / "planar_atlas_fourier_census.csv", census_rows)

    census_graphs = len(
        {int(row["atlas_index"]) for row in census_rows}
    )
    census_summary = []
    for observable in (
        "2^nullity",
        "3^nullity",
        "strict_maximum_indicator",
    ):
        selected = [
            row for row in census_rows if row["observable"] == observable
        ]
        failures = [
            row for row in selected if not row["translation_kernel_psd"]
        ]
        census_summary.append(
            {
                "observable": observable,
                "graphs_tested": len(selected),
                "psd_count": len(selected) - len(failures),
                "indefinite_count": len(failures),
                "first_indefinite_atlas_index": (
                    min(int(row["atlas_index"]) for row in failures)
                    if failures
                    else ""
                ),
            }
        )
    write_csv(OUTPUT_DIR / "planar_atlas_fourier_summary.csv", census_summary)

    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    plotted = [
        row
        for row in fixture_rows
        if row["fixture"] in {"K4", "graph_atlas_48"}
        and row["observable"] in {"2^nullity", "3^nullity"}
    ]
    for axis, row in zip(axes.flat, plotted, strict=True):
        raw_graph = (
            nx.complete_graph(4)
            if row["fixture"] == "K4"
            else nx.graph_atlas(48)
        )
        graph = nx.convert_node_labels_to_integers(raw_graph)
        _, nullities = graph_nullities(graph)
        base = 2 if row["observable"] == "2^nullity" else 3
        spectrum = walsh_hadamard(
            [base**value for value in nullities]
        )
        axis.bar(
            np.arange(len(spectrum)),
            spectrum,
            color=[
                "#B2182B" if value < 0 else "#2166AC"
                for value in spectrum
            ],
            width=1.0,
        )
        axis.axhline(0, color="#222222", linewidth=0.8)
        axis.set_title(f"{row['fixture']}: {row['observable']}")
        axis.set_xlabel("Walsh character")
        axis.set_ylabel("exact coefficient")
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "selected_walsh_spectra.png",
        dpi=180,
    )
    plt.close(figure)

    negative_controls = [
        row
        for row in fixture_rows
        if (
            row["fixture"],
            row["observable"],
        )
        in {
            ("graph_atlas_48", "3^nullity"),
            ("graph_atlas_44", "strict_maximum_indicator"),
            ("K4", "3colorable_indicator"),
            ("graph_atlas_51", "colorable_strict_indicator"),
        }
    ]
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.4))
    for axis, row in zip(axes.flat, negative_controls, strict=True):
        raw_graph = {
            "K4": nx.complete_graph(4),
            "graph_atlas_44": nx.graph_atlas(44),
            "graph_atlas_48": nx.graph_atlas(48),
            "graph_atlas_51": nx.graph_atlas(51),
        }[str(row["fixture"])]
        graph = nx.convert_node_labels_to_integers(raw_graph)
        edges, nullities = graph_nullities(graph)
        observable = str(row["observable"])
        if observable == "3^nullity":
            values = [3**value for value in nullities]
        elif observable == "strict_maximum_indicator":
            values = strict_local_maximum_indicator(nullities, len(edges))
        else:
            colorable = [
                int(
                    chromatic_bucket(
                        state_component_graph(graph, edges, state)
                    )
                    in {"1", "2", "3"}
                )
                for state in range(1 << len(edges))
            ]
            if observable == "3colorable_indicator":
                values = colorable
            else:
                strict = strict_local_maximum_indicator(
                    nullities,
                    len(edges),
                )
                values = [
                    left * right
                    for left, right in zip(
                        strict,
                        colorable,
                        strict=True,
                    )
                ]
        spectrum = sorted(walsh_hadamard(values))
        shown = spectrum[: min(32, len(spectrum))]
        axis.bar(
            np.arange(len(shown)),
            shown,
            color=[
                "#B2182B" if value < 0 else "#2166AC"
                for value in shown
            ],
        )
        axis.axhline(0, color="#222222", linewidth=0.8)
        axis.set_title(f"{row['fixture']}: {observable}")
        axis.set_xlabel("smallest Walsh coefficients, sorted")
        axis.set_ylabel("exact coefficient")
        axis.text(
            0.98,
            0.08,
            f"min = {shown[0]}",
            transform=axis.transAxes,
            horizontalalignment="right",
            color="#B2182B",
            fontweight="bold",
        )
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "negative_walsh_controls.png",
        dpi=180,
    )
    plt.close(figure)

    summary_map = {
        str(row["observable"]): row for row in census_summary
    }
    audit = {
        "connected_planar_atlas_graphs_tested": census_graphs,
        "maximum_edges": 10,
        "two_to_nullity_indefinite_count": int(
            summary_map["2^nullity"]["indefinite_count"]
        ),
        "three_to_nullity_indefinite_count": int(
            summary_map["3^nullity"]["indefinite_count"]
        ),
        "strict_maximum_indicator_indefinite_count": int(
            summary_map["strict_maximum_indicator"]["indefinite_count"]
        ),
        "strict_nash_equals_strict_local_maximum_all_fixtures": all(
            bool(row["sets_identical"]) for row in game_rows
        ),
        "prisoners_dilemma_claimed": False,
        "four_color_proof_claimed": False,
        "literature_novelty_claimed": False,
    }
    (OUTPUT_DIR / "picture_fourier_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )

    summary_table = "\n".join(
        "| {observable} | {graphs_tested} | {psd_count} | "
        "{indefinite_count} | {first_indefinite_atlas_index} |".format(
            **row
        )
        for row in census_summary
    )
    fixture_table = "\n".join(
        "| {fixture} | {observable} | {spectrum_minimum} | "
        "{negative_coefficient_count} | {translation_kernel_psd} |".format(
            **row
        )
        for row in fixture_rows
    )
    report = f"""# Picture-Fourier Smoothing Report

## Diagram-derived operation

Following the Jaffe-Liu quarter-turn diagram, the Boolean smoothing cube is
treated as `Z_2^E`. Its exact Walsh transform interchanges XOR convolution and
pointwise multiplication. Translation-kernel positivity is then equivalent
to a nonnegative Walsh spectrum.

## Exact proposition

For every graph and every nonnegative integer `r`,
`2^(r nullity(L_s))` is positive definite on the smoothing cube. It counts
tuples of mod-2 Laplacian kernel vectors and is a sum of indicators of linear
subspaces. Each such indicator has nonnegative Walsh transform.

## Bounded atlas census

| observable | planar graphs | PSD | indefinite | first indefinite atlas index |
| --- | ---: | ---: | ---: | ---: |
{summary_table}

## Selected fixtures

| fixture | observable | spectrum minimum | negative coefficients | PSD |
| --- | --- | ---: | ---: | --- |
{fixture_table}

![Selected Walsh spectra](selected_walsh_spectra.png)

The signed controls make the small negative coefficients visible:

![Negative Walsh controls](negative_walsh_controls.png)

## Game interpretation

Giving every edge-player the common payoff `nullity(L_s)` makes strict Nash
equilibria exactly the strict local maxima of the smoothing landscape. This is
an identical-interest potential game, not a Prisoner's Dilemma. Some such
equilibria are not 3-colorable, so equilibrium under the surrogate objective
does not solve the global coloring problem.

## Boundary

The positive nullity kernel does not imply positivity of `3^nullity`, the
strict-maximum indicator, or the 3-colorable-state indicator. The census and
fixtures are exact finite diagnostics. No Four Color proof or literature
novelty claim is made.
"""
    (OUTPUT_DIR / "PICTURE_FOURIER_SMOOTHING_REPORT.md").write_text(
        report,
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
