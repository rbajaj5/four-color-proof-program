"""Build the Four Color special-case and weighted-lift diagnostic registry."""

from __future__ import annotations

import csv
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.four_color_special_cases import (
    SPECIAL_CASES,
    ZAMIR_REFERENCE,
    goemans_style_weighted_diagnosis,
    planar_chromatic_decision_hierarchy,
    validate_special_case_registry,
    zamir_solver_diagnoses,
)


OUTPUT_DIR = ROOT / "results" / "four_color_special_case_dictionary"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_matrix(rows: list[dict[str, Any]], output_dir: Path) -> None:
    labels = [str(row["case_id"]) for row in rows]
    matrix = np.array(
        [
            (
                bool(row["z2_curvature"]),
                bool(row["klein_flux"]),
                row["weighted_lift_status"] == "fails_edge_relaxation_lift",
                "exact" in str(row["implementation_status"]),
            )
            for row in rows
        ],
        dtype=float,
    )
    figure, axis = plt.subplots(figsize=(8.4, 6.2))
    axis.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    axis.set_xticks(
        range(4),
        ("Z2 curvature", "Klein flux", "weighted-lift warning", "exact code"),
        rotation=20,
        ha="right",
    )
    axis.set_yticks(range(len(labels)), labels)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                "yes" if matrix[row, column] else "-",
                ha="center",
                va="center",
                color="white" if matrix[row, column] else "#334155",
            )
    axis.set_title("Four Color special-case diagnostic dictionary")
    figure.tight_layout()
    figure.savefig(
        output_dir / "four_color_special_case_matrix.png",
        dpi=190,
    )
    plt.close(figure)


def render_solver_matrix(
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    labels = [str(row["case_id"]) for row in rows]
    matrix = np.array(
        [
            (
                bool(row["structural_bypass_generic_search"]),
                bool(row["zamir_fixed_palette_applicable"]),
                row["target_k"] <= 2,
                bool(row["weighted_warning"]),
            )
            for row in rows
        ],
        dtype=float,
    )
    figure, axis = plt.subplots(figsize=(8.8, 6.2))
    axis.imshow(matrix, cmap="PuBuGn", vmin=0.0, vmax=1.0, aspect="auto")
    axis.set_xticks(
        range(4),
        (
            "structural bypass",
            "fixed-palette fallback",
            "polynomial base case",
            "weighted scope warning",
        ),
        rotation=20,
        ha="right",
    )
    axis.set_yticks(range(len(labels)), labels)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                "yes" if matrix[row, column] else "-",
                ha="center",
                va="center",
                color="white" if matrix[row, column] else "#334155",
            )
    axis.set_title("Structural certificates before generic fixed-k search")
    figure.tight_layout()
    figure.savefig(output_dir / "zamir_solver_hierarchy.png", dpi=190)
    plt.close(figure)


def make_report(
    rows: list[dict[str, Any]],
    weighted: dict[str, Any],
    solver_rows: list[dict[str, Any]],
    hierarchy: list[dict[str, Any]],
) -> str:
    table = "\n".join(
        "| {case_id} | {color_bound} | {exact_diagnosis} | "
        "{weighted_lift_status} |".format(**row)
        for row in rows
    )
    solver_table = "\n".join(
        "| {case_id} | {target_k} | {problem_formulation} | "
        "{primary_solver_route} | {zamir_role} | "
        "{weighted_objective_in_scope} |".format(**row)
        for row in solver_rows
    )
    hierarchy_table = "\n".join(
        "| {stage} | {condition} | {outcome} | {method} |".format(**row)
        for row in hierarchy
    )
    return f"""# Four Color Special-Case Dictionary

## Purpose

This registry separates mechanisms that are often collapsed into the phrase
"Four Color." It asks which smaller theorem or exact certificate actually
does the work, whether a `Z2` curvature or Klein-four flow interpretation is
available, and whether a weighted optimization claim needs an additional
integrality audit.

| Case | Color bound | Diagnosis | Weighted-lift status |
| --- | ---: | --- | --- |
{table}

## Goemans-Style Weighted Control

The planar conflict triangle is 3-colorable, and its fractional point
satisfies all pairwise edge inequalities. Nevertheless:

- fractional direct cost: `{weighted["fractional_direct_cost"]}`;
- minimum integral direct cost: `{weighted["minimum_integral_direct_cost"]}`;
- exact gap: `{weighted["integrality_gap"]}`;
- omitted triangle-inequality slack: `{weighted["missing_odd_cycle_slack"]}`.

Thus a planar coloring can schedule conflict-free layers while failing to
lift a fractional weighted choice to a whole-route integral choice. The
diagnosis is: `{weighted["diagnosis"]}`.

## Fixed-Palette Algorithmic Layer

Zamir's 2026 theorem proves that, for every fixed palette size `K`,
fixed-palette list coloring has a randomized
`O*((2-epsilon_K)^n)` algorithm with exponentially small one-sided error.
An explicit coloring can be recovered from a decision algorithm with only
polynomial overhead. The result is useful here as a generic fallback, not as
the primary solver for structured planar cases.

Source: [{ZAMIR_REFERENCE["title"]}]({ZAMIR_REFERENCE["url"]}), Or Zamir,
arXiv:{ZAMIR_REFERENCE["arxiv"]}.

| Case | Target k | Formulation | Preferred route | Zamir role | Weighted objective in scope |
| --- | ---: | --- | --- | --- | --- |
{solver_table}

The implementation does **not** reproduce Zamir's research algorithm. It
records theorem applicability and keeps the existing exact structural
solvers in front. For `k=3`, specialized algorithms also have much stronger
explicit constants than the paper's general bootstrap. The paper's own
asymptotic quantitative estimate gives a reciprocal saving with tower height
`Theta(K)`, so this result should not be advertised as a practical solver
for these small fixtures.

## Planar Chromatic-Number Decision Hierarchy

| Stage | Certified condition | Exact outcome | Route |
| ---: | --- | --- | --- |
{hierarchy_table}

This hierarchy makes the headline distinction operational: one should not
compute the unrestricted chromatic number when a fixed `3`-color decision
plus the Four Color upper bound resolves the planar case.

## How To Use The Dictionary

1. Use bipartite/checkerboard structure before invoking Four Color.
2. Use the Heawood parity test for sphere triangulations.
3. Use `Z2 x Z2` differences when a conserved Tait/Klein-flow certificate is
   useful.
4. Use the weighted conflict audit whenever colors are interpreted as
   selectable routes, resources, or physical channels.
5. Keep curvature-generated maps labeled by projection, scale, kernel, and
   compactification.
6. Use fixed-palette search only after structural certificates fail.

## Claim Boundary

This is an organizational and diagnostic result. It neither strengthens the
Four Color Theorem nor proves a Goemans conjecture. The exact 58/60 row is an
abstract conflict-core control, not a certified directed-path realization.
Zamir's theorem concerns colorability, not weighted routing, and its full
algorithm has not been implemented in this repository.
"""


def main() -> None:
    validate_special_case_registry()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [dict(row) for row in SPECIAL_CASES]
    weighted = goemans_style_weighted_diagnosis()
    solver_rows = [dict(row) for row in zamir_solver_diagnoses()]
    hierarchy = [dict(row) for row in planar_chromatic_decision_hierarchy()]
    write_csv(OUTPUT_DIR / "four_color_special_cases.csv", rows)
    write_csv(OUTPUT_DIR / "goemans_weighted_diagnosis.csv", [weighted])
    write_csv(OUTPUT_DIR / "zamir_solver_diagnosis.csv", solver_rows)
    write_csv(OUTPUT_DIR / "planar_chromatic_decision_hierarchy.csv", hierarchy)
    render_matrix(rows, OUTPUT_DIR)
    render_solver_matrix(solver_rows, OUTPUT_DIR)
    (OUTPUT_DIR / "FOUR_COLOR_SPECIAL_CASE_DICTIONARY.md").write_text(
        make_report(rows, weighted, solver_rows, hierarchy),
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} special cases to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
