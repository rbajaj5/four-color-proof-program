"""A claim-bounded dictionary of useful Four Color special cases."""

from __future__ import annotations

from typing import Any

from src.goemans_conflict_fixture import (
    edge_relaxation_is_feasible,
    fractional_direct_cost,
    minimum_integral_cost,
    missing_triangle_inequality_slack,
)


SPECIAL_CASES: tuple[dict[str, Any], ...] = (
    {
        "case_id": "bipartite_planar_graph",
        "structure": "planar graph with no odd cycle",
        "color_bound": 2,
        "exact_diagnosis": "bipartiteness",
        "certificate": "two vertex classes",
        "z2_curvature": False,
        "klein_flux": False,
        "weighted_lift_status": "not_tested",
        "implementation_status": "standard_graph_check",
        "notes": "Four Color is unnecessary.",
    },
    {
        "case_id": "triangle_free_planar_graph",
        "structure": "planar graph with no triangle",
        "color_bound": 3,
        "exact_diagnosis": "Groetzsch theorem gives an upper bound",
        "certificate": "constructive three-coloring when supplied",
        "z2_curvature": False,
        "klein_flux": False,
        "weighted_lift_status": "not_tested",
        "implementation_status": "theorem_only",
        "notes": "The chromatic number may still be one or two.",
    },
    {
        "case_id": "generic_knot_diagram_regions",
        "structure": "regions of a generic link projection",
        "color_bound": 2,
        "exact_diagnosis": "checkerboard face coloring",
        "certificate": "black-white region partition",
        "z2_curvature": False,
        "klein_flux": False,
        "weighted_lift_status": "not_applicable",
        "implementation_status": "conceptual_control",
        "notes": "Ordinary diagram regions do not expose a Four Color effect.",
    },
    {
        "case_id": "eulerian_sphere_triangulation",
        "structure": "sphere triangulation with every degree even",
        "color_bound": 3,
        "exact_diagnosis": "Heawood parity criterion",
        "certificate": "all degrees even plus explicit three-coloring",
        "z2_curvature": True,
        "klein_flux": True,
        "weighted_lift_status": "not_tested",
        "implementation_status": "exact_fixture",
        "notes": "The triangle lower bound makes the chromatic number exactly three.",
    },
    {
        "case_id": "non_eulerian_sphere_triangulation",
        "structure": "sphere triangulation with at least one odd degree",
        "color_bound": 4,
        "exact_diagnosis": "parity obstruction plus Four Color",
        "certificate": "odd vertex and explicit four-coloring",
        "z2_curvature": True,
        "klein_flux": True,
        "weighted_lift_status": "not_tested",
        "implementation_status": "exact_fixture_and_sweeps",
        "notes": "The chromatic number is exactly four.",
    },
    {
        "case_id": "dual_cubic_tait_instance",
        "structure": "dual cubic graph of a sphere triangulation",
        "color_bound": 4,
        "exact_diagnosis": "Tait/Klein-four reformulation",
        "certificate": "nonzero conserved Z2xZ2 edge flow",
        "z2_curvature": False,
        "klein_flux": True,
        "weighted_lift_status": "not_tested",
        "implementation_status": "exact_flux_certificate",
        "notes": "Discrete flux labels are not physical magnetic flux.",
    },
    {
        "case_id": "curvature_generated_triangulation",
        "structure": "mixed-curvature diagonal map with exterior compactification",
        "color_bound": 4,
        "exact_diagnosis": "local XOR defects and degree parity",
        "certificate": "3/4 classification plus explicit coloring",
        "z2_curvature": True,
        "klein_flux": True,
        "weighted_lift_status": "not_tested",
        "implementation_status": "exact_cuda_fractional_and_knot_sweeps",
        "notes": "Projection, grid, kernel, and scale remain declared choices.",
    },
    {
        "case_id": "weighted_planar_conflict_triangle",
        "structure": "three pairwise-incompatible weighted choices",
        "color_bound": 3,
        "exact_diagnosis": "planar coloring plus relaxation audit",
        "certificate": "three-coloring and exact 58/60 arithmetic",
        "z2_curvature": False,
        "klein_flux": False,
        "weighted_lift_status": "fails_edge_relaxation_lift",
        "implementation_status": "exact_goemans_style_control",
        "notes": "Colorability does not imply weighted integral route selection.",
    },
    {
        "case_id": "arbitrary_planar_graph",
        "structure": "finite loopless planar graph",
        "color_bound": 4,
        "exact_diagnosis": "Four Color theorem",
        "certificate": "proper four-coloring",
        "z2_curvature": False,
        "klein_flux": False,
        "weighted_lift_status": "not_tested",
        "implementation_status": "scope_boundary",
        "notes": "A coloring alone supplies no weighted optimization guarantee.",
    },
)


def validate_special_case_registry() -> None:
    """Validate required fields and unique identifiers."""

    required = {
        "case_id",
        "structure",
        "color_bound",
        "exact_diagnosis",
        "certificate",
        "z2_curvature",
        "klein_flux",
        "weighted_lift_status",
        "implementation_status",
        "notes",
    }
    identifiers = set()
    for row in SPECIAL_CASES:
        if set(row) != required:
            raise ValueError(f"registry fields differ for {row.get('case_id')}")
        if row["case_id"] in identifiers:
            raise ValueError("special-case identifiers must be unique")
        identifiers.add(row["case_id"])
        if int(row["color_bound"]) not in (2, 3, 4):
            raise ValueError("color bounds must be two, three, or four")


def goemans_style_weighted_diagnosis() -> dict[str, Any]:
    """Return the exact weighted conflict-triangle control."""

    fractional = fractional_direct_cost()
    integral = minimum_integral_cost()
    return {
        "case_id": "weighted_planar_conflict_triangle",
        "edge_relaxation_feasible": edge_relaxation_is_feasible(),
        "fractional_direct_cost": str(fractional),
        "minimum_integral_direct_cost": integral,
        "integrality_gap": str(integral - fractional),
        "missing_odd_cycle_slack": str(missing_triangle_inequality_slack()),
        "proper_color_bound": 3,
        "coloring_implies_integral_lift": False,
        "diagnosis": (
            "add the triangle odd-cycle inequality; do not infer weighted "
            "whole-route integrality from planar colorability"
        ),
    }
