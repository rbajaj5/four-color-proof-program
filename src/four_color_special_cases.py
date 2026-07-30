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

ZAMIR_REFERENCE = {
    "title": "k-Coloring is Faster than Computing the Chromatic Number",
    "author": "Or Zamir",
    "arxiv": "2607.25973",
    "url": "https://arxiv.org/abs/2607.25973",
    "theorem": (
        "For every fixed palette size K, fixed-palette list coloring has a "
        "randomized O*((2-epsilon_K)^n) algorithm with exponentially small "
        "one-sided error."
    ),
    "scope": (
        "The result decides fixed-palette colorability. It does not compute a "
        "weighted integral lift and the repository does not reimplement the "
        "paper's exponential-time algorithm."
    ),
}

_PRIMARY_SOLVER_ROUTES = {
    "bipartite_planar_graph": "bipartite BFS/DFS",
    "triangle_free_planar_graph": (
        "bipartite test; otherwise Groetzsch certifies exact chi=3"
    ),
    "generic_knot_diagram_regions": "checkerboard face coloring",
    "eulerian_sphere_triangulation": "degree parity plus explicit 3-coloring",
    "non_eulerian_sphere_triangulation": (
        "odd-degree obstruction plus explicit 4-coloring"
    ),
    "dual_cubic_tait_instance": "nonzero conserved Klein-four dual flow",
    "curvature_generated_triangulation": (
        "local XOR parity plus explicit 3/4-coloring"
    ),
    "weighted_planar_conflict_triangle": (
        "3-color conflict graph, then run a separate weighted-integrality audit"
    ),
    "arbitrary_planar_graph": (
        "bipartite test, then 3-color decision, then Four Color fallback"
    ),
}

_ZAMIR_TARGETS = {
    "bipartite_planar_graph": (2, "vertex coloring"),
    "triangle_free_planar_graph": (3, "vertex coloring"),
    "generic_knot_diagram_regions": (2, "vertex coloring of the region-dual graph"),
    "eulerian_sphere_triangulation": (3, "vertex coloring"),
    "non_eulerian_sphere_triangulation": (3, "vertex coloring decision returning NO"),
    "dual_cubic_tait_instance": (3, "vertex coloring of the line graph"),
    "curvature_generated_triangulation": (3, "vertex coloring decision"),
    "weighted_planar_conflict_triangle": (3, "vertex coloring of the conflict graph"),
    "arbitrary_planar_graph": (3, "vertex coloring decision"),
}


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


def zamir_solver_diagnoses() -> tuple[dict[str, Any], ...]:
    """Map each registered case to structural and generic solver routes.

    Zamir's theorem is recorded as a theoretical fixed-palette fallback. The
    full randomized algorithm is not reproduced here.
    """

    validate_special_case_registry()
    rows = []
    for case in SPECIAL_CASES:
        case_id = str(case["case_id"])
        target_k, problem_formulation = _ZAMIR_TARGETS[case_id]
        weighted_warning = (
            case["weighted_lift_status"] == "fails_edge_relaxation_lift"
        )
        structural_bypass = case_id != "arbitrary_planar_graph"
        if target_k <= 2:
            generic_bound = "polynomial via 2-list-coloring / 2-SAT"
            zamir_role = "polynomial base case"
        else:
            generic_bound = f"O*((2-epsilon_{target_k})^n), randomized"
            zamir_role = "theoretical fixed-palette fallback"
        rows.append(
            {
                "case_id": case_id,
                "target_k": target_k,
                "problem_formulation": problem_formulation,
                "primary_solver_route": _PRIMARY_SOLVER_ROUTES[case_id],
                "structural_bypass_generic_search": structural_bypass,
                "zamir_fixed_palette_applicable": True,
                "zamir_role": zamir_role,
                "generic_time_bound": generic_bound,
                "certificate_recovery": "decision-to-search, polynomial overhead",
                "weighted_objective_in_scope": False,
                "weighted_warning": weighted_warning,
                "recommendation": (
                    "use structural certificate first"
                    if structural_bypass
                    else (
                        "use a specialized 3-color solver when practical; "
                        "Zamir supplies the generic sub-2^n existence result"
                    )
                ),
            }
        )
    return tuple(rows)


def planar_chromatic_decision_hierarchy() -> tuple[dict[str, Any], ...]:
    """Return the exact theorem-selection hierarchy for planar graphs."""

    return (
        {
            "stage": 1,
            "condition": "no edges",
            "outcome": "chi=1",
            "method": "direct inspection",
            "zamir_needed": False,
        },
        {
            "stage": 2,
            "condition": "has edges and is bipartite",
            "outcome": "chi=2",
            "method": "BFS/DFS odd-cycle test",
            "zamir_needed": False,
        },
        {
            "stage": 3,
            "condition": "triangle-free and non-bipartite",
            "outcome": "chi=3",
            "method": "Groetzsch upper bound plus odd-cycle lower bound",
            "zamir_needed": False,
        },
        {
            "stage": 4,
            "condition": "sphere triangulation",
            "outcome": "chi=3 iff all degrees are even; otherwise chi=4",
            "method": "Heawood parity criterion plus Four Color",
            "zamir_needed": False,
        },
        {
            "stage": 5,
            "condition": "remaining planar graph",
            "outcome": "chi=3 if 3-colorable; otherwise chi=4",
            "method": (
                "fixed 3-color decision plus Four Color; Zamir gives a "
                "generic randomized sub-2^n route"
            ),
            "zamir_needed": True,
        },
    )


def diagnose_planar_chromatic_case(
    *,
    has_edges: bool,
    bipartite: bool,
    triangle_free: bool = False,
    sphere_triangulation: bool = False,
    all_degrees_even: bool | None = None,
    three_colorable: bool | None = None,
) -> dict[str, Any]:
    """Diagnose a planar graph from already-certified structural facts."""

    if not has_edges:
        return {"chromatic_number": 1, "route": "edgeless", "resolved": True}
    if bipartite:
        return {"chromatic_number": 2, "route": "bipartite", "resolved": True}
    if triangle_free:
        return {
            "chromatic_number": 3,
            "route": "Groetzsch plus non-bipartite lower bound",
            "resolved": True,
        }
    if sphere_triangulation:
        if all_degrees_even is None:
            raise ValueError(
                "sphere triangulations require an all-degrees-even certificate"
            )
        return {
            "chromatic_number": 3 if all_degrees_even else 4,
            "route": "Heawood parity criterion",
            "resolved": True,
        }
    if three_colorable is not None:
        return {
            "chromatic_number": 3 if three_colorable else 4,
            "route": "3-color decision plus Four Color",
            "resolved": True,
        }
    return {
        "chromatic_number": None,
        "route": "requires fixed-palette 3-color decision",
        "resolved": False,
    }


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
