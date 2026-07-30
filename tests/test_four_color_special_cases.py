from __future__ import annotations

from fractions import Fraction

from src.four_color_special_cases import (
    SPECIAL_CASES,
    ZAMIR_REFERENCE,
    diagnose_planar_chromatic_case,
    goemans_style_weighted_diagnosis,
    planar_chromatic_decision_hierarchy,
    validate_special_case_registry,
    zamir_solver_diagnoses,
)


def test_special_case_registry_is_well_formed() -> None:
    validate_special_case_registry()
    identifiers = {row["case_id"] for row in SPECIAL_CASES}
    assert len(identifiers) == len(SPECIAL_CASES)
    assert "curvature_generated_triangulation" in identifiers
    assert "weighted_planar_conflict_triangle" in identifiers


def test_goemans_style_control_keeps_coloring_and_integrality_separate() -> None:
    diagnosis = goemans_style_weighted_diagnosis()
    assert diagnosis["edge_relaxation_feasible"]
    assert Fraction(diagnosis["fractional_direct_cost"]) == 58
    assert diagnosis["minimum_integral_direct_cost"] == 60
    assert Fraction(diagnosis["integrality_gap"]) == 2
    assert Fraction(diagnosis["missing_odd_cycle_slack"]) == Fraction(-1, 15)
    assert not diagnosis["coloring_implies_integral_lift"]


def test_zamir_layer_prefers_structural_solvers_and_excludes_weights() -> None:
    rows = {row["case_id"]: row for row in zamir_solver_diagnoses()}
    assert ZAMIR_REFERENCE["arxiv"] == "2607.25973"
    assert rows["eulerian_sphere_triangulation"][
        "structural_bypass_generic_search"
    ]
    assert not rows["arbitrary_planar_graph"][
        "structural_bypass_generic_search"
    ]
    assert rows["arbitrary_planar_graph"]["target_k"] == 3
    assert rows["dual_cubic_tait_instance"]["problem_formulation"] == (
        "vertex coloring of the line graph"
    )
    assert "sub-2^n" in rows["arbitrary_planar_graph"]["recommendation"]
    assert not rows["weighted_planar_conflict_triangle"][
        "weighted_objective_in_scope"
    ]
    assert rows["weighted_planar_conflict_triangle"]["weighted_warning"]


def test_planar_chromatic_decision_hierarchy_is_exact_on_certified_cases() -> None:
    assert len(planar_chromatic_decision_hierarchy()) == 5
    assert diagnose_planar_chromatic_case(
        has_edges=False,
        bipartite=True,
    )["chromatic_number"] == 1
    assert diagnose_planar_chromatic_case(
        has_edges=True,
        bipartite=True,
    )["chromatic_number"] == 2
    assert diagnose_planar_chromatic_case(
        has_edges=True,
        bipartite=False,
        triangle_free=True,
    )["chromatic_number"] == 3
    assert diagnose_planar_chromatic_case(
        has_edges=True,
        bipartite=False,
        sphere_triangulation=True,
        all_degrees_even=False,
    )["chromatic_number"] == 4
    unresolved = diagnose_planar_chromatic_case(
        has_edges=True,
        bipartite=False,
    )
    assert not unresolved["resolved"]
    assert unresolved["chromatic_number"] is None


def test_sphere_triangulation_diagnosis_requires_parity_certificate() -> None:
    try:
        diagnose_planar_chromatic_case(
            has_edges=True,
            bipartite=False,
            sphere_triangulation=True,
        )
    except ValueError as error:
        assert "all-degrees-even" in str(error)
    else:
        raise AssertionError("missing parity certificate should fail")
