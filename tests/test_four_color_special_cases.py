from __future__ import annotations

from fractions import Fraction

from src.four_color_special_cases import (
    SPECIAL_CASES,
    goemans_style_weighted_diagnosis,
    validate_special_case_registry,
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
