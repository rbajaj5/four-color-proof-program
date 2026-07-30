from fractions import Fraction

import pytest

from src.plane_saturation_gate import (
    compensation_grid,
    compensation_row,
    construction_one_row,
    construction_one_rows,
    conjecture_4_1_bound,
    coarse_grain_scope_rows,
    theorem_1_5_bound,
    twin_fixture_rows,
)


def test_construction_one_counts_and_claim_inequality() -> None:
    row = construction_one_row(7)
    assert row["vertices_G"] == row["vertices_H"] == 57
    assert row["edges_G"] == 120
    assert row["edges_H"] == 41
    assert row["edge_ratio"] == Fraction(41, 120)
    assert row["claim_2_1_inequality_holds"]


def test_construction_ratio_converges_down_to_one_sixteenth() -> None:
    rows = construction_one_rows()
    distances = [row["distance_above_1_16"] for row in rows]
    assert all(distance > 0 for distance in distances)
    assert distances == sorted(distances, reverse=True)
    assert distances[-1] < Fraction(1, 400)


def test_two_compensation_cases_cover_every_nonzero_pair() -> None:
    rows = compensation_grid(64)
    assert len(rows) == 65 * 65 - 1
    assert all(row["at_least_one_case_applies"] for row in rows)
    assert all(row["an_applicable_case_proves_1_16"] for row in rows)


def test_compensation_boundaries_are_exact() -> None:
    first_boundary = compensation_row(4, 1)
    assert first_boundary["first_grouped_ratio"] == Fraction(1, 16)
    assert first_boundary["first_condition_4r3_ge_r2"]

    second_boundary = compensation_row(2, 1)
    assert second_boundary["second_grouped_ratio"] == Fraction(1, 16)
    assert second_boundary["second_condition_r2_ge_2r3"]


def test_published_and_conjectured_parameter_status_remain_distinct() -> None:
    assert theorem_1_5_bound(1, 1) == Fraction(1, 16)
    assert theorem_1_5_bound(3, 0) == Fraction(1, 12)
    assert conjecture_4_1_bound(0, 0) == Fraction(1, 9)
    with pytest.raises(ValueError, match="excluded"):
        theorem_1_5_bound(1, 0)


def test_low_degree_twins_diagnose_fixture_scope() -> None:
    rows = {row["fixture_id"]: row for row in twin_fixture_rows()}
    assert rows["cycle_C7"]["theorem_1_3_low_degree_twin_hypothesis"]
    assert not rows["star_K1_7"]["theorem_1_3_low_degree_twin_hypothesis"]
    assert not rows["complete_bipartite_K2_6"][
        "theorem_1_3_low_degree_twin_hypothesis"
    ]
    assert rows["complete_graph_K4"][
        "theorem_1_3_low_degree_twin_hypothesis"
    ]
    assert rows["octahedral_graph"][
        "theorem_1_3_low_degree_twin_hypothesis"
    ]
    assert not rows["octahedral_graph"]["fully_twin_free"]


def test_scope_matrix_does_not_apply_simple_graph_theorem_to_multigraphs() -> None:
    rows = {row["object"]: row for row in coarse_grain_scope_rows()}
    assert rows["knot_projection_multigraph"]["status"] == (
        "blocked_without_extension"
    )
    assert rows["region_dual_graph"]["status"] == "separate_observable"
