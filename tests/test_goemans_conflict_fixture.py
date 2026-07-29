from fractions import Fraction

from src.goemans_conflict_fixture import (
    edge_relaxation_is_feasible,
    fractional_direct_cost,
    integral_rows,
    minimum_integral_cost,
    missing_triangle_inequality_slack,
)


def test_exact_58_60_conflict_core() -> None:
    assert edge_relaxation_is_feasible()
    assert fractional_direct_cost() == Fraction(58)
    assert minimum_integral_cost() == 60
    assert missing_triangle_inequality_slack() == Fraction(-1, 15)


def test_only_zero_or_one_integral_detours_are_feasible() -> None:
    rows = integral_rows()
    assert len(rows) == 8
    for row in rows:
        detour_count = sum(
            int(row[key])
            for key in ("detour_1", "detour_2", "detour_3")
        )
        assert bool(row["feasible"]) == (detour_count <= 1)
