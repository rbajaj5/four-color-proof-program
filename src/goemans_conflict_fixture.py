"""Exact conflict-core abstraction for splittable versus whole-route choice.

This is deliberately an abstract stable-set fixture.  It reproduces the
reported 58-versus-60 arithmetic behind a recent public discussion of the
cost-strengthened Goemans conjecture, but it does not certify that discussion's
directed path realization.
"""

from __future__ import annotations

from fractions import Fraction
import itertools


DETOUR_FRACTIONS = (
    Fraction(1, 3),
    Fraction(2, 5),
    Fraction(1, 3),
)
DIRECT_COSTS = (30, 30, 30)
CONFLICT_EDGES = ((0, 1), (0, 2), (1, 2))


def fractional_direct_cost() -> Fraction:
    """Return the exact cost of the reported fractional detour mixture."""

    return sum(
        Fraction(cost) * (1 - detour_fraction)
        for cost, detour_fraction in zip(
            DIRECT_COSTS,
            DETOUR_FRACTIONS,
            strict=True,
        )
    )


def integral_rows() -> tuple[dict[str, object], ...]:
    """Enumerate every integral detour/direct choice."""

    rows = []
    for detours in itertools.product((0, 1), repeat=3):
        feasible = all(
            detours[left] + detours[right] <= 1
            for left, right in CONFLICT_EDGES
        )
        cost = sum(
            direct_cost * (1 - detour)
            for direct_cost, detour in zip(
                DIRECT_COSTS,
                detours,
                strict=True,
            )
        )
        rows.append(
            {
                "detour_1": detours[0],
                "detour_2": detours[1],
                "detour_3": detours[2],
                "feasible": feasible,
                "direct_cost": cost,
            }
        )
    return tuple(rows)


def minimum_integral_cost() -> int:
    """Return the minimum direct cost among feasible whole-route choices."""

    return min(
        int(row["direct_cost"])
        for row in integral_rows()
        if bool(row["feasible"])
    )


def edge_relaxation_is_feasible() -> bool:
    """Check the three pairwise conflict inequalities exactly."""

    return all(
        DETOUR_FRACTIONS[left] + DETOUR_FRACTIONS[right] <= 1
        for left, right in CONFLICT_EDGES
    )


def missing_triangle_inequality_slack() -> Fraction:
    """Return ``1 - sum(z_i)`` for the omitted odd-cycle inequality."""

    return Fraction(1) - sum(DETOUR_FRACTIONS, start=Fraction(0))
