"""Exact finite checks inspired by Clifton-Salia plane saturation."""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from typing import Iterable


Edge = tuple[int, int]


def normalize_edges(edges: Iterable[Edge]) -> tuple[Edge, ...]:
    """Return a sorted simple loopless edge set."""

    normalized = set()
    for left, right in edges:
        if left == right:
            raise ValueError("simple graph fixtures cannot contain loops")
        normalized.add((min(left, right), max(left, right)))
    return tuple(sorted(normalized))


def adjacency(
    edges: Iterable[Edge],
    *,
    vertices: Iterable[int] | None = None,
) -> dict[int, frozenset[int]]:
    """Build an adjacency dictionary for a finite simple graph."""

    normalized = normalize_edges(edges)
    vertex_set = set(vertices or ())
    for left, right in normalized:
        vertex_set.add(left)
        vertex_set.add(right)
    out = {vertex: set() for vertex in vertex_set}
    for left, right in normalized:
        out[left].add(right)
        out[right].add(left)
    return {
        vertex: frozenset(neighbors)
        for vertex, neighbors in sorted(out.items())
    }


def twin_profile(
    edges: Iterable[Edge],
    *,
    vertices: Iterable[int] | None = None,
) -> dict[str, object]:
    """Count open-neighborhood twin multiplicities by degree."""

    graph = adjacency(edges, vertices=vertices)
    classes: dict[tuple[int, frozenset[int]], list[int]] = defaultdict(list)
    all_classes: dict[frozenset[int], list[int]] = defaultdict(list)
    for vertex, neighbors in graph.items():
        classes[(len(neighbors), neighbors)].append(vertex)
        all_classes[neighbors].append(vertex)

    def largest_class(degree: int) -> int:
        sizes = [
            len(members)
            for (class_degree, _), members in classes.items()
            if class_degree == degree
        ]
        return max(sizes, default=0)

    degree_one = largest_class(1)
    degree_two = largest_class(2)
    return {
        "vertex_count": len(graph),
        "edge_count": len(normalize_edges(edges)),
        "maximum_degree_1_twin_class": degree_one,
        "maximum_degree_2_twin_class": degree_two,
        "no_degree_1_or_2_twins": degree_one <= 1 and degree_two <= 1,
        "fully_twin_free": all(
            len(members) == 1
            for members in all_classes.values()
        ),
    }


def theorem_1_5_bound(k1: int, k2: int) -> Fraction:
    """Return the published strict lower bound when Theorem 1.5 applies."""

    if k1 <= 0 or k2 < 0:
        raise ValueError("Theorem 1.5 assumes k1 positive and k2 nonnegative")
    if (k1, k2) in {(1, 0), (2, 0)}:
        raise ValueError("this parameter pair is excluded from Theorem 1.5")
    return Fraction(1, 9 + k1 + 6 * k2)


def conjecture_4_1_bound(k1: int, k2: int) -> Fraction:
    """Return the conjectured strict lower bound, clearly labeled."""

    if k1 < 0 or k2 < 0:
        raise ValueError("twin multiplicity bounds must be nonnegative")
    return Fraction(1, 9 + k1 + 6 * k2)


def construction_one_row(m: int) -> dict[str, object]:
    """Audit the counts in Construction 1 and Claim 2.1."""

    if m < 7:
        raise ValueError("Construction 1 assumes m >= 7")
    vertices_g = 7 * m + 8
    edges_g = 16 * m + 8
    vertices_h = 7 * m + 8
    edges_h = m + 34
    ratio = Fraction(edges_h, edges_g)
    limit = Fraction(1, 16)
    published_upper_bound = limit + Fraction(3, m)
    return {
        "m": m,
        "vertices_G": vertices_g,
        "edges_G": edges_g,
        "vertices_H": vertices_h,
        "edges_H": edges_h,
        "edge_ratio": ratio,
        "distance_above_1_16": ratio - limit,
        "published_upper_bound": published_upper_bound,
        "claim_2_1_inequality_holds": ratio < published_upper_bound,
        "same_vertex_count": vertices_g == vertices_h,
    }


def construction_one_rows(
    values: Iterable[int] = (7, 16, 64, 256, 1024),
) -> list[dict[str, object]]:
    return [construction_one_row(value) for value in values]


def compensation_row(r2: int, r3: int) -> dict[str, object]:
    """Audit the two grouped inequalities in Sections 2.2.3-2.2.4."""

    if r2 < 0 or r3 < 0 or (r2 == 0 and r3 == 0):
        raise ValueError("r2 and r3 must be nonnegative and not both zero")
    first_ratio = Fraction(r2 + r3, 17 * r2 + 12 * r3)
    second_ratio = Fraction(r2 + r3, 15 * r2 + 18 * r3)
    first_condition = 4 * r3 >= r2
    second_condition = r2 >= 2 * r3
    return {
        "r2": r2,
        "r3": r3,
        "first_grouped_ratio": first_ratio,
        "first_condition_4r3_ge_r2": first_condition,
        "first_ratio_ge_1_16": first_ratio >= Fraction(1, 16),
        "second_grouped_ratio": second_ratio,
        "second_condition_r2_ge_2r3": second_condition,
        "second_ratio_ge_1_16": second_ratio >= Fraction(1, 16),
        "at_least_one_case_applies": first_condition or second_condition,
        "an_applicable_case_proves_1_16": (
            (first_condition and first_ratio >= Fraction(1, 16))
            or (second_condition and second_ratio >= Fraction(1, 16))
        ),
    }


def compensation_grid(max_count: int = 32) -> list[dict[str, object]]:
    if max_count <= 0:
        raise ValueError("max_count must be positive")
    return [
        compensation_row(r2, r3)
        for r2 in range(max_count + 1)
        for r3 in range(max_count + 1)
        if r2 or r3
    ]


def cycle_edges(size: int) -> tuple[Edge, ...]:
    if size < 3:
        raise ValueError("cycle size must be at least three")
    return normalize_edges((index, (index + 1) % size) for index in range(size))


def star_edges(leaf_count: int) -> tuple[Edge, ...]:
    if leaf_count <= 0:
        raise ValueError("leaf_count must be positive")
    return normalize_edges((0, leaf) for leaf in range(1, leaf_count + 1))


def complete_bipartite_edges(left_size: int, right_size: int) -> tuple[Edge, ...]:
    if left_size <= 0 or right_size <= 0:
        raise ValueError("part sizes must be positive")
    return normalize_edges(
        (left, left_size + right)
        for left in range(left_size)
        for right in range(right_size)
    )


def complete_graph_edges(size: int) -> tuple[Edge, ...]:
    if size <= 0:
        raise ValueError("size must be positive")
    return normalize_edges(
        (left, right)
        for left in range(size)
        for right in range(left + 1, size)
    )


def octahedral_edges() -> tuple[Edge, ...]:
    """Return K6 minus three disjoint antipodal edges."""

    excluded = {(0, 1), (2, 3), (4, 5)}
    return tuple(
        edge
        for edge in complete_graph_edges(6)
        if edge not in excluded
    )


def twin_fixture_rows() -> list[dict[str, object]]:
    fixtures = (
        ("cycle_C7", cycle_edges(7), range(7), True),
        ("star_K1_7", star_edges(7), range(8), True),
        ("complete_bipartite_K2_6", complete_bipartite_edges(2, 6), range(8), True),
        ("complete_graph_K4", complete_graph_edges(4), range(4), True),
        ("octahedral_graph", octahedral_edges(), range(6), True),
    )
    rows = []
    for fixture_id, edges, vertices, simple_planar in fixtures:
        profile = twin_profile(edges, vertices=vertices)
        low_degree_condition = bool(profile["no_degree_1_or_2_twins"])
        rows.append(
            {
                "fixture_id": fixture_id,
                **profile,
                "simple_planar_fixture": simple_planar,
                "theorem_1_3_low_degree_twin_hypothesis": (
                    low_degree_condition
                ),
                "certified_strict_lower_bound": (
                    "1/16" if low_degree_condition else "not_from_theorem_1_3"
                ),
            }
        )
    return rows


def coarse_grain_scope_rows() -> list[dict[str, object]]:
    """State exactly where plane-saturation results can enter the project."""

    return [
        {
            "object": "simple_planar_cell_adjacency_graph",
            "status": "compatible_after_saturation_check",
            "reason": (
                "The host is a finite simple planar graph; twin profiling and "
                "plane-saturated subgraph certification are meaningful."
            ),
        },
        {
            "object": "simple_four_regular_projection_graph",
            "status": "theorem_1_3_hypothesis_automatic",
            "reason": (
                "Minimum degree four excludes degree-1 and degree-2 twins, "
                "provided the projection graph is simple."
            ),
        },
        {
            "object": "knot_projection_multigraph",
            "status": "blocked_without_extension",
            "reason": (
                "Generic knot projections can have parallel edges or loops; "
                "the simple-graph theorem cannot be imported silently."
            ),
        },
        {
            "object": "simplified_underlying_projection_graph",
            "status": "blocked_without_preservation_lemma",
            "reason": (
                "Deleting loops or merging parallel edges can change both "
                "the host subgraph relation and saturation."
            ),
        },
        {
            "object": "region_dual_graph",
            "status": "separate_observable",
            "reason": (
                "Four-colorability concerns a coloring certificate, whereas "
                "plane saturation concerns maximality of a partial embedding."
            ),
        },
        {
            "object": "curvature_coarse_grained_graph",
            "status": "diagnostic_only",
            "reason": (
                "Track edge ratio, skeleton components, isolates, and low-"
                "degree twin classes at every scale; no topological "
                "preservation follows from those statistics alone."
            ),
        },
    ]
