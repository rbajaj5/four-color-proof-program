"""Exact reflection and gluing fixtures for planar picture languages."""

from __future__ import annotations

from itertools import product
from typing import Iterable


Pair = tuple[int, int]
Pairing = tuple[Pair, ...]
Vertex = tuple[str, str, str]


def noncrossing_pairings(pair_count: int) -> tuple[Pairing, ...]:
    """Enumerate noncrossing pairings of cyclically ordered boundary points."""
    if pair_count < 0:
        raise ValueError("pair_count must be nonnegative")

    def recurse(points: tuple[int, ...]) -> tuple[Pairing, ...]:
        if not points:
            return ((),)
        first = points[0]
        output: list[Pairing] = []
        for partner_index in range(1, len(points), 2):
            partner = points[partner_index]
            for inside in recurse(points[1:partner_index]):
                for outside in recurse(points[partner_index + 1 :]):
                    output.append(
                        tuple(
                            sorted(
                                ((first, partner), *inside, *outside),
                            )
                        )
                    )
        return tuple(output)

    return recurse(tuple(range(2 * pair_count)))


def reflect_pairing(pairing: Pairing, boundary_count: int) -> Pairing:
    """Reflect a boundary pairing across the vertical diameter."""
    if boundary_count < 0:
        raise ValueError("boundary_count must be nonnegative")
    reflected = []
    for left, right in pairing:
        mapped = sorted(
            (boundary_count - 1 - left, boundary_count - 1 - right)
        )
        reflected.append((mapped[0], mapped[1]))
    return tuple(sorted(reflected))


def gluing_loop_count(
    first: Pairing,
    second: Pairing,
    boundary_count: int,
) -> int:
    """Count loops obtained by gluing two boundary pairings."""
    parent = list(range(boundary_count))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for pairing in (first, second):
        for left, right in pairing:
            union(left, right)
    return len({find(index) for index in range(boundary_count)})


def loop_gram_matrix(pairings: Iterable[Pairing], loop_weight: int) -> list[list[int]]:
    """Return the reflected-gluing Gram matrix at integer loop weight."""
    selected = tuple(pairings)
    if loop_weight < 1:
        raise ValueError("loop_weight must be positive")
    boundary_count = 0 if not selected else 2 * len(selected[0])
    return [
        [
            loop_weight
            ** gluing_loop_count(left, right, boundary_count)
            for right in selected
        ]
        for left in selected
    ]


def pairing_color_assignments(
    pairing: Pairing,
    color_count: int,
) -> frozenset[tuple[int, ...]]:
    """Indicator-feature support for colors constant on paired endpoints."""
    if color_count < 1:
        raise ValueError("color_count must be positive")
    boundary_count = 2 * len(pairing)
    assignments = set()
    for pair_colors in product(range(color_count), repeat=len(pairing)):
        boundary = [-1] * boundary_count
        for color, (left, right) in zip(
            pair_colors,
            pairing,
            strict=True,
        ):
            boundary[left] = color
            boundary[right] = color
        assignments.add(tuple(boundary))
    return frozenset(assignments)


def feature_gram_matrix(
    pairings: Iterable[Pairing],
    color_count: int,
) -> list[list[int]]:
    """Construct the same Gram matrix as intersections of color features."""
    features = [
        pairing_color_assignments(pairing, color_count)
        for pairing in pairings
    ]
    return [
        [len(left & right) for right in features]
        for left in features
    ]


def levi_civita(first: int, second: int, third: int) -> int:
    """Three-dimensional Levi-Civita symbol."""
    if len({first, second, third}) < 3:
        return 0
    inversions = (
        int(first > second)
        + int(first > third)
        + int(second > third)
    )
    return -1 if inversions % 2 else 1


def boundary_amplitudes(
    vertices: Iterable[Vertex],
    boundary_edges: Iterable[str],
) -> dict[tuple[int, ...], int]:
    """Contract internal edge colors and retain a boundary amplitude vector."""
    selected_vertices = tuple(vertices)
    selected_boundary = tuple(boundary_edges)
    occurrences: dict[str, int] = {}
    for vertex in selected_vertices:
        if len(vertex) != 3:
            raise ValueError("every vertex must have three incident edges")
        for edge in vertex:
            occurrences[edge] = occurrences.get(edge, 0) + 1
    if len(set(selected_boundary)) != len(selected_boundary):
        raise ValueError("boundary edges must be distinct")
    boundary_set = set(selected_boundary)
    for edge, count in occurrences.items():
        expected = 1 if edge in boundary_set else 2
        if count != expected:
            raise ValueError(
                f"edge {edge!r} occurs {count} times; expected {expected}"
            )
    if boundary_set != {
        edge for edge, count in occurrences.items() if count == 1
    }:
        raise ValueError("boundary edge list does not match dangling edges")

    internal_edges = tuple(
        sorted(edge for edge, count in occurrences.items() if count == 2)
    )
    amplitudes: dict[tuple[int, ...], int] = {}
    for boundary_colors in product(range(3), repeat=len(selected_boundary)):
        fixed = dict(zip(selected_boundary, boundary_colors, strict=True))
        total = 0
        for internal_colors in product(range(3), repeat=len(internal_edges)):
            coloring = fixed | dict(
                zip(internal_edges, internal_colors, strict=True)
            )
            term = 1
            for first, second, third in selected_vertices:
                term *= levi_civita(
                    coloring[first],
                    coloring[second],
                    coloring[third],
                )
            total += term
        amplitudes[boundary_colors] = total
    return amplitudes


def amplitude_inner_product(
    first: dict[tuple[int, ...], int],
    second: dict[tuple[int, ...], int],
) -> int:
    """Glue two compatible half-pictures along their labeled boundary."""
    if first.keys() != second.keys():
        raise ValueError("boundary amplitude domains do not match")
    return sum(first[key] * second[key] for key in first)

