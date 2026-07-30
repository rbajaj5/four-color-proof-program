"""Exact fixed-label saturation checks for connected plane embeddings."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, Mapping, Sequence

from src.plane_saturation_gate import Edge, adjacency, normalize_edges


Rotation = Mapping[int, Sequence[int]]
Dart = tuple[int, int]


def validate_rotation_system(
    edges: Iterable[Edge],
    rotation: Rotation,
) -> tuple[Edge, ...]:
    """Validate a connected orientable rotation system."""

    edge_tuple = normalize_edges(edges)
    graph = adjacency(edge_tuple)
    if set(rotation) != set(graph):
        raise ValueError("rotation vertices must match graph vertices")
    for vertex, neighbors in graph.items():
        cyclic_order = tuple(rotation[vertex])
        if len(cyclic_order) != len(set(cyclic_order)):
            raise ValueError("a rotation cannot repeat a neighbor")
        if set(cyclic_order) != set(neighbors):
            raise ValueError(
                f"rotation at vertex {vertex} does not match its neighbors"
            )

    start = next(iter(graph), None)
    if start is None:
        raise ValueError("the embedded graph must be nonempty")
    seen = {start}
    stack = [start]
    while stack:
        vertex = stack.pop()
        for neighbor in graph[vertex]:
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    if seen != set(graph):
        raise ValueError("this checker requires a connected embedding")
    return edge_tuple


def face_boundaries(
    edges: Iterable[Edge],
    rotation: Rotation,
) -> tuple[tuple[int, ...], ...]:
    """Traverse all face boundaries of a connected rotation system."""

    edge_tuple = validate_rotation_system(edges, rotation)
    unvisited = {
        dart
        for left, right in edge_tuple
        for dart in ((left, right), (right, left))
    }
    faces: list[tuple[int, ...]] = []
    while unvisited:
        start = min(unvisited)
        current = start
        boundary: list[int] = []
        while True:
            if current not in unvisited:
                if current != start:
                    raise ValueError("rotation traversal closed at wrong dart")
                break
            unvisited.remove(current)
            left, right = current
            boundary.append(left)
            order = tuple(rotation[right])
            next_index = (order.index(left) + 1) % len(order)
            current = (right, order[next_index])
        faces.append(tuple(boundary))

    vertex_count = len(rotation)
    edge_count = len(edge_tuple)
    if vertex_count - edge_count + len(faces) != 2:
        raise ValueError(
            "rotation system is not a connected sphere embedding"
        )
    return tuple(sorted(faces, key=lambda face: (len(face), face)))


def cofacial_nonedge_pairs(
    edges: Iterable[Edge],
    rotation: Rotation,
) -> frozenset[Edge]:
    """Return nonedges whose endpoints share a face."""

    edge_set = set(normalize_edges(edges))
    pairs: set[Edge] = set()
    for boundary in face_boundaries(edges, rotation):
        boundary_vertices = sorted(set(boundary))
        for left, right in combinations(boundary_vertices, 2):
            edge = (left, right)
            if edge not in edge_set:
                pairs.add(edge)
    return frozenset(pairs)


def fixed_labeled_saturation_certificate(
    host_edges: Iterable[Edge],
    retained_edges: Iterable[Edge],
    rotation: Rotation,
) -> dict[str, object]:
    """Certify edge saturation for one fixed labeled spanning embedding.

    The result is deliberately narrower than Clifton-Salia plane saturation:
    labels and the embedding are fixed, all host vertices are already present,
    and only missing edges between existing vertices are considered.
    """

    host = normalize_edges(host_edges)
    retained = validate_rotation_system(retained_edges, rotation)
    host_vertices = set(adjacency(host))
    retained_vertices = set(adjacency(retained))
    if host_vertices != retained_vertices:
        raise ValueError("retained graph must span the host vertices")
    if not set(retained).issubset(host):
        raise ValueError("retained edges must be a subset of host edges")

    missing = tuple(sorted(set(host) - set(retained)))
    cofacial = cofacial_nonedge_pairs(retained, rotation)
    addable = tuple(edge for edge in missing if edge in cofacial)
    blocked = tuple(edge for edge in missing if edge not in cofacial)
    faces = face_boundaries(retained, rotation)
    return {
        "host_vertex_count": len(host_vertices),
        "host_edge_count": len(host),
        "retained_edge_count": len(retained),
        "edge_ratio_numerator": len(retained),
        "edge_ratio_denominator": len(host),
        "face_count": len(faces),
        "euler_characteristic": (
            len(retained_vertices) - len(retained) + len(faces)
        ),
        "missing_host_edges": missing,
        "cofacial_missing_host_edges": addable,
        "embedding_blocked_host_edges": blocked,
        "fixed_labeled_edge_saturated": not addable,
        "clifton_salia_plane_saturation_certified": False,
    }


def separator_triangle_fixture(
    *,
    blocked: bool,
) -> tuple[tuple[Edge, ...], tuple[Edge, ...], dict[int, tuple[int, ...]]]:
    """Return a triangle with leaves embedded on opposite sides."""

    retained = normalize_edges(
        (
            (0, 1),
            (1, 2),
            (2, 0),
            (0, 3),
            (1, 4),
        )
    )
    added_edge = (3, 4) if blocked else (0, 4)
    host = normalize_edges((*retained, added_edge))
    rotation = {
        0: (1, 3, 2),
        1: (2, 0, 4),
        2: (0, 1),
        3: (0,),
        4: (1,),
    }
    return host, retained, rotation


def fixed_embedding_fixture_rows() -> list[dict[str, object]]:
    """Evaluate saturated, unsaturated, and complete controls."""

    blocked_host, separator, separator_rotation = (
        separator_triangle_fixture(blocked=True)
    )
    addable_host, _, _ = separator_triangle_fixture(blocked=False)
    complete_host = normalize_edges(
        ((0, 1), (1, 2), (2, 0))
    )
    complete_rotation = {
        0: (1, 2),
        1: (2, 0),
        2: (0, 1),
    }
    fixtures = (
        (
            "separator_blocked_edge",
            blocked_host,
            separator,
            separator_rotation,
            True,
        ),
        (
            "separator_cofacial_edge",
            addable_host,
            separator,
            separator_rotation,
            False,
        ),
        (
            "complete_triangle_control",
            complete_host,
            complete_host,
            complete_rotation,
            True,
        ),
    )
    rows = []
    for fixture_id, host, retained, rotation, expected in fixtures:
        certificate = fixed_labeled_saturation_certificate(
            host,
            retained,
            rotation,
        )
        rows.append(
            {
                "fixture_id": fixture_id,
                **certificate,
                "expected_fixed_labeled_edge_saturated": expected,
                "matches_expected": (
                    certificate["fixed_labeled_edge_saturated"] == expected
                ),
            }
        )
    return rows
