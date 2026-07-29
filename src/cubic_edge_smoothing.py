"""Exact edge-smoothing and Tait-coloring lift checks for cubic graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence


Vertex = Hashable
Edge = tuple[Vertex, Vertex]
Coloring = tuple[int, ...]


@dataclass(frozen=True)
class SmoothingChild:
    """One of the two pairings obtained by smoothing an edge of a cubic graph."""

    original_edges: tuple[Edge, ...]
    central_edge_id: int
    pairing_index: int
    removed_vertices: tuple[Vertex, Vertex]
    edges: tuple[Edge, ...]
    child_to_original_edge: tuple[int | None, ...]
    replacement_edge_ids: tuple[int, int]
    replacement_source_pairs: tuple[tuple[int, int], tuple[int, int]]


@dataclass(frozen=True)
class InvalidSmoothing:
    """A smoothing pairing that leaves the loopless cubic category."""

    central_edge_id: int
    pairing_index: int
    reason: str


def _incident_edge_ids(edges: Sequence[Edge]) -> dict[Vertex, list[int]]:
    incident: dict[Vertex, list[int]] = {}
    for edge_id, (left, right) in enumerate(edges):
        if left == right:
            raise ValueError("loop edges are outside this experiment")
        incident.setdefault(left, []).append(edge_id)
        incident.setdefault(right, []).append(edge_id)
    return incident


def validate_loopless_cubic(edges: Sequence[Edge]) -> None:
    """Raise if ``edges`` does not describe a loopless cubic multigraph."""

    if not edges:
        raise ValueError("a cubic graph must contain at least one edge")
    incident = _incident_edge_ids(edges)
    for vertex, edge_ids in incident.items():
        if len(edge_ids) != 3:
            raise ValueError(
                f"vertex {vertex!r} has degree {len(edge_ids)}, not three"
            )


def enumerate_tait_colorings(edges: Sequence[Edge]) -> tuple[Coloring, ...]:
    """Enumerate all proper edge 3-colorings of a cubic multigraph."""

    edge_tuple = tuple(edges)
    validate_loopless_cubic(edge_tuple)
    incident = _incident_edge_ids(edge_tuple)
    assignments = [-1] * len(edge_tuple)
    used = {vertex: set() for vertex in incident}
    colorings: list[Coloring] = []

    def choose_edge() -> int | None:
        candidates = [
            edge_id
            for edge_id, color in enumerate(assignments)
            if color < 0
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda edge_id: (
                len(used[edge_tuple[edge_id][0]])
                + len(used[edge_tuple[edge_id][1]]),
                -edge_id,
            ),
        )

    def search() -> None:
        edge_id = choose_edge()
        if edge_id is None:
            colorings.append(tuple(assignments))
            return

        left, right = edge_tuple[edge_id]
        forbidden = used[left] | used[right]
        for color in range(3):
            if color in forbidden:
                continue
            assignments[edge_id] = color
            used[left].add(color)
            used[right].add(color)
            search()
            used[left].remove(color)
            used[right].remove(color)
            assignments[edge_id] = -1

    search()
    return tuple(colorings)


def _far_endpoint(edge: Edge, vertex: Vertex) -> Vertex:
    left, right = edge
    if left == vertex:
        return right
    if right == vertex:
        return left
    raise ValueError(f"edge {edge!r} is not incident to {vertex!r}")


def smooth_cubic_edge(
    edges: Sequence[Edge],
    central_edge_id: int,
    pairing_index: int,
) -> SmoothingChild | InvalidSmoothing:
    """Delete an edge and its endpoints, then pair the four dangling ends."""

    original_edges = tuple(edges)
    validate_loopless_cubic(original_edges)
    if central_edge_id < 0 or central_edge_id >= len(original_edges):
        raise IndexError("central edge id is out of range")
    if pairing_index not in (0, 1):
        raise ValueError("pairing_index must be zero or one")

    left_vertex, right_vertex = original_edges[central_edge_id]
    incident = _incident_edge_ids(original_edges)
    left_edges = sorted(
        edge_id
        for edge_id in incident[left_vertex]
        if edge_id != central_edge_id
    )
    right_edges = sorted(
        edge_id
        for edge_id in incident[right_vertex]
        if edge_id != central_edge_id
    )
    if len(left_edges) != 2 or len(right_edges) != 2:
        raise ValueError("central edge endpoints must each have two other edges")

    if pairing_index == 0:
        source_pairs = (
            (left_edges[0], right_edges[0]),
            (left_edges[1], right_edges[1]),
        )
    else:
        source_pairs = (
            (left_edges[0], right_edges[1]),
            (left_edges[1], right_edges[0]),
        )

    replacement_edges = tuple(
        (
            _far_endpoint(original_edges[left_edge_id], left_vertex),
            _far_endpoint(original_edges[right_edge_id], right_vertex),
        )
        for left_edge_id, right_edge_id in source_pairs
    )
    if any(left == right for left, right in replacement_edges):
        return InvalidSmoothing(
            central_edge_id=central_edge_id,
            pairing_index=pairing_index,
            reason="pairing creates a loop",
        )

    removed_edge_ids = set(
        incident[left_vertex] + incident[right_vertex]
    )
    child_edges: list[Edge] = []
    child_to_original: list[int | None] = []
    for edge_id, edge in enumerate(original_edges):
        if edge_id in removed_edge_ids:
            continue
        child_edges.append(edge)
        child_to_original.append(edge_id)

    replacement_edge_ids = (
        len(child_edges),
        len(child_edges) + 1,
    )
    child_edges.extend(replacement_edges)
    child_to_original.extend((None, None))
    child = SmoothingChild(
        original_edges=original_edges,
        central_edge_id=central_edge_id,
        pairing_index=pairing_index,
        removed_vertices=(left_vertex, right_vertex),
        edges=tuple(child_edges),
        child_to_original_edge=tuple(child_to_original),
        replacement_edge_ids=replacement_edge_ids,
        replacement_source_pairs=source_pairs,
    )
    validate_loopless_cubic(child.edges)
    return child


def lift_child_coloring(
    child: SmoothingChild,
    child_coloring: Sequence[int],
) -> Coloring | None:
    """Lift a child coloring to the parent, or return ``None`` if it cannot."""

    if len(child_coloring) != len(child.edges):
        raise ValueError("child coloring length does not match child edges")
    if any(color not in (0, 1, 2) for color in child_coloring):
        raise ValueError("colors must lie in {0, 1, 2}")

    parent = [-1] * len(child.original_edges)
    for child_edge_id, original_edge_id in enumerate(
        child.child_to_original_edge
    ):
        if original_edge_id is not None:
            parent[original_edge_id] = child_coloring[child_edge_id]

    for replacement_edge_id, source_pair in zip(
        child.replacement_edge_ids,
        child.replacement_source_pairs,
        strict=True,
    ):
        color = child_coloring[replacement_edge_id]
        for original_edge_id in source_pair:
            parent[original_edge_id] = color

    left_vertex, right_vertex = child.removed_vertices
    incident = _incident_edge_ids(child.original_edges)
    left_other_colors = {
        parent[edge_id]
        for edge_id in incident[left_vertex]
        if edge_id != child.central_edge_id
    }
    right_other_colors = {
        parent[edge_id]
        for edge_id in incident[right_vertex]
        if edge_id != child.central_edge_id
    }
    if len(left_other_colors) != 2 or len(right_other_colors) != 2:
        return None

    left_missing = {0, 1, 2} - left_other_colors
    right_missing = {0, 1, 2} - right_other_colors
    if left_missing != right_missing:
        return None
    parent[child.central_edge_id] = left_missing.pop()

    if any(color < 0 for color in parent):
        raise AssertionError("internal error: incomplete lifted coloring")
    parent_coloring = tuple(parent)
    if not is_proper_tait_coloring(child.original_edges, parent_coloring):
        raise AssertionError("internal error: reconstructed coloring is improper")
    return parent_coloring


def replacement_edges_are_bichromatic(
    child: SmoothingChild,
    child_coloring: Sequence[int],
) -> bool:
    """Return whether the two smoothing replacement edges have distinct colors."""

    if len(child_coloring) != len(child.edges):
        raise ValueError("child coloring length does not match child edges")
    first, second = child.replacement_edge_ids
    return child_coloring[first] != child_coloring[second]


def replacement_edges_are_adjacent(child: SmoothingChild) -> bool:
    """Return whether the two replacement edges share a child vertex."""

    first, second = child.replacement_edge_ids
    return bool(set(child.edges[first]) & set(child.edges[second]))


def is_proper_tait_coloring(
    edges: Sequence[Edge],
    coloring: Sequence[int],
) -> bool:
    """Return whether a color vector is a proper Tait edge coloring."""

    if len(edges) != len(coloring):
        return False
    try:
        incident = _incident_edge_ids(edges)
    except ValueError:
        return False
    return all(
        {coloring[edge_id] for edge_id in edge_ids} == {0, 1, 2}
        for edge_ids in incident.values()
    )


def liftable_parent_colorings(
    child: SmoothingChild,
) -> tuple[Coloring, ...]:
    """Return the distinct parent colorings reconstructed from one child."""

    lifted = {
        parent
        for coloring in enumerate_tait_colorings(child.edges)
        if replacement_edges_are_bichromatic(child, coloring)
        and (parent := lift_child_coloring(child, coloring)) is not None
    }
    return tuple(sorted(lifted))


def smoothing_partition(
    edges: Sequence[Edge],
    central_edge_id: int,
) -> tuple[
    tuple[SmoothingChild | InvalidSmoothing, ...],
    tuple[Coloring, ...],
]:
    """Compute both smoothings and their disjoint lifted parent colorings."""

    branches = tuple(
        smooth_cubic_edge(edges, central_edge_id, pairing_index)
        for pairing_index in (0, 1)
    )
    lifted: list[Coloring] = []
    for branch in branches:
        if isinstance(branch, SmoothingChild):
            lifted.extend(liftable_parent_colorings(branch))
    return branches, tuple(lifted)
