"""Exact finite Penrose/Tait calculations for cubic rotation systems."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Hashable, Mapping, Sequence


Vertex = Hashable
Edge = tuple[Vertex, Vertex]


@dataclass(frozen=True)
class CubicRotationSystem:
    """A cubic multigraph with a cyclic edge order at every vertex."""

    edges: tuple[Edge, ...]
    rotations: Mapping[Vertex, tuple[int, int, int]]

    def vertices(self) -> tuple[Vertex, ...]:
        values: set[Vertex] = set()
        for left, right in self.edges:
            values.add(left)
            values.add(right)
        return tuple(sorted(values, key=repr))

    def incident_edges(self) -> dict[Vertex, tuple[int, ...]]:
        incident: dict[Vertex, list[int]] = {
            vertex: [] for vertex in self.vertices()
        }
        for edge_id, (left, right) in enumerate(self.edges):
            if left == right:
                raise ValueError("loop edges are outside this fixture engine")
            incident[left].append(edge_id)
            incident[right].append(edge_id)
        return {
            vertex: tuple(edge_ids)
            for vertex, edge_ids in incident.items()
        }

    def validate(self) -> None:
        incident = self.incident_edges()
        if set(self.rotations) != set(incident):
            raise ValueError("rotation vertices must match graph vertices")
        for vertex, edge_ids in incident.items():
            if len(edge_ids) != 3:
                raise ValueError(f"vertex {vertex!r} is not cubic")
            rotation = tuple(self.rotations[vertex])
            if len(rotation) != 3 or set(rotation) != set(edge_ids):
                raise ValueError(
                    f"rotation at {vertex!r} must list its three incident edges"
                )


@dataclass(frozen=True)
class PenroseTaitResult:
    tait_coloring_count: int
    signed_epsilon_sum: int
    penrose_contraction: int


def levi_civita(colors: Sequence[int]) -> int:
    """Return the sign of a permutation of `(0,1,2)`, or zero."""

    if len(colors) != 3 or set(colors) != {0, 1, 2}:
        return 0
    inversions = sum(
        colors[left] > colors[right]
        for left in range(3)
        for right in range(left + 1, 3)
    )
    return -1 if inversions % 2 else 1


def rotation_from_coordinates(
    edges: Sequence[Edge],
    coordinates: Mapping[Vertex, tuple[float, float]],
) -> dict[Vertex, tuple[int, int, int]]:
    """Derive counterclockwise rotations from a straight-line drawing."""

    incident: dict[Vertex, list[tuple[float, int]]] = {
        vertex: [] for vertex in coordinates
    }
    for edge_id, (left, right) in enumerate(edges):
        if left not in coordinates or right not in coordinates:
            raise ValueError("every endpoint needs a coordinate")
        lx, ly = coordinates[left]
        rx, ry = coordinates[right]
        incident[left].append((math.atan2(ry - ly, rx - lx), edge_id))
        incident[right].append((math.atan2(ly - ry, lx - rx), edge_id))
    rotations: dict[Vertex, tuple[int, int, int]] = {}
    for vertex, angle_edges in incident.items():
        if len(angle_edges) != 3:
            raise ValueError(f"vertex {vertex!r} is not cubic")
        ordered = tuple(
            edge_id for _, edge_id in sorted(angle_edges)
        )
        rotations[vertex] = ordered
    return rotations


def evaluate_penrose_tait(
    system: CubicRotationSystem,
) -> PenroseTaitResult:
    """Count Tait colorings and their exact signed tensor contraction."""

    system.validate()
    vertices = system.vertices()
    incident = system.incident_edges()
    assignments = [-1] * len(system.edges)
    used = {vertex: set() for vertex in vertices}
    coloring_count = 0
    signed_sum = 0

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
                len(used[system.edges[edge_id][0]])
                + len(used[system.edges[edge_id][1]]),
                -edge_id,
            ),
        )

    def search() -> None:
        nonlocal coloring_count, signed_sum
        edge_id = choose_edge()
        if edge_id is None:
            vertex_sign = 1
            for vertex in vertices:
                colors = tuple(
                    assignments[index]
                    for index in system.rotations[vertex]
                )
                vertex_sign *= levi_civita(colors)
            coloring_count += 1
            signed_sum += vertex_sign
            return

        left, right = system.edges[edge_id]
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
    # Cubic graphs have an even number of vertices, so i^|V| is real.
    phase = -1 if (len(vertices) // 2) % 2 else 1
    return PenroseTaitResult(
        tait_coloring_count=coloring_count,
        signed_epsilon_sum=signed_sum,
        penrose_contraction=phase * signed_sum,
    )


def has_bridge(edges: Sequence[Edge]) -> bool:
    """Return whether an undirected loopless multigraph has a bridge."""

    vertices: set[Vertex] = set()
    adjacency: dict[Vertex, list[tuple[Vertex, int]]] = {}
    for edge_id, (left, right) in enumerate(edges):
        if left == right:
            raise ValueError("loop edges are outside this fixture engine")
        vertices.update((left, right))
        adjacency.setdefault(left, []).append((right, edge_id))
        adjacency.setdefault(right, []).append((left, edge_id))

    discovery: dict[Vertex, int] = {}
    low: dict[Vertex, int] = {}
    time = 0
    bridge_found = False

    def visit(vertex: Vertex, parent_edge: int | None) -> None:
        nonlocal time, bridge_found
        discovery[vertex] = time
        low[vertex] = time
        time += 1
        for neighbor, edge_id in adjacency[vertex]:
            if edge_id == parent_edge:
                continue
            if neighbor not in discovery:
                visit(neighbor, edge_id)
                low[vertex] = min(low[vertex], low[neighbor])
                if low[neighbor] > discovery[vertex]:
                    bridge_found = True
            else:
                low[vertex] = min(low[vertex], discovery[neighbor])

    for vertex in vertices:
        if vertex not in discovery:
            visit(vertex, None)
    return bridge_found
