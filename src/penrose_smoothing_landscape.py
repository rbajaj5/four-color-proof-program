"""Exact smoothing-state reconstruction for plane Tait graphs."""

from __future__ import annotations

from collections.abc import Sequence

import networkx as nx
import numpy as np


Edge = tuple[int, int]


def canonical_edges(graph: nx.Graph) -> tuple[Edge, ...]:
    """Return a stable edge order for a simple integer-labeled graph."""

    if nx.number_of_selfloops(graph):
        raise ValueError("loop edges are outside the smoothing census")
    return tuple(
        sorted(
            (min(int(left), int(right)), max(int(left), int(right)))
            for left, right in graph.edges()
        )
    )


def gf2_rank(matrix: np.ndarray) -> int:
    """Compute matrix rank over GF(2) by exact row reduction."""

    reduced = np.asarray(matrix, dtype=np.uint8).copy() % 2
    row = 0
    for column in range(reduced.shape[1]):
        pivot = next(
            (
                candidate
                for candidate in range(row, reduced.shape[0])
                if reduced[candidate, column]
            ),
            None,
        )
        if pivot is None:
            continue
        reduced[[row, pivot]] = reduced[[pivot, row]]
        for candidate in range(reduced.shape[0]):
            if candidate != row and reduced[candidate, column]:
                reduced[candidate] ^= reduced[row]
        row += 1
        if row == reduced.shape[0]:
            break
    return row


def state_laplacian(
    vertex_count: int,
    edges: Sequence[Edge],
    state_index: int,
) -> np.ndarray:
    """Return the mod-2 Laplacian of the edge subset encoded by a state."""

    matrix = np.zeros((vertex_count, vertex_count), dtype=np.uint8)
    for edge_id, (left, right) in enumerate(edges):
        if not (state_index >> edge_id) & 1:
            continue
        matrix[left, left] ^= 1
        matrix[right, right] ^= 1
        matrix[left, right] ^= 1
        matrix[right, left] ^= 1
    return matrix


def state_nullity(
    vertex_count: int,
    edges: Sequence[Edge],
    state_index: int,
) -> int:
    """Return the exact GF(2) Laplacian nullity of one smoothing state."""

    return vertex_count - gf2_rank(
        state_laplacian(vertex_count, edges, state_index)
    )


def is_strict_local_maximum(
    vertex_count: int,
    edges: Sequence[Edge],
    state_index: int,
) -> bool:
    """Check strict local maximality in the Boolean smoothing cube."""

    value = state_nullity(vertex_count, edges, state_index)
    return all(
        value
        > state_nullity(
            vertex_count,
            edges,
            state_index ^ (1 << edge_id),
        )
        for edge_id in range(len(edges))
    )


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[tuple[int, int, int], tuple[int, int, int]] = {}

    def find(self, item: tuple[int, int, int]) -> tuple[int, int, int]:
        self.parent.setdefault(item, item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(
        self,
        left: tuple[int, int, int],
        right: tuple[int, int, int],
    ) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def state_component_graph(
    graph: nx.Graph,
    edges: Sequence[Edge],
    state_index: int,
) -> nx.Graph:
    """Reconstruct the component graph of a plane smoothing state.

    The zero state follows a separate circle around each Tait-graph vertex.
    Toggling an edge installs the crossing pairing. The pairing convention is
    orientation-compatible with the planar embedding and reproduces the
    mod-2 Laplacian nullity formula used by Kauffman-Silver-Williams.
    """

    planar, embedding = nx.check_planarity(graph)
    if not planar:
        raise ValueError("component reconstruction requires a plane graph")
    if set(graph) != set(range(len(graph))):
        raise ValueError("vertices must be labeled consecutively from zero")

    edge_tuple = tuple(edges)
    edge_ids = {
        frozenset((left, right)): edge_id
        for edge_id, (left, right) in enumerate(edge_tuple)
    }
    union_find = _UnionFind()

    def token(vertex: int, edge_id: int, side: int) -> tuple[int, int, int]:
        return vertex, edge_id, side

    for vertex in graph:
        incident = [
            edge_ids[frozenset((vertex, neighbor))]
            for neighbor in embedding.neighbors_cw_order(vertex)
        ]
        for index, edge_id in enumerate(incident):
            next_edge = incident[(index + 1) % len(incident)]
            union_find.union(
                token(vertex, edge_id, 1),
                token(vertex, next_edge, 0),
            )

    local_arcs: list[
        tuple[
            tuple[tuple[int, int, int], tuple[int, int, int]],
            tuple[tuple[int, int, int], tuple[int, int, int]],
        ]
    ] = []
    for edge_id, (left, right) in enumerate(edge_tuple):
        if (state_index >> edge_id) & 1:
            pairings = (
                (token(left, edge_id, 0), token(right, edge_id, 0)),
                (token(left, edge_id, 1), token(right, edge_id, 1)),
            )
        else:
            pairings = (
                (token(left, edge_id, 0), token(left, edge_id, 1)),
                (token(right, edge_id, 0), token(right, edge_id, 1)),
            )
        for first, second in pairings:
            union_find.union(first, second)
        local_arcs.append(pairings)

    roots = {
        union_find.find(item)
        for item in union_find.parent
    }
    root_ids = {
        root: component_id
        for component_id, root in enumerate(
            sorted(roots, key=repr)
        )
    }
    component_graph = nx.Graph()
    component_graph.add_nodes_from(range(len(root_ids)))
    for pairings in local_arcs:
        first = root_ids[union_find.find(pairings[0][0])]
        second = root_ids[union_find.find(pairings[1][0])]
        component_graph.add_edge(first, second)
    return component_graph


def is_k_colorable(graph: nx.Graph, color_count: int) -> bool:
    """Decide small fixed-palette graph coloring by exact DSATUR search."""

    if color_count < 1:
        return not graph
    if nx.number_of_selfloops(graph):
        return False
    colors: dict[int, int] = {}

    def search() -> bool:
        if len(colors) == len(graph):
            return True
        vertex = max(
            (candidate for candidate in graph if candidate not in colors),
            key=lambda candidate: (
                len(
                    {
                        colors[neighbor]
                        for neighbor in graph[candidate]
                        if neighbor in colors
                    }
                ),
                graph.degree(candidate),
                -int(candidate),
            ),
        )
        forbidden = {
            colors[neighbor]
            for neighbor in graph[vertex]
            if neighbor in colors
        }
        for color in range(color_count):
            if color in forbidden:
                continue
            colors[vertex] = color
            if search():
                return True
            del colors[vertex]
        return False

    return search()


def chromatic_bucket(graph: nx.Graph) -> str:
    """Return the exact chromatic number through four, or the bucket `>4`."""

    for color_count in range(1, 5):
        if is_k_colorable(graph, color_count):
            return str(color_count)
    return ">4"
