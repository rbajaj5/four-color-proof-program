from __future__ import annotations

import networkx as nx

from scripts.check_cubic_edge_smoothing_lifts import canonical_edges
from src.cubic_edge_smoothing import (
    InvalidSmoothing,
    SmoothingChild,
    enumerate_tait_colorings,
    lift_child_coloring,
    replacement_edges_are_adjacent,
    replacement_edges_are_bichromatic,
    smoothing_partition,
)


def assert_partition_identity(graph: nx.Graph) -> None:
    edges = canonical_edges(graph)
    original = set(enumerate_tait_colorings(edges))
    for central_edge_id in range(len(edges)):
        branches, lifted = smoothing_partition(edges, central_edge_id)
        assert len(lifted) == len(set(lifted))
        assert set(lifted) == original
        for branch in branches:
            if isinstance(branch, SmoothingChild):
                colorings = enumerate_tait_colorings(branch.edges)
                if replacement_edges_are_adjacent(branch):
                    assert all(
                        replacement_edges_are_bichromatic(branch, coloring)
                        for coloring in colorings
                    )
                for coloring in colorings:
                    parent = lift_child_coloring(branch, coloring)
                    assert (
                        parent is not None
                        if replacement_edges_are_bichromatic(branch, coloring)
                        else parent is None
                    )
                    assert parent is None or parent in original


def test_k4_partition_identity_and_loop_branch() -> None:
    graph = nx.tetrahedral_graph()
    assert_partition_identity(graph)
    edges = canonical_edges(graph)
    branches, _ = smoothing_partition(edges, 0)
    assert any(isinstance(branch, InvalidSmoothing) for branch in branches)
    assert any(isinstance(branch, SmoothingChild) for branch in branches)


def test_cube_partition_identity() -> None:
    assert_partition_identity(nx.cubical_graph())


def test_petersen_has_no_lifted_parent_coloring() -> None:
    graph = nx.petersen_graph()
    edges = canonical_edges(graph)
    assert enumerate_tait_colorings(edges) == ()
    saw_colorable_child = False
    for central_edge_id in range(len(edges)):
        branches, lifted = smoothing_partition(edges, central_edge_id)
        assert lifted == ()
        for branch in branches:
            if (
                isinstance(branch, SmoothingChild)
                and enumerate_tait_colorings(branch.edges)
            ):
                saw_colorable_child = True
    assert saw_colorable_child
