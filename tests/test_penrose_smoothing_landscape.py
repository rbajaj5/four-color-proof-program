from __future__ import annotations

import networkx as nx

from src.penrose_smoothing_landscape import (
    canonical_edges,
    chromatic_bucket,
    is_strict_local_maximum,
    state_component_graph,
    state_nullity,
)


def test_component_count_matches_nullity_on_k4() -> None:
    graph = nx.complete_graph(4)
    edges = canonical_edges(graph)
    for state_index in range(1 << len(edges)):
        component_graph = state_component_graph(
            graph,
            edges,
            state_index,
        )
        assert len(component_graph) == state_nullity(
            len(graph),
            edges,
            state_index,
        )


def test_strict_maxima_are_exactly_loopless_states_on_k4() -> None:
    graph = nx.complete_graph(4)
    edges = canonical_edges(graph)
    maxima: list[int] = []
    for state_index in range(1 << len(edges)):
        component_graph = state_component_graph(
            graph,
            edges,
            state_index,
        )
        loopless = nx.number_of_selfloops(component_graph) == 0
        strict = is_strict_local_maximum(
            len(graph),
            edges,
            state_index,
        )
        assert strict == loopless
        if strict:
            maxima.append(state_index)
    assert maxima == [0, (1 << len(edges)) - 1]


def test_k4_nonzero_maximum_is_k3() -> None:
    graph = nx.complete_graph(4)
    edges = canonical_edges(graph)
    zero_graph = state_component_graph(graph, edges, 0)
    nonzero_graph = state_component_graph(
        graph,
        edges,
        (1 << len(edges)) - 1,
    )
    assert chromatic_bucket(zero_graph) == "4"
    assert nx.is_isomorphic(nonzero_graph, nx.complete_graph(3))
    assert chromatic_bucket(nonzero_graph) == "3"


def test_universal_nonzero_maximum_rule_has_small_counterexample() -> None:
    graph = nx.convert_node_labels_to_integers(nx.graph_atlas(51))
    edges = canonical_edges(graph)
    assert nx.check_planarity(graph)[0]
    assert not any(
        chromatic_bucket(graph) == str(color_count)
        for color_count in range(1, 4)
    )
    bad_nonzero = []
    for state_index in range(1, 1 << len(edges)):
        if not is_strict_local_maximum(
            len(graph),
            edges,
            state_index,
        ):
            continue
        component_graph = state_component_graph(
            graph,
            edges,
            state_index,
        )
        if chromatic_bucket(component_graph) not in {"1", "2", "3"}:
            bad_nonzero.append(state_index)
    assert bad_nonzero == [311, 504]
