from __future__ import annotations

import networkx as nx

from src.penrose_smoothing_landscape import (
    canonical_edges,
    chromatic_bucket,
    state_component_graph,
    state_nullity,
)
from src.picture_fourier_smoothing import (
    inverse_walsh,
    pointwise_product,
    strict_local_maximum_indicator,
    strict_nash_indicator,
    translation_kernel_is_positive,
    walsh_hadamard,
    xor_convolution,
)


def nullities(graph: nx.Graph) -> list[int]:
    edges = canonical_edges(graph)
    return [
        state_nullity(len(graph), edges, state)
        for state in range(1 << len(edges))
    ]


def test_walsh_is_involutive_up_to_cube_size() -> None:
    values = [3, -2, 5, 7, 0, 4, -1, 6]
    assert inverse_walsh(walsh_hadamard(values)) == values


def test_rotation_interchanges_convolution_and_multiplication() -> None:
    first = [1, 2, -1, 3]
    second = [0, -2, 4, 1]
    assert walsh_hadamard(xor_convolution(first, second)) == (
        pointwise_product(
            walsh_hadamard(first),
            walsh_hadamard(second),
        )
    )


def test_nullity_power_of_two_kernel_is_positive_on_controls() -> None:
    for graph in (
        nx.complete_graph(4),
        nx.convert_node_labels_to_integers(nx.graph_atlas(51)),
    ):
        values = nullities(graph)
        assert translation_kernel_is_positive([2**value for value in values])
        assert translation_kernel_is_positive([4**value for value in values])


def test_three_to_nullity_has_a_small_planar_counterexample() -> None:
    graph = nx.convert_node_labels_to_integers(nx.graph_atlas(48))
    spectrum = walsh_hadamard(
        [3**value for value in nullities(graph)]
    )
    assert min(spectrum) == -6


def test_colorable_state_indicator_is_not_reflection_positive_on_k4() -> None:
    graph = nx.complete_graph(4)
    edges = canonical_edges(graph)
    indicator = [
        int(
            chromatic_bucket(
                state_component_graph(graph, edges, state)
            )
            in {"1", "2", "3"}
        )
        for state in range(1 << len(edges))
    ]
    assert min(walsh_hadamard(indicator)) == -1


def test_strict_maxima_are_strict_nash_equilibria() -> None:
    graph = nx.convert_node_labels_to_integers(nx.graph_atlas(51))
    values = nullities(graph)
    assert strict_local_maximum_indicator(
        values,
        len(canonical_edges(graph)),
    ) == strict_nash_indicator(
        values,
        len(canonical_edges(graph)),
    )

