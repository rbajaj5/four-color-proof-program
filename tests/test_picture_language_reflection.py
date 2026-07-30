from src.picture_language_reflection import (
    amplitude_inner_product,
    boundary_amplitudes,
    feature_gram_matrix,
    loop_gram_matrix,
    noncrossing_pairings,
    reflect_pairing,
)


def test_noncrossing_pairing_counts_are_catalan() -> None:
    assert [len(noncrossing_pairings(n)) for n in range(6)] == [
        1,
        1,
        2,
        5,
        14,
        42,
    ]


def test_reflection_is_an_involution() -> None:
    for pair_count in range(1, 6):
        boundary_count = 2 * pair_count
        for pairing in noncrossing_pairings(pair_count):
            assert (
                reflect_pairing(
                    reflect_pairing(pairing, boundary_count),
                    boundary_count,
                )
                == pairing
            )


def test_loop_gram_has_constructive_color_feature_factorization() -> None:
    for pair_count in range(1, 5):
        pairings = noncrossing_pairings(pair_count)
        for color_count in (2, 3):
            assert loop_gram_matrix(pairings, color_count) == (
                feature_gram_matrix(pairings, color_count)
            )


def test_penrose_tripod_double_is_positive_but_mixed_gluing_can_be_negative() -> None:
    tripod = boundary_amplitudes((("a", "b", "c"),), ("a", "b", "c"))
    reversed_tripod = boundary_amplitudes(
        (("b", "a", "c"),),
        ("a", "b", "c"),
    )
    assert amplitude_inner_product(tripod, tripod) == 6
    assert amplitude_inner_product(reversed_tripod, reversed_tripod) == 6
    assert amplitude_inner_product(tripod, reversed_tripod) == -6


def test_two_vertex_channel_has_positive_reflection_double() -> None:
    channel = boundary_amplitudes(
        (
            ("a", "b", "x"),
            ("x", "c", "d"),
        ),
        ("a", "b", "c", "d"),
    )
    assert amplitude_inner_product(channel, channel) == 12

