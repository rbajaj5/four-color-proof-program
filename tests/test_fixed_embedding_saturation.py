import pytest

from src.fixed_embedding_saturation import (
    cofacial_nonedge_pairs,
    face_boundaries,
    fixed_embedding_fixture_rows,
    fixed_labeled_saturation_certificate,
    separator_triangle_fixture,
)


def test_triangle_rotation_has_two_faces() -> None:
    edges = ((0, 1), (1, 2), (0, 2))
    rotation = {
        0: (1, 2),
        1: (2, 0),
        2: (0, 1),
    }
    faces = face_boundaries(edges, rotation)
    assert len(faces) == 2
    assert all(set(face) == {0, 1, 2} for face in faces)


def test_separator_leaves_occupy_different_faces() -> None:
    _, retained, rotation = separator_triangle_fixture(blocked=True)
    faces = face_boundaries(retained, rotation)
    assert len(faces) == 2
    assert not any({3, 4}.issubset(face) for face in map(set, faces))
    assert (3, 4) not in cofacial_nonedge_pairs(retained, rotation)
    assert (0, 4) in cofacial_nonedge_pairs(retained, rotation)


def test_separator_blocks_one_host_edge_but_not_a_cofacial_edge() -> None:
    blocked_host, retained, rotation = separator_triangle_fixture(
        blocked=True
    )
    blocked = fixed_labeled_saturation_certificate(
        blocked_host,
        retained,
        rotation,
    )
    assert blocked["fixed_labeled_edge_saturated"]
    assert blocked["embedding_blocked_host_edges"] == ((3, 4),)

    addable_host, retained, rotation = separator_triangle_fixture(
        blocked=False
    )
    addable = fixed_labeled_saturation_certificate(
        addable_host,
        retained,
        rotation,
    )
    assert not addable["fixed_labeled_edge_saturated"]
    assert addable["cofacial_missing_host_edges"] == ((0, 4),)


def test_fixture_registry_matches_expected_results() -> None:
    rows = fixed_embedding_fixture_rows()
    assert len(rows) == 3
    assert all(row["matches_expected"] for row in rows)
    assert all(row["euler_characteristic"] == 2 for row in rows)
    assert not any(
        row["clifton_salia_plane_saturation_certified"]
        for row in rows
    )


def test_checker_requires_a_connected_spanning_subgraph() -> None:
    with pytest.raises(ValueError, match="connected"):
        fixed_labeled_saturation_certificate(
            ((0, 1), (2, 3)),
            ((0, 1), (2, 3)),
            {
                0: (1,),
                1: (0,),
                2: (3,),
                3: (2,),
            },
        )


def test_nonplanar_rotation_genus_is_rejected() -> None:
    # A one-face rotation of K3,3 has Euler characteristic zero.
    edges = tuple(
        (left, right)
        for left in range(3)
        for right in range(3, 6)
    )
    rotation = {
        0: (3, 4, 5),
        1: (3, 4, 5),
        2: (3, 4, 5),
        3: (0, 1, 2),
        4: (0, 1, 2),
        5: (0, 1, 2),
    }
    with pytest.raises(ValueError, match="sphere embedding"):
        face_boundaries(edges, rotation)
