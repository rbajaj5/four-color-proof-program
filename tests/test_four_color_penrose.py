from __future__ import annotations

import unittest

from scripts.check_four_color_penrose_fixtures import evaluate_rows
from src.four_color_penrose import (
    CubicRotationSystem,
    evaluate_penrose_tait,
    has_bridge,
    levi_civita,
)


class FourColorPenroseTests(unittest.TestCase):
    def test_levi_civita(self) -> None:
        self.assertEqual(levi_civita((0, 1, 2)), 1)
        self.assertEqual(levi_civita((1, 0, 2)), -1)
        self.assertEqual(levi_civita((0, 0, 2)), 0)

    def test_theta_calibration(self) -> None:
        system = CubicRotationSystem(
            edges=((0, 1), (0, 1), (0, 1)),
            rotations={0: (0, 1, 2), 1: (0, 2, 1)},
        )
        result = evaluate_penrose_tait(system)
        self.assertEqual(result.tait_coloring_count, 6)
        self.assertEqual(result.penrose_contraction, 6)
        self.assertFalse(has_bridge(system.edges))

    def test_all_plane_fixture_identities(self) -> None:
        rows = evaluate_rows()
        for row in rows:
            if row["plane_embedding_declared"]:
                self.assertTrue(
                    row["plane_penrose_identity_passed"],
                    row["fixture"],
                )

    def test_expected_fixture_counts(self) -> None:
        rows = evaluate_rows()
        expected = {
            "theta_plane_multigraph": 6,
            "k4_tetrahedral": 6,
            "k33_nonplanar_control": 12,
            "petersen_snark_control": 0,
            "planar_bridge_control": 0,
        }
        for row in rows:
            if row["fixture"] in expected:
                self.assertEqual(
                    row["tait_coloring_count"],
                    expected[row["fixture"]],
                )

    def test_bridge_control_detected(self) -> None:
        row = next(
            row for row in evaluate_rows()
            if row["fixture"] == "planar_bridge_control"
        )
        self.assertTrue(row["has_bridge"])
        self.assertEqual(row["tait_coloring_count"], 0)

    def test_naive_penrose_contraction_has_a_planarity_boundary(self) -> None:
        row = next(
            row for row in evaluate_rows()
            if row["fixture"] == "k33_nonplanar_control"
        )
        self.assertEqual(row["tait_coloring_count"], 12)
        self.assertEqual(row["penrose_contraction"], 0)
        self.assertNotEqual(
            row["penrose_contraction"],
            row["tait_coloring_count"],
        )

    def test_invalid_rotation_rejected(self) -> None:
        system = CubicRotationSystem(
            edges=((0, 1), (0, 1), (0, 1)),
            rotations={0: (0, 1, 1), 1: (0, 2, 1)},
        )
        with self.assertRaises(ValueError):
            evaluate_penrose_tait(system)


if __name__ == "__main__":
    unittest.main()
