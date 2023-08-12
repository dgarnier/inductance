"""Test the mutual inductance of filaments and paths."""

import unittest

import coverage_env  # noqa: F401
from numpy import array, cos, dstack, linspace, ones_like, pi, sin

from inductance.filaments import (
    M_path_path,
    _loop_segmented_mutual,
    mutual_inductance_fil,
)


class TestMutual(unittest.TestCase):
    """Test and compare mutual inductance of filaments and paths."""

    def setUp(self) -> None:
        """Set up the test data."""
        self.coplanar_fil1 = array([1, 10, 1])
        self.coplanar_fil2 = array([100, 10, 1])
        t = linspace(0, 2 * pi, 10000)
        self.coplanar_fil1_path = dstack((-1 * sin(t), 1 * cos(t), 10 * ones_like(t)))[
            0
        ]
        self.coplanar_fil2_path = dstack(
            (-100 * sin(t), 100 * cos(t), 10 * ones_like(t))
        )[0]

        self.coaxial_fil1 = array([1, 0, 1])
        self.coaxial_fil2 = array([1, 100, 1])
        self.coaxial_fil1_path = dstack((-sin(t), cos(t), 0 * ones_like(t)))[0]
        self.coaxial_fil2_path = dstack((-sin(t), cos(t), 100 * ones_like(t)))[0]

        mu_0 = 1.256637062e-6  # H/m
        self.coplanar_analytic_sol = (
            mu_0 * pi * self.coplanar_fil1[0] ** 2 / (2 * self.coplanar_fil2[0])
        )
        self.coaxial_analytic_sol = (
            mu_0
            * pi
            * self.coaxial_fil1[0] ** 4
            / (2 * abs(self.coaxial_fil2[1] - self.coaxial_fil1[1]) ** 3)
        )
        return super().setUp()

    def test_mutual_fil_fil(self):
        """Test filaments.mutual_inductance_fil."""
        coplanar_filaments_sol = mutual_inductance_fil(
            self.coplanar_fil1, self.coplanar_fil2
        )
        coaxial_filaments_sol = mutual_inductance_fil(
            self.coaxial_fil1, self.coaxial_fil2
        )

        self.assertAlmostEqual(
            self.coplanar_analytic_sol, coplanar_filaments_sol, places=11
        )
        self.assertAlmostEqual(
            self.coaxial_analytic_sol, coaxial_filaments_sol, places=11
        )

    def test_mutual_fil_path(self):
        """Test filaments._loop_segmented_mutual."""
        coplanar_path_sol = _loop_segmented_mutual(
            *self.coplanar_fil1[:2], self.coplanar_fil2_path
        )
        coaxial_path_sol = _loop_segmented_mutual(
            *self.coaxial_fil1[:2], self.coaxial_fil2_path
        )

        self.assertAlmostEqual(coplanar_path_sol, self.coplanar_analytic_sol, places=11)
        self.assertAlmostEqual(coaxial_path_sol, self.coaxial_analytic_sol, places=11)

    def test_mutual_path_path(self):
        """Test filaments.M_path_path."""
        coplanar_path_sol = M_path_path(
            self.coplanar_fil1_path, self.coplanar_fil2_path
        )
        coaxial_path_sol = M_path_path(self.coaxial_fil1_path, self.coaxial_fil2_path)

        self.assertAlmostEqual(coplanar_path_sol, self.coplanar_analytic_sol, places=10)
        self.assertAlmostEqual(coaxial_path_sol, self.coaxial_analytic_sol, places=10)


if __name__ == "__main__":
    unittest.main()
