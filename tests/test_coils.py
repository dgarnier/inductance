"""Test the Coil conductor dispatch in Coil.L_filament."""

import unittest
from enum import Enum

import coverage_env  # noqa: F401
import pytest

from inductance.coils import Coil, Conductor, Shape
from inductance.self import self_inductance_by_filaments


class UnknownShape(Enum):
    """A conductor shape the inductance routines don't know about."""

    hexagon = "hexagon"


def make_coil():
    """Make a small filamented coil to test the L_filament branches."""
    return Coil(0.5, 0.0, 0.05, 0.04, nt=10, nr=3, nz=2)


class TestConductor(unittest.TestCase):
    """Test the Conductor dataclass."""

    def test_shape_from_string(self):
        """A string shape is converted to the Shape enum."""
        assert Conductor(shape="round").shape is Shape.round
        assert Conductor(shape="hollow_round").shape is Shape.hollow_round
        assert Conductor(shape="rectangle").shape is Shape.rect

    def test_unknown_shape_string(self):
        """An unknown shape string is rejected by the enum."""
        with pytest.raises(ValueError, match="hexagon"):
            Conductor(shape="hexagon")


class TestLFilament(unittest.TestCase):
    """Test that L_filament dispatches on the conductor shape."""

    def test_no_conductor_uses_filament_sections(self):
        """Without a conductor, each filament is a section of the coil."""
        coil = make_coil()
        expected = self_inductance_by_filaments(
            coil.fils,
            conductor="rect",
            dr=coil.dr / coil.nr,
            dz=coil.dz / coil.nz,
        )
        assert coil.L_filament() == pytest.approx(expected, rel=1e-12)

    def test_round_conductor(self):
        """A round conductor uses its radius, not the section size."""
        coil = make_coil()
        coil.conductor = Conductor(shape="round", r=0.004)
        expected = self_inductance_by_filaments(coil.fils, conductor="round", a=0.004)
        assert coil.L_filament() == pytest.approx(expected, rel=1e-12)

    def test_hollow_round_conductor(self):
        """A hollow round conductor carries skin current, so L is lower."""
        coil = make_coil()
        coil.conductor = Conductor(shape="hollow_round", r=0.004)
        expected = self_inductance_by_filaments(
            coil.fils, conductor="hollow_round", a=0.004
        )
        assert coil.L_filament() == pytest.approx(expected, rel=1e-12)

        solid = make_coil()
        solid.conductor = Conductor(shape="round", r=0.004)
        assert coil.L_filament() < solid.L_filament()

    def test_rect_conductor(self):
        """A rectangular conductor uses its own width and height."""
        coil = make_coil()
        coil.conductor = Conductor(shape="rectangle", dr=0.01, dz=0.015)
        expected = self_inductance_by_filaments(
            coil.fils, conductor="rect", dr=0.01, dz=0.015
        )
        assert coil.L_filament() == pytest.approx(expected, rel=1e-12)

    def test_unsupported_conductor_shape_raises(self):
        """An unsupported shape raises instead of silently returning None."""
        coil = make_coil()
        coil.conductor = Conductor(shape=UnknownShape.hexagon, r=0.004)
        with pytest.raises(ValueError, match="Unsupported conductor shape"):
            coil.L_filament()

    def test_refilamentizes_when_given_sections(self):
        """Passing nr and nz re-filaments the coil before integrating."""
        coil = make_coil()
        coil.L_filament(4, 5)
        assert (coil.nr, coil.nz) == (4, 5)
        assert coil.fils.shape[0] == 20


if __name__ == "__main__":
    unittest.main()
