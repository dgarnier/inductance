"""Test the decorators used when numba is missing or disabled.

These stand in for numba.jit and numba.guvectorize.  They must leave the
decorated function alone -- code that ran with numba should still run, just
without acceleration.
"""

import os
import unittest
import warnings
from unittest import mock

import coverage_env  # noqa: F401
import numpy as np
import pytest

from inductance import _numba
from inductance._numba import _guvectorize, _jit

GUVECTORIZE_ARGS = (["void(float64[:], float64[:])"], "(n)->(n)")


def double(a, out):
    """Kernel in guvectorize form: write 2*a into the output array."""
    for i in range(a.shape[0]):
        out[i] = 2 * a[i]


class TestJitFallback(unittest.TestCase):
    """Test the numba.jit / numba.njit stand-in."""

    def test_bare_decorator_returns_function(self):
        """Used as @njit, the function itself comes back."""
        assert _jit(double) is double

    def test_decorator_with_options_returns_function(self):
        """Used as @njit(parallel=True), the function itself comes back."""
        assert _jit(parallel=True)(double) is double
        assert _jit("float64(float64)", nogil=True)(double) is double

    def test_decorator_with_only_signature(self):
        """A signature-only call is still a decorator factory."""
        assert _jit("float64(float64)")(double) is double


class TestGuvectorizeFallback(unittest.TestCase):
    """Test the numba.guvectorize stand-in."""

    def test_decorated_function_still_works(self):
        """The kernel is returned unchanged, so it stays callable."""
        with mock.patch.dict(os.environ, {"COVERAGE_RUN": "1"}):
            decorated = _guvectorize(*GUVECTORIZE_ARGS)(double)
        assert decorated is double

        out = np.zeros(3)
        decorated(np.array([1.0, 2.0, 3.0]), out)
        assert out == pytest.approx([2.0, 4.0, 6.0])

    def test_warns_that_numba_is_needed(self):
        """Decorating warns, at the decoration site, that numba is needed."""
        with (
            mock.patch.dict(os.environ, {"COVERAGE_RUN": ""}),
            pytest.warns(RuntimeWarning, match="double requires Numba JIT"),
        ):
            _guvectorize(*GUVECTORIZE_ARGS)(double)

    def test_no_warning_under_coverage(self):
        """The warning is suppressed while measuring coverage."""
        with (
            mock.patch.dict(os.environ, {"COVERAGE_RUN": "1"}),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("error")
            _guvectorize(*GUVECTORIZE_ARGS)(double)


class TestNumbaExports(unittest.TestCase):
    """Test what the module exports, with or without numba."""

    def test_all_names_are_exported(self):
        """Every name in __all__ is bound, either to numba's or to a stub."""
        for name in _numba.__all__:
            assert getattr(_numba, name, None) is not None

    def test_jit_decorators_are_usable(self):
        """njit and jit decorate a function without raising."""
        assert callable(_numba.njit(double))
        assert callable(_numba.jit(nopython=True)(double))


if __name__ == "__main__":
    unittest.main()
