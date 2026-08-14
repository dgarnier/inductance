"""Try to import numba."""

import os
from inspect import currentframe, getframeinfo
from warnings import warn, warn_explicit

# try to enable use without numba.  Those with guvectorize will not be
# vectorized: they stay callable, but need an explicit output array.


def _jit(*args, **kwargs):
    """Replace numba.jit/njit with a decorator that leaves the function alone."""
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        # called as @decorator
        return args[0]
    # called as @decorator(*args, **kwargs)
    return lambda f: f


def _guvectorize(*_args, **_kwargs):
    """Replace numba.guvectorize with a decorator that leaves the function alone.

    The function is returned unchanged, so it stays callable as plain Python,
    but it is not a generalized ufunc: it gets no broadcasting and the output
    array must be passed in explicitly.  Warn at the decoration site.
    """

    def fake_decorator(f):
        if not os.getenv("COVERAGE_RUN", ""):  # pragma: no cover
            warning = f"{f.__name__} requires Numba JIT."
            finfo = getframeinfo(currentframe().f_back)
            warn_explicit(warning, RuntimeWarning, finfo.filename, finfo.lineno)
        return f

    return fake_decorator


try:
    from numba import guvectorize, jit, njit, prange

    # if numba is disabled, redfine the jit decorator to do nothing
    if os.getenv("NUMBA_DISABLE_JIT", "0") == "1":  # pragma: no cover
        raise ImportError

except ImportError:  # pragma: no cover
    if not os.getenv("COVERAGE_RUN", ""):  # pragma: no cover
        _WARNING = "Numba acceleration disabled. Some API will not be available."
        warn(_WARNING, RuntimeWarning, stacklevel=2)

    guvectorize = _guvectorize
    njit = _jit
    prange = range
    jit = _jit


__all__ = ["guvectorize", "jit", "njit", "prange"]
