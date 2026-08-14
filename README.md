# Inductance

[![PyPI](https://img.shields.io/pypi/v/inductance.svg)][pypi]
[![Status](https://img.shields.io/pypi/status/inductance.svg)][pypi]
[![Python Version](https://img.shields.io/pypi/pyversions/inductance)][pypi]
[![License](https://img.shields.io/pypi/l/inductance)][license]
[![Read the documentation at https://inductance.readthedocs.io/](https://img.shields.io/readthedocs/inductance/latest.svg?label=Read%20the%20Docs)][read the docs]
[![Tests](https://github.com/dgarnier/inductance/workflows/Tests/badge.svg)][tests]
[![Codecov](https://codecov.io/gh/dgarnier/inductance/branch/main/graph/badge.svg)][codecov]
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)][Ruff]

[pypi]: https://pypi.org/project/inductance/
[read the docs]: https://inductance.readthedocs.io/
[tests]: https://github.com/dgarnier/inductance/actions?workflow=Tests
[codecov]: https://app.codecov.io/gh/dgarnier/inductance
[pre-commit]: https://github.com/pre-commit/pre-commit
[Ruff]: https://github.com/astral-sh/ruff

This is a python library to calculate inductance. Mostly for the purposes of calcualting magnetically confined plasmas. It might someday actually contain some plasma physics, but lets not get too carried away.

## Features

- Self-inductance formulas

  - self inductance of circular, circular hollow, and rectangular section by Maxwell's approximation
  - Lyle's approximation for thick coil solenoid self inductances to 4th and 6th order.
  - Butterworth's approximation for long solenoids
  - Lorentz's perfect analytic solution for current sheet solenoids
  - Babic and Akyel's approximation for thin solenoids

- Mutual-inductance formulas

  - mutual inductance of filaments (Maxwell)

- Filamentary models

  - utility functions to create filament arrays from rectangular definitions of coils and subcoils
  - calculation for filament array mutual inductance
  - calculation of filament array self inductance

- Green's functions

  - calculation of Green's functions for Psi, Br, and Bz from filamented coils to points
    - with Numba, calculation of green's functions for arbitrary grids of points
  - calculation fo Green's functions for coil forces

- Arbitrary coil shapes
  - rudimentary support for arbitrary wire filament coil shapes

## Requirements

_Inductance_ requires [_NumPy_][numpy] and uses [_Numba_][numba] for acceleration. It is written in mostly pure python referencing academic articles for calculating inductances by various methods, most of which rely on elliptic functions. _Inductance_ provides _Numba_ accelerated pure python elliptic functions.

_Numba_ is optional, though it is installed by default. Without it, _Inductance_ falls back to pure python: everything still works, just slower, apart from the vectorized Green's function helpers (`BrGreenFil`, `BzGreenFil`, `AGreenFil`), which stay callable but are no longer generalized ufuncs, so they need an explicit output array. The plan is to provide different options, including with alternative accelerators, such as [_JAX_][jax]. The requirements are:

- python >= 3.11
- numpy >= 2.3
- numba >= 0.63 — optional, installed by default on any python _Numba_ supports

[numba]: https://numba.readthedocs.io/
[numpy]: https://numpy.org
[jax]: https://jax.readthedocs.io/

## Installation

You can install _inductance_ via [pip] from [PyPI]:

```console
$ pip install inductance
```

That installs _Numba_ too. On a python newer than _Numba_ supports, it is
skipped automatically so the install still succeeds; once _Numba_ catches up,
ask for it explicitly with:

```console
$ pip install "inductance[numba]"
```

To install without _Numba_ on a python that does support it:

```console
$ pip install --no-deps inductance numpy
```

If _Numba_ is installed but you want to bypass compilation — when debugging, or
under coverage — set `NUMBA_DISABLE_JIT=1` in the environment instead.

## Reference

Please see the [reference] for details.

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [MIT license][license],
_inductance_ is free and open source software.

## Issues

For now, this is a very early release. It is likely a new top level API will be
developed as the library matures.

If you encounter any problems,
please [file an issue] along with a detailed description.

## Credits

This project was generated from [@cjolowicz]'s [Hypermodern Python Cookiecutter] template.

[@cjolowicz]: https://github.com/cjolowicz
[hypermodern python cookiecutter]: https://github.com/cjolowicz/cookiecutter-hypermodern-python
[file an issue]: https://github.com/dgarnier/inductance/issues
[pip]: https://pip.pypa.io/

<!-- github-only -->

[license]: https://github.com/dgarnier/inductance/blob/main/LICENSE
[contributor guide]: https://github.com/dgarnier/inductance/blob/main/CONTRIBUTING.md
[reference]: https://inductance.readthedocs.io/en/latest/reference.html
