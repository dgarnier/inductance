# AGENTS.md

Notes for AI agents working in this repository. `inductance` is a small,
numerics-heavy library: self and mutual inductance of coils, filament models,
and elliptic integrals, transcribed from academic papers. The physics matters
more than the style guide — read this before "cleaning up" anything.

## Commands

```console
$ .venv/bin/python -m pytest -q          # tests (or: uv run pytest)
$ NUMBA_DISABLE_JIT=1 .venv/bin/python -m pytest -q   # same, no JIT
$ uvx ruff check . && uvx ruff format .  # ruff is NOT on PATH; use uvx
$ .venv/bin/pre-commit run --all-files   # all hooks
$ .venv/bin/nox -l                       # sessions; -s tests, -s docs-build, ...
$ .venv/bin/mypy src tests               # stubs/ holds the numba stub
```

The local `.venv` is a free-threaded 3.14 build. CI drives everything through
`uv run nox`.

## Numba is the thing to understand first

`src/inductance/_numba.py` decides at import time whether `njit`/`guvectorize`
are real or stand-ins. There are **three** distinct modes, and a change can
break one while passing in another:

1. numba installed and active — the normal case
2. numba installed, `NUMBA_DISABLE_JIT=1` — falls back to `_jit`/`_guvectorize`
3. numba **absent** — same fallbacks, plus `RuntimeWarning`s at import

Mode 3 is real: numba lags new Python releases, so `pyproject.toml` carries
`numba>=0.63.0; python_version < '3.15'`. CI covers it in the 3.15 job.

Rules for that layer:

- The fallbacks **warn but keep working**. `_guvectorize` returns the function
  unchanged; the decorated kernel stays callable, it just isn't a generalized
  ufunc any more (no broadcasting, caller passes the output array). Do not
  "improve" this into raising or into a stub that returns `None`.
- Anything under `@njit` must stay numba-compilable. Prefer plain loops and
  explicit tuple unpacking over comprehensions, starred unpacking, or f-strings.
- Run the suite in at least modes 1 and 2 before claiming a change works.

## Conventions in the numerical code

- **Names come from the literature, not PEP 8.** `L`, `M0`, `GMD`, `MU0`,
  `dLdR_lyle6`, `L_lyle6_appendix` are correct as written. Do not lowercase or
  "clarify" them.
- **If a lint rule makes you delete a meaningful name, leave a comment naming
  the quantity.** Ruff's `RET504` wants `eq3 = ...; return eq3` collapsed; when
  you collapse it, add `# equation #3` (or `# M0, the mutual inductance ...`)
  so the formula still says what it is. This is explicit maintainer preference.
- `src/inductance/elliptics.py` is a transcription of Fukushima's algorithm and
  is wrapped in `# fmt: off`. Leave its coefficient tables and layout alone.
- Docstrings are Google-style `Args:`/`Returns:` (ruff's pydocstyle convention
  is set to numpy, but the source uses Google sections; follow the file).

## Tests

- `unittest.TestCase` classes, run under pytest. Assertions are plain `assert`
  with `pytest.approx` — `assertAlmostEqual` is banned by ruff `PT009`.
- Tolerance convention: the old `places=N` maps to `abs=0.5e-N`. Passing only
  `abs` to `pytest.approx` disables the relative tolerance, which is what makes
  it equivalent to `assertAlmostEqual`.
- Ruff `SIM300` treats ALL-CAPS locals as constants, so `assert LFC == approx(...)`
  is flagged as a Yoda condition. Use mixed case for compared locals
  (`L_fcoil`, `M_fc_lc`, `Fz_kg`) rather than reversing the comparison.
- **The hard-coded expected values are physical benchmarks** (LDX coil
  inductances, Lyle's published tables, analytic limits). If one fails, suspect
  the change, not the number.
- `tests/coverage_env.py` is imported for its side effect: coverage.py sets
  `COVERAGE_RUN=true`, which makes it set `NUMBA_DISABLE_JIT=1` so coverage can
  see inside jitted functions. That is why `# noqa: F401` sits on that import.

## Packaging and versions

- Runtime dependencies are **numpy only**, plus numba under the marker above.
  Nothing else belongs in `[project] dependencies` — dev tooling goes in
  `[dependency-groups]`.
- There is a `numba` extra for opting back in on a Python past the marker. It
  pins numba unconditionally, so **never use `--all-extras`** in CI or scripts:
  it cannot resolve where numba has no wheels.
- After any dependency edit, refresh the lock: `uv lock` (`uv lock --check`
  tells you if it is stale).
- The version is static in `[project] version` and `[tool.bumpversion]
  current_version`; bump-my-version keeps both in sync. **Do not hand-edit
  either in a feature PR.**
- `commit = false` and `tag = false` are deliberate. The *Prepare release*
  workflow bumps and opens a PR (the PR action makes the commit); `release.yml`
  is the only thing that creates tags. Do not add tagging anywhere else.

## Docs

- Docs dependencies live in `[dependency-groups] docs`. Read the Docs installs
  them via `build.jobs.pre_install: python -m pip install --group docs`. Do not
  reintroduce `docs/requirements.txt` — a stale reference to a deleted copy of
  it kept the docs build red while the published site quietly served stale HTML.
- `docs/index.md` includes `README.md` up to the `<!-- github-only -->` marker,
  so duplicate link definitions in the README become Sphinx warnings.
- RTD's `tools.python` must satisfy `requires-python`, and RTD shallow-clones
  (relevant if versioning ever moves to git tags).

## Lint miscellany

- codespell's `# codespell:ignore word` must sit on the **same physical line**
  as the word, so a line ending in `\` has nowhere to put it. Often the
  backslash is redundant because the expression is already inside brackets —
  dropping it makes room (see `elliptics.py:84`, for the elliptic *nome*). Use
  `ignore-words-list` in `pyproject.toml` for terms that recur repo-wide.
- The codespell and ruff hooks are `types: [python]` / `types_or: [python]`, so
  markdown, YAML and notebooks are not checked by either.
- `tests/triangle_gmd.ipynb` and `fitting_data.npy` are untracked scratch work.
  Leave them alone.

## Working style

- Verify empirically instead of asserting. Scratch clones and throwaway venvs
  (`uv venv /tmp/... --python 3.x`) are cheap; use them to prove a packaging or
  CI claim before writing it down.
- Stay inside the request. This code is full of things that look like mistakes
  and are not.
