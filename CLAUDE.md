# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this tool does

`snapshot-tool` is a snapshot-testing harness for [ASV](https://github.com/airspeed-velocity/asv) benchmarks. Given a benchmark directory it:

1. Discovers ASV benchmarks (functions/methods prefixed `time_`, `timeraw_`, `mem_`, `peakmem_`, `track_`) by AST-parsing files — it never imports them at discovery time.
2. Runs each benchmark while a `sys.settrace`-based tracer captures the return value of the shallowest meaningful user-code call (stdlib + numpy/shapely internals are skipped).
3. Persists the captured return value into `<.snapshots>/snapshots.db` — a single SQLite database with a content-addressed `blobs` table (sha256 → gzipped pickle, refcounted) and a `snapshots` table. A JSON metadata sidecar is also written per `(module, class, benchmark, parameters)` tuple for downstream tooling.
4. Reruns later and compares outputs with tolerance-aware numerical comparison (pure-Python `isclose`; numpy is optional).

The roundtrip executed by `customtest.sh` is the canonical demo: `list -> capture -> baseline -> verify`.

## Common commands

This project uses `uv`. Always invoke tools through `uv run` (or after `uv sync --group dev && uv pip install -e .`).

```bash
# Install dev environment
uv sync --group dev && uv pip install -e .

# Run the full test suite (excludes the heavy benchmark-repo roundtrips, matching CI)
uv run pytest -v --ignore=tests/test_repos/ --ignore=tests/test_cli_roundtrip.py

# Run a single test / file
uv run pytest tests/test_comparator_comprehensive.py -v
uv run pytest tests/test_comparator_comprehensive.py::TestComparator::test_name -v

# Run the slow real-repo roundtrip tests locally
uv run pytest tests/test_cli_roundtrip.py -v
# Filter inside the roundtrip via env vars (the same knobs CI uses):
SNAPSHOT_TOOL_FILTER='^benchmarks\.(coordinates|units)' SNAPSHOT_TOOL_TIMEOUT=10 \
  uv run pytest tests/test_cli_roundtrip.py::TestAstropyRoundtrip -x

# Lint / format (matches the lint.yml CI job — no auto-fix in CI)
uv run ruff format --check src/ tests/
uv run ruff check src/ tests/

# Manual smoke test against one of the bundled repos
bash customtest.sh   # currently wired to tests/test_repos/shapely_benchmarks
```

## CLI surface

Entry point: `snapshot-tool` (defined in `pyproject.toml` -> `snapshot_tool.cli:main`). Subcommands:

- `list <benchmark_dir> [--filter REGEX]`
- `capture <benchmark_dir> [--filter REGEX] [--snapshot-dir DIR] [--timeout SEC]`
- `verify <benchmark_dir> [--filter REGEX] [--snapshot-dir DIR] [--tolerance RTOL ATOL] [--summary summary.json] [--timeout SEC]`
- `baseline <benchmark_dir> ...` — like verify but records pass/fail per test_id into `<snapshot_dir>/baseline.json`. A subsequent `verify` reads it and emits a 3x3 pass/fail/skip transition matrix via `transitions.compute_transitions`.
- `clean`, `config --init|--show`

`--filter` is a Python regex matched against `f"{module_path}.{benchmark_name}"`.

## Architecture

The package lives under `src/snapshot_tool/`. The pipeline is intentionally a chain of single-purpose modules; understand them in this order:

1. **`discovery.py`** — `BenchmarkDiscovery` walks the benchmark directory and AST-parses every `.py` file (skipping `__init__.py`). It returns `BenchmarkInfo` records describing each function- or class-method benchmark, including `params`, `param_names`, whether the class has `setup` / `setup_cache`, and `needs_runtime_eval` (set when `params` is something dynamic that can't be evaluated statically — e.g. a comprehension or call). Parameter evaluation for these cases is deferred to the runner.

2. **`tracer.py`** — `ExecutionTracer` installs a `sys.settrace` callback that filters out frames belonging to stdlib (`sys.stdlib_module_names` + a hand-curated set), numpy internals, shapely internals, dunder methods, and lambdas. It captures the *shallowest* non-meaningless return value — comments in `_handle_return` describe this as "deepest" but the code prefers shallower depths (`<= self.deepest_call.depth`). The trace result is what gets snapshotted, not the benchmark function's own return value.

3. **`rng_patcher.py`** — `RNGPatcher` deterministically reseeds numpy (legacy + Generator API), PyTorch, and TensorFlow before every benchmark run. `BenchmarkRunner._reset_random_state` calls this on every invocation — do not bypass it, the whole point of snapshot comparison is bit-stable output.

4. **`runner.py`** — `BenchmarkRunner` ties the above together. Key behaviors:
   - Loads benchmark modules via `importlib.util` and synthesizes parent-package `ModuleType` entries in `sys.modules` so files using relative imports (`from .common import setup`) work.
   - Caches loaded modules and class-level `setup_cache()` results.
   - Wraps each run in a `ThreadPoolExecutor.submit(...).result(timeout=...)` so per-benchmark timeouts are enforceable (default 300 s from the CLI). On timeout the future is cancelled but the thread keeps running — be aware this can leak threads on hangs.
   - For class-based benchmarks calls `setup_cache` (once), then `setup(*params)` before the benchmark method. For methods declared with parameters but invoked without, it falls back to the first parameter combination and logs a warning.

5. **`storage.py`** — `SnapshotManager` keeps all captured values in a single SQLite database at `<snapshot_dir>/snapshots.db`. The schema has two tables joined by `blob_hash`: `blobs(hash, data, refcount, raw_size, compressed_size)` content-addresses each gzipped pickle by sha256 (so benchmarks producing identical outputs share a single blob), and `snapshots(test_id PRIMARY KEY, ...)` carries per-test metadata. PRAGMAs: `journal_mode=WAL`, `synchronous=NORMAL`, `foreign_keys=ON`. A JSON metadata sidecar is still written per snapshot at `<snapshot_dir>/<module>/<class.benchmark>/<param-hash>.json` for downstream tooling — `store_snapshot` returns that path. `SnapshotManager` also owns `baseline.json` (kept as a flat JSON file) — the test_id->status map produced by `baseline` and consumed by `verify`. There is no `.pkl` / `.pkl.gz` read path; the SQLite backend is a clean break.

6. **`comparator.py`** — `Comparator` performs tolerance-aware comparison. Numpy is imported lazily and detected via module/type-name probing (`_is_numpy_array`), so the package itself does not depend on numpy. Pure-Python `_py_isclose` mirrors `numpy.isclose` semantics for scalars. **`ComparisonConfig.equal_nan` defaults to `True`** (and so does `SnapshotConfig.tolerance["equal_nan"]`): for snapshot regression testing a deterministic NaN that reappears unchanged is not a change. Pass `equal_nan=False` for strict `numpy.isclose` semantics.

7. **`transitions.py`** — pure function `compute_transitions(baseline, verify)` returning the 9-cell pass/fail/skip transition matrix. Legacy status `"failed_to_pass"` is normalized to `"fail"`.

8. **`cli.py`** — argparse front end; thin glue over the modules above.

Public API is re-exported in `src/snapshot_tool/__init__.py`; prefer adding to `__all__` there over import-from-submodule patterns elsewhere.

## Project conventions to respect

- **Python 3.8+ compatibility is enforced.** `pyproject.toml` pins `target-version = "py38"` and ignores `UP007` (no `X | Y` unions). Every module uses `from __future__ import annotations`; keep this for any new file. The full test matrix in CI covers 3.8 -> 3.13.
- Ruff lints with `E,W,F,I,B,C4,UP` and ignores `E501`; formatter is the source of truth (CI runs `ruff format --check`).
- `tests/test_repos/` is excluded by `norecursedirs` in `pyproject.toml` — those directories are vendored benchmark sources (astropy, pandas, shapely), not test files. They are exercised only through `tests/test_cli_roundtrip.py`, which CI runs in dedicated `test-astropy.yml` / `test-pandas.yml` / `test-shapely.yml` jobs sharded by benchmark module regex.
- `tests/test_cli_roundtrip.py` is a **regression** gate, not a perfection gate. It runs `list -> capture -> baseline -> verify` and asserts the baseline→verify transition matrix has **no `pass-to-fail` / `skip-to-fail`** (read from `verify`'s `--summary` JSON). Real third-party suites contain inherently un-snapshotable benchmarks (memory addresses in reprs, timing-sensitive, dtype-unstable); those stay `fail-to-fail` and are tolerated. Don't reintroduce a "zero failures" assertion.
- `pytest -v --strict-markers` is configured; the `slow` marker is registered for full-repo roundtrips. Don't introduce new markers without registering them.
- The package's logger is configured at import time via `configure_logging()` in `__init__.py`. Use `logging.getLogger(__name__)` in submodules — don't add new root-level handlers.
- Snapshot files are written under `.snapshots/` (or wherever `--snapshot-dir` points). `customtest.sh` blows that directory away before each run; treat it as disposable build output, not source.
