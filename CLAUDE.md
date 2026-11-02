# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this tool does

`snapshot-tool` is a snapshot-testing harness for [ASV](https://github.com/airspeed-velocity/asv) benchmarks. Given a benchmark directory it:

1. Discovers ASV benchmarks (functions/methods prefixed `time_`, `timeraw_`, `mem_`, `peakmem_`, `track_`) by AST-parsing files — it never imports them at discovery time.
2. Runs each benchmark and captures a return value. The runner first rewrites a trailing expression statement in the benchmark into a `return`; a `sys.settrace` tracer then keeps the shallowest meaningful user-code return value (stdlib and numpy/shapely internals are skipped).
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
bash customtest.sh   # runs against tests/test_repos/shapely_benchmarks
```

## CLI surface

Entry point: `snapshot-tool` (defined in `pyproject.toml` -> `snapshot_tool.cli:main`). Subcommands:

- `list <benchmark_dir> [--filter REGEX]`
- `capture <benchmark_dir> [--filter REGEX] [--snapshot-dir DIR] [--timeout SEC]`
- `verify <benchmark_dir> [--filter REGEX] [--snapshot-dir DIR] [--tolerance RTOL ATOL] [--summary summary.json] [--timeout SEC]`
- `baseline <benchmark_dir> ...` — like verify but records pass/fail per test_id into `<snapshot_dir>/baseline.json`. A subsequent `verify` reads it and emits a 3x3 pass/fail/skip transition matrix via `transitions.compute_transitions`.
- `clean` (prints snapshot statistics; deletes nothing), `config --init|--show`

The global `--config PATH` flag is parsed but not used: the CLI always reads `./snapshot_config.json`.

`--filter` is a Python regex matched against `f"{module_path}.{benchmark_name}"`.

## Architecture

The package lives under `src/snapshot_tool/`. The pipeline is intentionally a chain of single-purpose modules; understand them in this order:

1. **`discovery.py`** — `BenchmarkDiscovery` walks the benchmark directory and AST-parses every `.py` file (skipping `__init__.py`). It returns `BenchmarkInfo` records describing each function- or class-method benchmark, including `params`, `param_names`, whether the class has `setup` / `setup_cache`, and `needs_runtime_eval`. `needs_runtime_eval` is set when `params` is a list or tuple literal that holds a non-literal element (a name or a call); the runner then reads `params` from the imported class. A `params` value that is itself a comprehension or a call is not detected, and the benchmark runs without parameters.

2. **`tracer.py`** — `ExecutionTracer` installs a `sys.settrace` callback that filters out frames belonging to stdlib (`sys.stdlib_module_names` + a hand-curated set), numpy internals, shapely internals, dunder methods, and lambdas. It keeps the *shallowest* meaningful return value (`<= self.deepest_call.depth`), although the attribute is named `deepest_call`. Because the runner rewrites a trailing expression into a `return`, the benchmark's own return value is usually what gets snapshotted.

3. **`rng_patcher.py`** — `RNGPatcher` reseeds Python `random`, the numpy legacy API (`np.random.seed`), PyTorch, and TensorFlow before every benchmark run. It does not seed `numpy.random.Generator` instances. `BenchmarkRunner._reset_random_state` calls this on every invocation — do not bypass it, the whole point of snapshot comparison is bit-stable output.

4. **`runner.py`** — `BenchmarkRunner` ties the above together. Key behaviors:
   - Loads benchmark modules via `importlib.util` and synthesizes parent-package `ModuleType` entries in `sys.modules` so files using relative imports (`from .common import setup`) work.
   - Caches loaded modules and class-level `setup_cache()` results.
   - Enforces the per-benchmark timeout (default 300 s from the CLI). On the main thread a `SIGALRM` timer raises `BenchmarkTimeout` inside the benchmark. Off the main thread it waits on a `ThreadPoolExecutor` future, and a benchmark that times out keeps running in its thread.
   - For class-based benchmarks calls `setup_cache` (once), then `setup(*params)` before the benchmark method. For methods declared with parameters but invoked without, it falls back to the first parameter combination and logs a warning.
   - Optional opt-in parallelism: `capture`/`verify`/`baseline` accept `--parallel` (and `--workers N`, default `min(cpu_count, 8)`). `runner.iter_task_results` dispatches `(benchmark, params)` tasks to a `ProcessPoolExecutor` (a per-worker `BenchmarkRunner` is built once via the pool initializer so its module/setup_cache caches are reused). Workers only run, trace and `serialize_value` the result; the **main process does every SQLite write** (SQLite has one writer) and all comparison. The default is serial. Tasks whose `(benchmark, params)` cannot be pickled run in the main process. If the pool breaks (for example, a worker is killed for memory), the remaining tasks run serially in the main process.

5. **`storage.py`** — `SnapshotManager` keeps all captured values in a single SQLite database at `<snapshot_dir>/snapshots.db`. The schema has two tables joined by `blob_hash`: `blobs(hash, data, refcount, raw_size, compressed_size)` content-addresses each gzipped pickle by sha256 (so benchmarks producing identical outputs share a single blob), and `snapshots(test_id PRIMARY KEY, ...)` carries per-test metadata. PRAGMAs: `journal_mode=WAL`, `synchronous=NORMAL`, `foreign_keys=ON`. A JSON metadata sidecar is also written per snapshot at `<snapshot_dir>/<module>/<class.benchmark>/<param-hash>.json` for downstream tooling — `store_snapshot` returns that path. `SnapshotManager` also reads and writes `baseline.json`, a flat JSON file with the test_id->status map that `baseline` writes and `verify` reads. Only `snapshots.db` is read; `.pkl` / `.pkl.gz` files are ignored.

6. **`comparator.py`** — `Comparator` performs tolerance-aware comparison. Numpy is imported lazily and detected via module/type-name probing (`_is_numpy_array`), so the package itself does not depend on numpy. Pure-Python `_py_isclose` mirrors `numpy.isclose` semantics for scalars. **`ComparisonConfig.equal_nan` defaults to `True`** (and so does `SnapshotConfig.tolerance["equal_nan"]`): for snapshot regression testing a deterministic NaN that reappears unchanged is not a change. Pass `equal_nan=False` for strict `numpy.isclose` semantics.

7. **`transitions.py`** — pure function `compute_transitions(baseline, verify)` returning `<a>-to-<b>` counts for the status pairs present in the data (`verify` logs all nine cells). The status `"failed_to_pass"` is read as `"fail"`.

8. **`cli.py`** — argparse front end; thin glue over the modules above.

Public API is re-exported in `src/snapshot_tool/__init__.py`; prefer adding to `__all__` there over import-from-submodule patterns elsewhere.

## Project conventions to respect

- **Python 3.8+ compatibility is enforced.** `pyproject.toml` pins `target-version = "py38"` and ignores `UP007` (no `X | Y` unions). Every module uses `from __future__ import annotations`; keep this for any new file. The full test matrix in CI covers 3.8 -> 3.13.
- Ruff lints with `E,W,F,I,B,C4,UP` and ignores `E501`; formatter is the source of truth (CI runs `ruff format --check`).
- `tests/test_repos/` is excluded by `norecursedirs` in `pyproject.toml` — those directories are vendored benchmark sources (astropy, pandas, shapely), not test files. They are exercised only through `tests/test_cli_roundtrip.py`, which CI runs in dedicated `test-astropy.yml` / `test-pandas.yml` / `test-shapely.yml` jobs sharded by benchmark module regex.
- `tests/test_cli_roundtrip.py` is a **regression** gate, not a perfection gate. It runs `list -> capture -> baseline -> verify` and asserts the baseline→verify transition matrix has **no `pass-to-fail` / `skip-to-fail`** (read from `verify`'s `--summary` JSON). Real third-party suites contain inherently un-snapshotable benchmarks (memory addresses in reprs, timing-sensitive, dtype-unstable); those stay `fail-to-fail` and are tolerated. Do not add a "zero failures" assertion.
- `pytest -v --strict-markers` is configured; the `slow` marker is registered for full-repo roundtrips. Don't introduce new markers without registering them.
- The package's logger is configured at import time via `configure_logging()` in `__init__.py` at INFO level, so `logger.debug` lines are not shown, even with `--verbose`. Use `logging.getLogger(__name__)` in submodules; don't add new root-level handlers.
- Snapshot files are written under `.snapshots/` (or wherever `--snapshot-dir` points). `customtest.sh` blows that directory away before each run; treat it as disposable build output, not source.
