# CLI (`snapshot-tool`)

`snapshot-tool` is the primary entrypoint. It discovers ASV benchmarks in a directory, captures their return values, and compares them on later runs. All operations are local — no network or database is involved.

## Quick reference

```bash
# 1. See what discovery finds
snapshot-tool list path/to/benchmarks

# 2. Capture a baseline snapshot of every benchmark's output
snapshot-tool capture path/to/benchmarks --timeout 60

# 3. (Optional) Record current pass/fail state for transition tracking
snapshot-tool baseline path/to/benchmarks --tolerance 1e-5 1e-8

# 4. Verify after a code change
snapshot-tool verify path/to/benchmarks --tolerance 1e-5 1e-8 --summary summary.json

# Show or initialize config
snapshot-tool config --show
snapshot-tool config --init
```

`snapshot-tool` exits non-zero on a verification failure; `0` otherwise. `baseline` always exits `0` — it only records state.

## Global flags

These apply to every subcommand:

| Flag | Description |
|------|-------------|
| `--config PATH`, `-c PATH` | Path to a `snapshot_config.json` (default: `./snapshot_config.json`) |
| `--verbose`, `-v` | Verbose logging (prints comparison details and full tracebacks on errors) |
| `--quiet`, `-q` | Suppress per-benchmark `[PASS]` / `[SKIP]` lines |

## Parallel execution

`capture`, `verify`, and `baseline` accept opt-in parallelism:

| Flag | Description | Default |
|------|-------------|---------|
| `--parallel` | Run benchmarks across worker **processes** instead of serially | off (serial) |
| `--workers N` | Worker count when `--parallel` is set | `min(cpu_count, 8)` |

Default behaviour is unchanged — without `--parallel` everything runs serially in-process. With `--parallel`, `(benchmark, parameters)` tasks are distributed to a process pool (true parallelism + per-process `sys.settrace`/RNG/import isolation). Workers only execute, trace, and serialize the captured value; the **main process performs every SQLite write** (SQLite is single-writer) and all comparison. Parallel runs are verified to produce **byte-identical** snapshots to serial (same content-addressed sha256 blob hashes), so determinism is preserved. A task whose parameters can't be pickled across the process boundary transparently falls back to in-process execution.

```bash
# Capture a large suite using all cores (capped at 8)
snapshot-tool capture path/to/benchmarks --parallel

# Pin the worker count
snapshot-tool verify path/to/benchmarks --parallel --workers 4
```

## Subcommands

---

### `list`

AST-discovers benchmarks under `benchmark_dir` and prints them. No code is imported or executed.

```bash
snapshot-tool list <benchmark_dir> [--filter REGEX]
```

| Flag | Description |
|------|-------------|
| `benchmark_dir` | Directory to scan recursively for `*.py` (skips `__init__.py`) |
| `--filter REGEX` | Python regex matched against `f"{module_path}.{benchmark_name}"` |

For each benchmark, `list` prints the dotted path, its type (`function` or `method`), the setup method if any, and the number of parameter combinations. Use `--verbose` to also print the first five parameter combinations explicitly.

```bash
snapshot-tool list tests/test_repos/shapely_benchmarks --filter '^geometry\.'
```

---

### `capture`

Runs each discovered benchmark with tracing enabled and writes a snapshot per `(module, class, benchmark, parameter-tuple)` tuple into `--snapshot-dir`.

```bash
snapshot-tool capture <benchmark_dir> \
    [--filter REGEX] [--snapshot-dir DIR] [--timeout SEC]
```

| Flag | Description | Default |
|------|-------------|---------|
| `benchmark_dir` | Directory containing benchmark files | **required** |
| `--filter REGEX` | Python regex matched against `f"{module_path}.{benchmark_name}"` | — |
| `--snapshot-dir DIR` | Where to write snapshots | `.snapshots/` (from config) |
| `--timeout SEC` | Per-benchmark timeout in seconds | `300` |

For every parameter combination, `capture`:

1. Reseeds RNGs (`RNGPatcher.patch_all()`).
2. Loads the benchmark module via `importlib.util`, synthesizing any parent packages in `sys.modules` so relative imports work.
3. Calls `setup_cache()` once per class (cached), then `setup(*params)`, then the benchmark method.
4. Installs `sys.settrace` and runs. On return, takes the shallowest meaningful user-code return value.
5. Pickles, gzips, and content-addresses the result by sha256, then inserts (or refcounts) it into `<snapshot_dir>/snapshots.db`. A JSON metadata sidecar is also written under `<snapshot_dir>/<module>/<class.benchmark>/<param-hash>.json` for downstream tooling.
6. If the benchmark raises, times out, or returns something unpicklable, a **failed capture marker** is written instead — a snapshot with `capture_failed=True` in metadata. `verify` will skip these.

```bash
# Capture everything with a tight 30-second budget per benchmark
snapshot-tool capture tests/test_repos/shapely_benchmarks --timeout 30

# Only capture the coordinates and units modules
snapshot-tool capture tests/test_repos/astropy_benchmarks \
    --filter '^benchmarks\.(coordinates|units)'
```

!!! warning
    `capture` is destructive in the sense that it overwrites existing snapshots at the same path. If you want to keep an old set of snapshots around (e.g., for A/B comparison) move the directory aside first or pass a different `--snapshot-dir`.

---

### `verify`

Re-runs each benchmark, loads the matching snapshot, and compares them with the configured tolerance.

```bash
snapshot-tool verify <benchmark_dir> \
    [--filter REGEX] [--snapshot-dir DIR] \
    [--tolerance RTOL ATOL] [--summary summary.json] [--timeout SEC]
```

| Flag | Description | Default |
|------|-------------|---------|
| `benchmark_dir` | Directory containing benchmark files | **required** |
| `--filter REGEX` | Python regex matched against `f"{module_path}.{benchmark_name}"` | — |
| `--snapshot-dir DIR` | Where to read snapshots from | `.snapshots/` (from config) |
| `--tolerance RTOL ATOL` | Override `rtol` and `atol` for numeric comparison | from config |
| `--summary PATH` | Where to write the JSON summary | `summary.json` |
| `--timeout SEC` | Per-benchmark timeout in seconds | `300` |

Per-test outcomes are:

| Outcome | Meaning |
|---------|---------|
| `[PASS]` | Comparison matched within tolerance |
| `[FAIL]` | The current return value differs, **or** the benchmark crashed during verify (but succeeded during capture — usually a real regression or environment drift) |
| `[SKIP]` | No snapshot exists, the snapshot was a failed-capture marker, or the comparison itself was unsupported (e.g., generators, callables, unpicklable values) |

At the end, `verify` writes a JSON summary to `--summary`:

```json
{
  "total": 124,
  "passed": 118,
  "failed": 2,
  "skipped": 4,
  "timestamp": "2026-05-14T10:14:32.018000",
  "snapshot_dir": ".snapshots",
  "benchmark_dir": "tests/test_repos/shapely_benchmarks"
}
```

If a `baseline.json` is present in `--snapshot-dir`, the summary is **augmented** with a 3×3 transition matrix derived from `compute_transitions(baseline, verify)`:

```json
{
  ...
  "pass-to-pass": 100,
  "pass-to-fail": 2,
  "pass-to-skip": 0,
  "fail-to-pass": 1,
  "fail-to-fail": 15,
  "fail-to-skip": 0,
  "skip-to-pass": 0,
  "skip-to-fail": 0,
  "skip-to-skip": 4
}
```

See [Baseline & Verify](baseline-and-verify.md) for the transition semantics.

`verify` exits with code `1` if any test was marked `FAIL`, else `0`. `SKIP` does not fail the run.

---

### `baseline`

Same execution path as `verify`, but instead of failing on mismatches it **records the pass/fail/skip status of each test_id** into `<snapshot_dir>/baseline.json`. A later `verify` consumes this file to compute the transition matrix.

```bash
snapshot-tool baseline <benchmark_dir> \
    [--filter REGEX] [--snapshot-dir DIR] \
    [--tolerance RTOL ATOL] [--timeout SEC]
```

The flags mean exactly what they do for `verify`, except there is no `--summary` — the output is `baseline.json` inside the snapshot directory.

A typical CI workflow:

```bash
# On the "before" revision
snapshot-tool capture  benchmarks
snapshot-tool baseline benchmarks   # writes baseline.json

# ... apply optimization patch, then on the "after" revision ...

snapshot-tool verify   benchmarks   # summary.json now includes the transition matrix
```

`baseline` always returns `0` — it only records state.

---

### `clean`

Reports on the snapshot directory. Currently informational only.

```bash
snapshot-tool clean [--snapshot-dir DIR] [--dry-run]
```

| Flag | Description |
|------|-------------|
| `--snapshot-dir DIR` | Directory to inspect (default: from config) |
| `--dry-run` | Show what *would* be deleted without deleting |

Today `clean` prints the snapshot count and total size on disk; it does not delete files. To wipe snapshots, remove the directory yourself (`rm -rf .snapshots/`) — `customtest.sh` is the canonical example.

---

### `config`

Manage the `snapshot_config.json` file.

```bash
snapshot-tool config --init    # write default snapshot_config.json
snapshot-tool config --show    # print the active config
```

The config file is loaded from `./snapshot_config.json` unless overridden with `--config PATH`. See [Configuration](configuration.md) for the schema and every field.

## Filter semantics

`--filter` is a Python regex (via `re.search`) matched against `f"{module_path}.{benchmark_name}"` — not the file path. For class-based benchmarks the benchmark name is the *method* name; the class is not part of the matched string. To filter on a class, target its module:

```bash
# Only TimeAngularSeparation.* methods, which live in benchmarks/coordinates.py
snapshot-tool list benchmarks --filter '^coordinates\..*'
```

## Typical workflow

```bash
# 1. Iterate locally on a benchmark suite
uv run snapshot-tool list      benchmarks
uv run snapshot-tool capture   benchmarks --timeout 30
uv run snapshot-tool baseline  benchmarks
uv run snapshot-tool verify    benchmarks   # baseline of yourself: should be 100% pass-to-pass

# 2. Change code, re-verify
uv run snapshot-tool verify    benchmarks   # transitions tell you what flipped
```

For a one-shot smoke test, `customtest.sh` runs the full roundtrip against the bundled shapely repo.

## Next steps

- [Configuration](configuration.md) — the `snapshot_config.json` schema.
- [Baseline & Verify](baseline-and-verify.md) — what each transition cell means.
- [Discovery](discovery.md) — what `list` is actually doing under the hood.
