# snapshot-tool

<p align="center">
  <a href="https://formula-code.github.io/">
    <img src="https://img.shields.io/badge/%F0%9F%8C%90%20Website-0A7A5E?style=for-the-badge" alt="FormulaCode Website">
  </a>
  <a href="https://github.com/formula-code/snapshot-tester">
    <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
  </a>
  <a href="https://github.com/airspeed-velocity/asv">
    <img src="https://img.shields.io/badge/ASV-Airspeed%20Velocity-1F6FEB?style=for-the-badge" alt="ASV">
  </a>
</p>

`snapshot-tool` is a snapshot-testing harness for [ASV](https://github.com/airspeed-velocity/asv) (airspeed-velocity) benchmarks. It captures the return value of each benchmark on a known-good revision, then runs the same benchmarks again after a code change and reports which outputs changed. Numbers are compared with a tolerance.

It is the correctness companion to ASV's performance numbers: ASV tells you *how fast* a change is, `snapshot-tool` tells you *whether the result is still correct*.

## How it works

```mermaid
graph LR
    A[benchmarks/*.py] -->|AST discovery| B
    B -->|sys.settrace| C
    C -->|pickle + json| D
    D -->|tolerance compare| E

    B["Benchmark<br/>Discovery"]
    C["Traced<br/>Execution"]
    D[".snapshots/"]
    E["Verify /<br/>Baseline"]
```

The pipeline runs in four phases:

1. **Discover** — AST-parse each `*.py` under `benchmark_dir` to find ASV-prefixed functions and methods (`time_`, `timeraw_`, `mem_`, `peakmem_`, `track_`). No imports happen at discovery time.
2. **Run** — Execute each benchmark with RNGs reseeded and a per-benchmark timeout. A trailing expression statement in the benchmark is rewritten into a `return`, and a `sys.settrace` tracer keeps the shallowest meaningful return value from user code (stdlib, numpy internals, and shapely internals are filtered out).
3. **Persist** — Write a row into a single SQLite database at `.snapshots/snapshots.db`. Payloads are pickled, gzipped, and content-addressed by sha256 so benchmarks producing identical outputs share one blob on disk. A JSON metadata sidecar is also written per `(module, class, benchmark, parameters)` tuple for downstream tooling.
4. **Compare** — On a later run, replay each benchmark and compare to the stored snapshot using a pure-Python `isclose` (`rtol` / `atol` / `equal_nan`).

## Get started

The full roundtrip is four commands:

```bash
# 1. See what's discoverable
snapshot-tool list path/to/benchmarks

# 2. Capture a baseline snapshot of every benchmark's output
snapshot-tool capture path/to/benchmarks

# 3. (Optional) Record the current pass/fail state as a baseline for transition tracking
snapshot-tool baseline path/to/benchmarks

# 4. After a code change, verify nothing regressed
snapshot-tool verify path/to/benchmarks --tolerance 1e-5 1e-8
```

See the **[CLI guide](guide/cli.md)** for the full subcommand reference.

## Key features

- **No changes to benchmarks** — Benchmarks do not need to return a value. The runner turns a trailing expression into a `return`, and the tracer keeps the shallowest meaningful user-code return value, so ASV benchmarks that compute and discard a result work unchanged.
- **Tolerance-aware comparison** — Pure-Python `isclose` (`rtol` / `atol` / `equal_nan`) for scalars, numpy arrays (element-wise), Python sequences, dicts, and custom classes with `__eq__`. Numpy is optional — detected lazily.
- **Reseeded RNGs** — Python `random`, the numpy legacy API, PyTorch, and TensorFlow are reseeded before every run via [`RNGPatcher`](guide/determinism.md). `numpy.random.Generator` instances are not seeded.
- **Per-benchmark timeouts** — On the main thread a `SIGALRM` timer stops a benchmark that runs past `--timeout`, so one hung benchmark does not stop the whole run.
- **One SQLite file, deduplicated and gzipped** — Every payload is pickled, gzipped, and content-addressed by sha256 in a single `snapshots.db`. Benchmarks with identical outputs share one blob through a refcount.
- **Baseline → verify transition matrix** — Record a `baseline.json` of pass/fail/skip statuses on one revision, then `verify` against another to get transition counts (`pass-to-fail`, `fail-to-pass`, etc.). Use this to grade optimization attempts in CI.
- **Python 3.8 → 3.13** — Standard library only at runtime. The comparator uses numpy only if it is installed.

## Quick links

- [Installation](getting-started/installation.md) — Set up the dev environment with `uv`.
- [**CLI (`snapshot-tool`)**](guide/cli.md) — The main entry point; full subcommand reference.
- [Python API Quickstart](getting-started/quickstart.md) — Programmatic usage of `BenchmarkDiscovery`, `BenchmarkRunner`, `SnapshotManager`, and `Comparator`.
- [Baseline & Verify](guide/baseline-and-verify.md) — The 3×3 transition matrix and how to wire it into CI.
- [Configuration](guide/configuration.md) — `snapshot_config.json` and CLI flag reference.
