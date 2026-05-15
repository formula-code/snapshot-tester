# Python API Quickstart

This guide walks through `snapshot-tool`'s public API. Everything below is re-exported from the top-level `snapshot_tool` package.

For day-to-day use the [CLI](../guide/cli.md) is enough — reach for the Python API when you need to embed snapshot testing inside a larger harness (e.g., a CI pre-flight, a notebook, or a custom comparison strategy).

## Discovering benchmarks

`BenchmarkDiscovery` walks a directory tree and AST-parses every `*.py` file. It never imports the benchmark modules at discovery time — that's deferred to the runner.

```python
from snapshot_tool import BenchmarkDiscovery

discovery = BenchmarkDiscovery("path/to/benchmarks")
benchmarks = discovery.discover_all()

for b in benchmarks:
    print(f"{b.module_path}.{b.name}  (type={b.benchmark_type})")
    if b.params:
        print(f"  params: {b.params}")
    if b.needs_runtime_eval:
        print("  params need runtime evaluation (not statically resolvable)")
```

Each entry is a `BenchmarkInfo` dataclass with everything the runner needs:

| Field | Meaning |
|-------|---------|
| `name` | The benchmark function/method name (e.g., `time_compute`) |
| `module_path` | Dotted module path relative to `benchmark_dir` (e.g., `geometry.union`) |
| `benchmark_type` | `"function"` or `"method"` |
| `class_name` | For methods, the enclosing class name |
| `params` | List-of-lists of parameter values (e.g., `[[1, 10, 100], ["a", "b"]]`) |
| `param_names` | Optional list of parameter names |
| `has_setup` / `setup_method` | Whether the class has a `setup` method to call before each run |
| `has_setup_cache` | Whether the class has a `setup_cache` method (called once per class) |
| `needs_runtime_eval` | `True` if `params` contains a non-literal expression (a comprehension, call, etc.) that has to be evaluated at runtime |

## Running a benchmark with tracing

`BenchmarkRunner` ties discovery, RNG patching, and the tracer together:

```python
from snapshot_tool import BenchmarkRunner

runner = BenchmarkRunner(
    benchmark_dir="path/to/benchmarks",
    seed=42,             # Deterministic seed applied before every run
    timeout=300.0,       # Per-benchmark timeout in seconds
)

result = runner.run_benchmark(benchmarks[0])

if result.success:
    print(f"Captured from {result.function_name} at depth {result.depth}")
    print(f"Return value: {result.return_value!r}")
else:
    print(f"Failed: {result.error}")
```

For parameterized benchmarks, generate all combinations and run each:

```python
for params in runner.get_param_combinations(benchmark):
    result = runner.run_benchmark(benchmark, params)
    ...
```

The runner wraps each call in a `ThreadPoolExecutor.submit(...).result(timeout=...)`. On timeout the future is cancelled and a `TraceResult` with `success=False` and `error=TimeoutError(...)` is returned — but the underlying thread keeps running, so persistent hangs can leak threads. Keep timeouts realistic.

## Storing and loading snapshots

`SnapshotManager` writes captured values into a single SQLite database (`<snapshot_dir>/snapshots.db`) — pickled, gzipped, and deduplicated by sha256 — and emits a JSON metadata sidecar next to where the snapshot would have lived under the old per-file layout. `store_snapshot` returns the sidecar path:

```python
from snapshot_tool import SnapshotManager

storage = SnapshotManager(".snapshots/")

storage.store_snapshot(
    benchmark_name=benchmark.name,
    module_path=benchmark.module_path,
    parameters=(),
    param_names=None,
    return_value=result.return_value,
    class_name=benchmark.class_name,
)
```

Later, load it back:

```python
loaded = storage.load_snapshot(
    benchmark_name=benchmark.name,
    module_path=benchmark.module_path,
    parameters=(),
    class_name=benchmark.class_name,
)
if loaded is not None:
    expected_value, metadata = loaded
    # metadata is a SnapshotMetadata: timestamp, git_commit, git_branch,
    # python_version, platform, capture_failed, failure_reason
```

See [Snapshot Storage](../guide/storage.md) for the full on-disk layout.

## Comparing outputs

`Comparator` dispatches across numpy arrays, scalars, sequences, dicts, and custom classes — falling back to `==` and `NotImplemented`-aware skips for types it doesn't recognize. Numpy is optional; the comparator probes for it lazily.

```python
from snapshot_tool import Comparator, ComparisonConfig

config = ComparisonConfig(
    rtol=1e-5,
    atol=1e-8,
    equal_nan=False,
    strict_types=True,
    strict_shapes=True,
)
comparator = Comparator(config)

comparison = comparator.compare(actual=result.return_value, expected=expected_value)

if comparison.match:
    print("OK")
elif comparison.skipped:
    print(f"Skipped: {comparison.details}")
else:
    print(f"Failed: {comparison.error_message}")
    if comparison.details:
        print(f"  details: {comparison.details}")
```

The strategy dispatch order is fixed:

1. Serialized placeholders (`__generator__`, `__callable__`, `__unpicklable__`) → skip.
2. Serialized class instance (`__class_instance__`) → compare `__dict__` recursively.
3. Numpy arrays → shape/dtype check, then element-wise `isclose`.
4. Numeric scalars → pure-Python `isclose`.
5. Objects with a custom `__eq__` → use it (handles array-returning equality on `SkyCoord`, pandas `Series`, etc.).
6. Sequences (list/tuple) → length + element-wise recursion.
7. Dicts → key-set match + value recursion.
8. Fallback `==` with `NotImplemented`-aware skip.

See [Comparison](../guide/comparison.md) for the full dispatch logic.

## A complete capture/verify loop

```python
from snapshot_tool import (
    BenchmarkDiscovery, BenchmarkRunner, SnapshotManager,
    Comparator, ComparisonConfig,
)

bench_dir = "path/to/benchmarks"
snap_dir = ".snapshots/"

discovery = BenchmarkDiscovery(bench_dir)
runner = BenchmarkRunner(bench_dir, seed=42, timeout=60.0)
storage = SnapshotManager(snap_dir)
comparator = Comparator(ComparisonConfig(rtol=1e-5, atol=1e-8))

# Phase 1: capture
for b in discovery.discover_all():
    for params in runner.get_param_combinations(b):
        result = runner.run_benchmark(b, params)
        if result and result.success:
            storage.store_snapshot(
                benchmark_name=b.name,
                module_path=b.module_path,
                parameters=params,
                param_names=b.param_names,
                return_value=result.return_value,
                class_name=b.class_name,
            )

# ... change code, then ...

# Phase 2: verify
for b in discovery.discover_all():
    for params in runner.get_param_combinations(b):
        loaded = storage.load_snapshot(b.name, b.module_path, params, b.class_name)
        if loaded is None:
            continue
        expected, _meta = loaded
        result = runner.run_benchmark(b, params)
        if not result or not result.success:
            print(f"FAIL (runtime): {b.module_path}.{b.name} {params}")
            continue
        cmp = comparator.compare(result.return_value, expected)
        if not cmp.match and not cmp.skipped:
            print(f"FAIL: {b.module_path}.{b.name} {params}: {cmp.error_message}")
```

## Forcing determinism without the runner

If you're running benchmark code outside of `BenchmarkRunner` (e.g., from a notebook), reseed all RNGs manually:

```python
from snapshot_tool import reset_all_rngs

reset_all_rngs(seed=42)   # reseeds Python random, numpy, torch, tensorflow
```

`reset_all_rngs` is the lightweight version of `RNGPatcher.patch_all()` — same seed, no state tracking. See [Determinism](../guide/determinism.md) for the full story.

## Next steps

- [CLI](../guide/cli.md) — Reference for all six subcommands.
- [Baseline & Verify](../guide/baseline-and-verify.md) — The 3×3 transition matrix.
- [Comparison](../guide/comparison.md) — Tolerance semantics and the strategy dispatch chain.
- [Determinism](../guide/determinism.md) — How `RNGPatcher` keeps captures bit-stable.
