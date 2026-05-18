# Configuration

`snapshot-tool` reads its configuration from `./snapshot_config.json` by default. The file is optional — every field has a sensible default — and is loaded by `ConfigManager` at CLI startup.

Override the path with `--config PATH` (or `-c PATH`). Override individual fields with the CLI flags described in the [CLI guide](cli.md).

## Initialize the file

```bash
snapshot-tool config --init
```

writes a default `snapshot_config.json` with every field at its built-in default:

```json
{
  "benchmark_dir": "benchmarks/",
  "snapshot_dir": ".snapshots/",
  "project_dir": null,
  "tolerance": {
    "rtol": 1e-5,
    "atol": 1e-8,
    "equal_nan": true
  },
  "exclude_benchmarks": [],
  "trace_depth_limit": 100,
  "verbose": false,
  "quiet": false
}
```

To inspect what's currently active:

```bash
snapshot-tool config --show
```

## Fields

### Directories

| Field | Default | What it does |
|-------|---------|--------------|
| `benchmark_dir` | `benchmarks/` | Default directory for `list` / `capture` / `verify` / `baseline` if not provided on the command line. **Currently informational** — the CLI requires `benchmark_dir` as a positional argument on every subcommand, so this is rarely used by itself. |
| `snapshot_dir` | `.snapshots/` | Where snapshots are written and read. Overridden by `--snapshot-dir`. |
| `project_dir` | `null` | Optional override for the project root. By default `BenchmarkRunner` uses `benchmark_dir.parent` and inserts it onto `sys.path` so the benchmarked package can be imported. Set this when your project root isn't the parent of the benchmark directory. |

### Comparison tolerance

| Field | Default | What it does |
|-------|---------|--------------|
| `tolerance.rtol` | `1e-5` | Relative tolerance for numeric comparison. Overridden by the first value of `--tolerance RTOL ATOL`. |
| `tolerance.atol` | `1e-8` | Absolute tolerance. Overridden by the second value of `--tolerance RTOL ATOL`. |
| `tolerance.equal_nan` | `true` | When `true`, two NaN values at the same position compare equal (snapshot default — a deterministic NaN that reappears unchanged is not a regression). Set `false` for strict `numpy.isclose` semantics. No CLI override — set in the file. |

The comparison formula is `|a - b| <= atol + rtol * |b|`, matching `numpy.isclose` semantics. `b` is the *expected* (snapshot) value.

See [Comparison](comparison.md) for the full dispatch and additional knobs (`strict_types`, `strict_shapes`) available when constructing a `ComparisonConfig` programmatically.

### Filtering

| Field | Default | What it does |
|-------|---------|--------------|
| `exclude_benchmarks` | `[]` | List of benchmark *names* (not full paths) to skip. Each entry is either an exact match or a prefix wildcard ending in `*`. Independent of `--filter`. |

`should_exclude_benchmark` does a simple `==` or `startswith` check against the benchmark `name` only — it doesn't see the module path. To exclude by module, use the `--filter` regex on the command line instead.

Examples:

```json
{
  "exclude_benchmarks": [
    "time_flaky",        // exact match
    "time_slow_*",       // prefix wildcard
    "peakmem_*"          // skip all peakmem benchmarks
  ]
}
```

### Tracing

| Field | Default | What it does |
|-------|---------|--------------|
| `trace_depth_limit` | `100` | Maximum call depth for the [tracer](tracing.md). The tracer stops recording new frames beyond this depth to avoid infinite-recursion blowups. The default is generous for typical benchmark code; lower it if you're seeing pathological recursion. |

This field is read into the dataclass but not currently threaded through to `ExecutionTracer`'s constructor by the CLI — the tracer always uses its built-in `max_depth=100`. The two values agree by default; if you want a different depth, instantiate `ExecutionTracer(max_depth=N)` and use the [Python API](../getting-started/quickstart.md) directly.

### Output

| Field | Default | What it does |
|-------|---------|--------------|
| `verbose` | `false` | Equivalent to `--verbose` / `-v`. Enables debug-level comparison detail and full tracebacks on errors. |
| `quiet` | `false` | Equivalent to `--quiet` / `-q`. Suppresses per-benchmark `[PASS]` / `[SKIP]` lines (failures are always printed). |

CLI flags take precedence — passing `-v` on the command line forces `verbose=True` for that invocation regardless of the file.

## CLI flag precedence

For every overlapping field, the precedence is:

```
CLI flag  >  snapshot_config.json  >  built-in default
```

So a `verify` command like:

```bash
snapshot-tool -v --config ./conf.json verify ./benchmarks \
    --snapshot-dir ./snapshots --tolerance 1e-4 1e-6
```

resolves to:

- `snapshot_dir = "./snapshots"` (from `--snapshot-dir`, overriding `conf.json`).
- `tolerance.rtol = 1e-4`, `tolerance.atol = 1e-6` (from `--tolerance`).
- `tolerance.equal_nan = <value from conf.json or true>` (not overridable on the CLI).
- `verbose = True` (from `-v`).

## Programmatic configuration

```python
from snapshot_tool import ConfigManager, SnapshotConfig
from pathlib import Path

# Load (or default if missing)
manager = ConfigManager(Path("snapshot_config.json"))
config: SnapshotConfig = manager.get_config()

# Mutate in memory
manager.update_config(verbose=True, snapshot_dir="custom/.snapshots")

# Persist
manager.save_config()

# Or build one from scratch
config = SnapshotConfig(
    benchmark_dir="benchmarks/",
    snapshot_dir=".snapshots/",
    tolerance={"rtol": 1e-4, "atol": 1e-6, "equal_nan": True},
    exclude_benchmarks=["time_flaky_*"],
)
config.save_to_file(Path("snapshot_config.json"))
```

## Next steps

- [CLI](cli.md) — Every subcommand and flag.
- [Comparison](comparison.md) — Tolerance semantics and the strategy chain.
- [Snapshot Storage](storage.md) — Where snapshots live on disk.
