# Configuration

`snapshot-tool` reads its configuration from `./snapshot_config.json` in the current directory. The file is optional; every field has a default. `ConfigManager` loads it when the CLI starts.

!!! warning
    The global `--config PATH` (`-c PATH`) flag is accepted but not used. The CLI always reads `./snapshot_config.json`. To use another file, run the tool from the directory that holds it, or load it with `ConfigManager(path)` from Python.

Some fields can be overridden with the CLI flags described in the [CLI guide](cli.md).

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

To print the active configuration:

```bash
snapshot-tool config --show
```

## Fields

### Directories

| Field | Default | What it does |
|-------|---------|--------------|
| `benchmark_dir` | `benchmarks/` | Not used by the CLI. Every subcommand takes `benchmark_dir` as a required positional argument. |
| `snapshot_dir` | `.snapshots/` | Where snapshots are written and read. Overridden by `--snapshot-dir`. |
| `project_dir` | `null` | Not used by the CLI. The CLI always puts `benchmark_dir.parent` on `sys.path`. To use another project root, pass `project_dir` to `BenchmarkRunner` from Python. |

### Comparison tolerance

| Field | Default | What it does |
|-------|---------|--------------|
| `tolerance.rtol` | `1e-5` | Relative tolerance for numeric comparison. Overridden by the first value of `--tolerance RTOL ATOL`. |
| `tolerance.atol` | `1e-8` | Absolute tolerance. Overridden by the second value of `--tolerance RTOL ATOL`. |
| `tolerance.equal_nan` | `true` | When `true`, two NaN values at the same position compare equal: a NaN that appears again unchanged is not a regression. Set `false` for strict `numpy.isclose` semantics. There is no CLI flag. The file value is used only when `--tolerance` is not given; with `--tolerance`, `equal_nan` is `true`. |

The comparison formula is `|a - b| <= atol + rtol * |b|`, matching `numpy.isclose` semantics. `b` is the *expected* (snapshot) value.

See [Comparison](comparison.md) for the full dispatch and the other options (`strict_types`, `strict_shapes`). These are available only when you construct a `ComparisonConfig` in Python.

### Filtering

| Field | Default | What it does |
|-------|---------|--------------|
| `exclude_benchmarks` | `[]` | List of benchmark *names* (not full paths) to skip. Each entry is either an exact match or a prefix wildcard ending in `*`. Independent of `--filter`. |

`should_exclude_benchmark` does an `==` or `startswith` check against the benchmark `name` only; it does not see the module path. To exclude by module, use the `--filter` regex on the command line.

Example: skip the benchmark `time_flaky` (exact match), every benchmark whose name starts with `time_slow_`, and all `peakmem_` benchmarks:

```json
{
  "exclude_benchmarks": ["time_flaky", "time_slow_*", "peakmem_*"]
}
```

### Tracing

| Field | Default | What it does |
|-------|---------|--------------|
| `trace_depth_limit` | `100` | Not used by the CLI. The [tracer](tracing.md) always uses `ExecutionTracer(max_depth=100)` and stops tracing frames deeper than that. For another depth, construct `ExecutionTracer(max_depth=N)` with the [Python API](../getting-started/quickstart.md). |

### Output

| Field | Default | What it does |
|-------|---------|--------------|
| `verbose` | `false` | Same as `--verbose` / `-v`. Prints a traceback when a command raises an error. Comparison details are logged at DEBUG level, which the default log setup does not show. |
| `quiet` | `false` | Same as `--quiet` / `-q`. Hides most per-benchmark `[PASS]` / `[SKIP]` lines. Failures are always printed. |

`-v` and `-q` on the command line set the field to `true` for that run. They cannot set it to `false`.

## CLI flag precedence

For every field that has a CLI flag, the order is:

```
CLI flag  >  ./snapshot_config.json  >  built-in default
```

So this `verify` command:

```bash
snapshot-tool -v verify ./benchmarks \
    --snapshot-dir ./snapshots --tolerance 1e-4 1e-6
```

resolves to:

- `snapshot_dir = "./snapshots"` (from `--snapshot-dir`, which overrides the file).
- `tolerance.rtol = 1e-4`, `tolerance.atol = 1e-6` (from `--tolerance`).
- `tolerance.equal_nan = true` (because `--tolerance` is given, the file value is not read).
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
