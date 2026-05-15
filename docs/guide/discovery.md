# Discovery

`BenchmarkDiscovery` finds ASV benchmarks under a directory without ever importing the benchmark modules. It AST-parses each `*.py` file (skipping `__init__.py`) and emits a `BenchmarkInfo` record per discovered benchmark.

## What counts as a benchmark

A function or method is a benchmark if its name starts with one of the standard ASV prefixes:

| Prefix | ASV semantics |
|--------|---------------|
| `time_` | Wall-clock timing |
| `timeraw_` | Wall-clock timing in a fresh subprocess |
| `mem_` | Memory consumption |
| `peakmem_` | Peak memory consumption |
| `track_` | A user-defined metric |

`snapshot-tool` discovers all five — it doesn't care about ASV's measurement type, only that the function/method exists and is callable. What `snapshot-tool` records is the **return value** captured by the [tracer](tracing.md), not the timing/memory metric.

## What gets emitted

Each discovered benchmark becomes a `BenchmarkInfo` dataclass:

```python
@dataclass
class BenchmarkInfo:
    name: str                          # e.g., "time_union"
    module_path: str                   # dotted path relative to benchmark_dir, e.g., "geometry.union"
    benchmark_type: str                # "function" or "method"
    class_name: Optional[str] = None
    params: Optional[list[list[Any]]] = None
    param_names: Optional[list[str]] = None
    setup_method: Optional[str] = None
    has_setup: bool = False
    has_setup_cache: bool = False
    method_params: Optional[list[str]] = None
    needs_runtime_eval: bool = False
```

The `module_path` is computed by taking the file's path relative to `benchmark_dir`, stripping the `.py` extension, and replacing path separators with `.`. So `benchmarks/geometry/union.py` becomes `geometry.union`.

## Function-level vs class-level benchmarks

`BenchmarkDiscovery` walks each file's top-level body twice:

1. **Functions at module scope** — `def time_*(...)` declared directly in the file. Emitted as `benchmark_type="function"`.
2. **Classes** — `class TimeFoo:` with one or more `time_*` methods. For each such method, `BenchmarkDiscovery` emits a separate `BenchmarkInfo` with `benchmark_type="method"`, and copies the class's `params`, `param_names`, `setup` / `setup_cache` attributes into every record.

Functions inside classes are only discovered as methods; nested functions inside `def` blocks are ignored.

## Parameters

ASV uses class-level `params` and `param_names` attributes to express parametric benchmarks. `BenchmarkDiscovery` extracts both:

```python
class TimeUnion:
    params = ([10, 100, 1000], ["polygon", "linestring"])
    param_names = ["n", "geom_type"]

    def setup(self, n, geom_type):
        ...

    def time_union(self, n, geom_type):
        ...
```

For each method in the class, the emitted `BenchmarkInfo` carries:

- `params = [[10, 100, 1000], ["polygon", "linestring"]]`
- `param_names = ["n", "geom_type"]`
- `setup_method = "setup"`, `has_setup = True`

`runner.get_param_combinations(benchmark)` produces the Cartesian product `[(10, "polygon"), (10, "linestring"), (100, "polygon"), ...]`.

### Runtime-evaluated params

ASV allows `params` to be any Python expression — a generator, a comprehension, a function call, a module-level constant. `BenchmarkDiscovery` can only statically extract **literal** values: `ast.Constant`, lists/tuples of constants, and nested combinations of those. Anything else is flagged:

```python
class TimeFoo:
    params = [n for n in range(10)]       # ← comprehension, not a literal list
    # or
    params = list(SOMETHING)              # ← call, not a literal list
```

When `BenchmarkDiscovery` sees a non-literal expression in `params`, it sets `needs_runtime_eval=True` and leaves `params=None`. The runner then evaluates the class's `params` attribute at runtime (after `_load_module`) and back-fills the field before generating combinations.

## What discovery skips

- `__init__.py` — never parsed.
- Files that fail to parse — a warning is logged via `logger.warning(...)`; discovery continues.
- Imports — never followed. If `benchmarks/foo.py` imports from `benchmarks/bar.py`, both are scanned independently as files.
- Inheritance — a class that inherits benchmark methods from a parent class will only have its **own** `time_*` methods discovered. ASV's class-based benchmark inheritance is not modeled.

## Programmatic use

```python
from snapshot_tool import BenchmarkDiscovery

discovery = BenchmarkDiscovery("benchmarks/")
all_benchmarks = discovery.discover_all()

# Look up a specific benchmark
b = discovery.get_benchmark_by_name("time_union")

# All benchmarks in one module
geom_benchmarks = discovery.get_benchmarks_by_module("geometry.union")

# Generate the parameter Cartesian product for a class benchmark
combinations = discovery.generate_parameter_combinations(b)
```

Note that `generate_parameter_combinations` on a benchmark with `needs_runtime_eval=True` returns the placeholder `[("<runtime_eval>",)]` — use `BenchmarkRunner.get_param_combinations(b)` instead, which loads the module and back-fills `params` first.

## Next steps

- [Tracing](tracing.md) — How `sys.settrace` decides which return value to capture.
- [CLI guide → `list`](cli.md#list) — The user-facing wrapper around `discover_all()`.
