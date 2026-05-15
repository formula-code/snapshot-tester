# Tracing

ASV benchmarks usually don't return anything — they exist to be timed, not to produce output. `snapshot-tool` extracts a return value anyway by installing a `sys.settrace` callback during execution and capturing the **shallowest meaningful user-code return value**.

## The problem

A typical ASV benchmark looks like this:

```python
def time_compute(self, n):
    self.solver.run(n)
```

There's nothing to compare here. The interesting value is what `self.solver.run(n)` returned internally — but we don't get to see it from the outside, because the function body discards it.

Two mechanisms cooperate to capture it:

1. **AST rewrite (preferred)** — Before execution, `BenchmarkRunner` re-parses the benchmark method's source. If the last statement is a bare expression (`self.solver.run(n)`), it rewrites it to `return self.solver.run(n)` and recompiles. This is the cheap path and works for the vast majority of ASV-style benchmarks. It's important because C/Cython-implemented internals are invisible to `sys.settrace`, so the AST rewrite is sometimes the **only** way to capture a value.
2. **`sys.settrace` (fallback / supplement)** — While the benchmark runs, a per-frame trace callback records the return value of every traced call. The shallowest meaningful one — the one closest to the benchmark itself — wins.

If both fail, the benchmark is still reported as `success=True` with `return_value=None`. The capture is silently empty; verify will read it back as `None == None` (a pass).

## What "meaningful" means

`ExecutionTracer._should_trace_frame` filters frames aggressively. A frame is traced only if **all** of the following are true:

- The frame has a `__name__` in its globals.
- The module isn't in the **stdlib**:
    - Python ≥ 3.10: `sys.stdlib_module_names` is consulted (the canonical list).
    - Plus a hand-curated set covering common stdlib modules and earlier Python versions (`builtins`, `os`, `pathlib`, `typing`, `re`, `json`, `io`, `threading`, `subprocess`, `argparse`, `logging`, `tempfile`, `traceback`, `pytest`, `_pytest`, ...).
- The module isn't a known third-party internal:
    - **numpy** internals: `numpy._core`, `numpy.core`, `numpy.lib`, `numpy.ma`, `numpy.array_api`, `numpy.f2py`, `numpy.fft`, `numpy.linalg`, `numpy.random`, `numpy.testing`, `numpy._`, `numpy.compat`, `numpy.matrixlib`.
    - **shapely** internals: `shapely.lib`, `shapely._`, `shapely.geos`, `shapely.geometry.base`.
- The function isn't a dunder method (`__init__`, `__repr__`, `__eq__`, `__len__`, `__iter__`, ...) or a lambda / synthetic name (`<lambda>`, anything starting with `<`).

The list of suppressed modules is intentionally broad — we want to trace **user benchmark code and the public surface of scientific libraries**, not their internal helpers. A `shapely.geometry.Polygon.union` call gets traced; the `shapely.lib.intersection` C-binding it calls does not.

## "Shallowest" vs "deepest"

The dataclass field is called `deepest_call`, but the capture rule prefers **shallower** frames:

```python
# tracer.py
if arg is not None and not self._is_meaningless_return(arg):
    if self.deepest_call is None or self.current_depth <= self.deepest_call.depth:
        self.deepest_call = TraceResult(...)
```

In other words: every meaningful return value at a depth `<=` the current best replaces it. This biases capture toward the **first user-code call from the benchmark**, not the innermost.

!!! note
    The field name and docstring say "deepest" but the comparator is `<=`, so what you actually get is the shallowest frame. This is intentional — naming hasn't been updated to match the behavior.

## What's "meaningless"

`_is_meaningless_return` skips returns that aren't worth snapshotting:

- `None` — always skipped.
- Empty containers (`len(value) == 0` if `__len__` exists).
- Trivial scalar values (`0`, `0.0`, `""`) — but not `False` (booleans are always kept).

Class instances are always kept, even if they're empty — they're typically the "real" return value. Numpy arrays, strings (non-empty), and non-zero numbers are always kept.

## Lifecycle

```python
tracer = ExecutionTracer(max_depth=100)
tracer.start_tracing()           # sys.settrace(self._trace_calls)
try:
    benchmark()
finally:
    result = tracer.stop_tracing()  # sys.settrace(None); returns the captured TraceResult or None
```

If `benchmark()` raises before any traced frame returns meaningfully, `result.success = False` and `result.error` is set. If `benchmark()` returns normally but nothing meaningful was traced (e.g., the whole call ran inside C code), `result` is `None`.

`BenchmarkRunner` normalizes `None` to `TraceResult(success=True, return_value=None)` — it considers "no trace captured" a successful run with empty output, not a failure.

## `TraceResult`

```python
@dataclass
class TraceResult:
    return_value: Any
    function_name: str            # name of the frame whose return was captured
    module_name: str              # module of that frame
    depth: int                    # call depth at capture time
    success: bool
    error: Optional[Exception] = None
```

`function_name` and `module_name` are useful for debugging: they tell you exactly which function the snapshot is coming from. If a snapshot starts failing after a refactor that renames or relocates a helper, this is where you'll see it.

## Overhead

`sys.settrace` is meaningfully slow — every Python function call goes through your callback. `snapshot-tool` runs benchmarks under tracing only during `capture` / `verify` / `baseline`. The actual ASV timing job is unaffected; the snapshot harness is separate.

## Programmatic use

```python
from snapshot_tool import ExecutionTracer

tracer = ExecutionTracer()
result = tracer.trace_execution(my_function, arg1, arg2)
if result.success:
    print(result.return_value)

print(tracer.get_trace_stats())
# {"max_depth_reached": 7, "deepest_call_depth": 1, ...}
```

## Next steps

- [Comparison](comparison.md) — How captured values are compared on `verify`.
- [Determinism](determinism.md) — Why RNGs are reseeded before every trace.
