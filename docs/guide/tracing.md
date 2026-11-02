# Tracing

ASV benchmarks usually do not return anything; they are timed, not checked. `snapshot-tool` gets a return value anyway. It rewrites the benchmark's last expression into a `return`, and it installs a `sys.settrace` callback that keeps the **shallowest meaningful user-code return value**.

## The problem

A typical ASV benchmark looks like this:

```python
def time_compute(self, n):
    self.solver.run(n)
```

There is nothing to compare here. The useful value is what `self.solver.run(n)` returned, but the function body discards it.

Two mechanisms cooperate to capture it:

1. **AST rewrite** — Before execution, `BenchmarkRunner` parses the benchmark's source again. If the last statement is a bare expression (`self.solver.run(n)`), it rewrites it to `return self.solver.run(n)` and compiles it. This works for most ASV benchmarks. It is necessary because `sys.settrace` cannot see functions written in C or Cython, so the rewrite is sometimes the **only** way to capture a value. For a module-level function benchmark, the value that the rewritten function returns is used directly, without the tracer.
2. **`sys.settrace`** — While a method benchmark runs, a per-frame trace callback records the return value of every traced call. The shallowest meaningful one wins. After the rewrite, that is usually the benchmark method itself.

If neither gives a value, the benchmark is still reported as `success=True` with `return_value=None`. The snapshot is empty, and verify reads it back as `None == None` (a pass).

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

The list of excluded modules is broad on purpose: the tracer must see **user benchmark code and the public API of scientific libraries**, not their internal helpers. A `shapely.geometry.Polygon.union` call gets traced; the `shapely.lib.intersection` C-binding it calls does not.

## "Shallowest" vs "deepest"

The dataclass field is called `deepest_call`, but the capture rule prefers **shallower** frames:

```python
# tracer.py
if arg is not None and not self._is_meaningless_return(arg):
    if self.deepest_call is None or self.current_depth <= self.deepest_call.depth:
        self.deepest_call = TraceResult(...)
```

Every meaningful return value at a depth `<=` the current best replaces it. So the capture prefers the call **closest to the benchmark**, not the innermost one. Among calls at the same depth, the last one wins.

!!! note
    The attribute is named `deepest_call`, but because of the `<=` it holds the shallowest frame.

## What's "meaningless"

`_is_meaningless_return` skips returns that aren't worth snapshotting:

- `None` — always skipped.
- Empty containers (`len(value) == 0` if `__len__` exists).
- Zero scalars: `0`, `0.0`, and `False` (because `bool` is a subclass of `int`).

Everything else is kept: non-zero numbers, `True`, non-empty strings, non-empty containers and arrays, and class instances without `__len__`.

## Lifecycle

```python
tracer = ExecutionTracer(max_depth=100)
tracer.start_tracing()           # sys.settrace(self._trace_calls)
try:
    benchmark()
finally:
    result = tracer.stop_tracing()  # sys.settrace(None); returns the captured TraceResult or None
```

If a traced frame raises an exception before a meaningful return value is recorded, the tracer records a result with `success = False` and `error` set. This also happens when the benchmark catches the exception. A later meaningful return at the same or a smaller depth replaces it. If nothing meaningful was traced (for example, all the work ran inside C code), `result` is `None`.

`BenchmarkRunner` changes `None` to `TraceResult(success=True, return_value=None)`: a run with no captured value is a success with empty output, not a failure.

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

`function_name` and `module_name` tell you which function the snapshot comes from. Use them to debug a snapshot that fails after a helper is renamed or moved.

## Overhead

`sys.settrace` is slow: every Python function call goes through the callback. `snapshot-tool` traces benchmarks only during `capture` / `verify` / `baseline`. ASV timing runs are separate and are not affected.

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
