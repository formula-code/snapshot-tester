# Determinism

Snapshot testing is only useful if the same code produces the same output twice. `BenchmarkRunner` reseeds the RNGs listed below before each benchmark run. The component responsible is `RNGPatcher`.

## What gets reseeded

`RNGPatcher.patch_all()` reseeds the following, in order, on every invocation:

| Library | Calls | Notes |
|---------|-------|-------|
| **Python stdlib** | `random.seed(seed)` | Always — `random` is in the stdlib. |
| **NumPy** (legacy API) | `np.random.seed(seed)` | If numpy is importable. Affects `np.random.rand`, `np.random.randn`, etc. Works with numpy 1.12 and later. |
| **PyTorch** | `torch.manual_seed(seed)`, plus `torch.cuda.manual_seed_all(seed)` if CUDA is available | If torch is importable. |
| **TensorFlow** | `tf.random.set_seed(seed)` | If tensorflow is importable. |

Default `seed=42`. `BenchmarkRunner` instantiates `RNGPatcher(seed=42)` and calls `patch_all()` before:

- Every benchmark run (via `_reset_random_state()` inside `_run_benchmark_internal`).
- Every `setup_cache()` call (once per class).
- Every `setup()` call (before each parameter combination).

## Numpy Generator API

`RNGPatcher` reseeds only the **legacy** numpy API (`np.random.seed`). It does not seed the `numpy.random.Generator` API. Each `Generator` has its own state, and `np.random.default_rng()` without a seed takes fresh entropy from the operating system, so its output changes on every run.

A benchmark that uses `Generator` gives stable snapshots only if it passes a fixed seed, for example `np.random.default_rng(0)`. This applies in `setup`, in the benchmark method, and at module import.

## Inside `BenchmarkRunner`

```python
class BenchmarkRunner:
    def __init__(self, benchmark_dir, ..., seed=42, timeout=None):
        self.rng_patcher = RNGPatcher(seed=seed)

    def _reset_random_state(self):
        self.rng_patcher.patch_all()
```

`_reset_random_state` is called:

- Once at the top of `_run_benchmark_internal`, before module load and parameter resolution.
- Again before `setup_cache()`, if the class has one.
- Again before `setup()`, if the class has one.

These calls make the reseeded RNG state the same before each user-code step in capture and in verify. **Do not remove these calls**; without them, snapshots are not reproducible.

## Module-level constants vs runtime state

`RNGPatcher` handles only RNG state. It does not handle other sources of nondeterminism:

- **Dict/set iteration order** — Python ≥ 3.7 dicts iterate insertion order; sets do not. If a benchmark returns a `set` or iterates one, the result may vary between runs.
- **Hash randomization** — `PYTHONHASHSEED` affects `hash()` for strings/bytes. Snapshots survive this if the benchmark doesn't expose hashes; if it does, set `PYTHONHASHSEED=0` (or any constant) in your environment.
- **Wall-clock time** — `datetime.now()`, `time.time()` and similar are not patched.
- **OS / hardware** — floating-point results can differ across BLAS implementations and CPU vector widths. Use `--tolerance` for this: if your `pass-to-fail` transitions are very small float differences, increase `rtol`/`atol` before you treat them as a real regression.

## Programmatic use

The simplest entry point is `reset_all_rngs`:

```python
from snapshot_tool import reset_all_rngs

reset_all_rngs(seed=42)  # one-shot: reseed Python random, numpy, torch, tensorflow
```

For longer-lived control, use the class directly:

```python
from snapshot_tool import RNGPatcher

patcher = RNGPatcher(seed=123)
patcher.patch_all()
# ... run code that consumes randomness ...

# Or as a context manager
with RNGPatcher(seed=123):
    do_random_stuff()
```

`patch_all_rngs` / `unpatch_all_rngs` wrap one global `_global_patcher`. `RNGPatcher` only seeds RNGs; it does not replace any function. So "unpatch" (and leaving the context manager) does not restore an earlier RNG state. `unpatch_all_rngs` only lets a later `patch_all_rngs(...)` call take effect.

## Quick determinism check

If you suspect a benchmark is non-deterministic despite the patcher:

```bash
# Capture, baseline and verify the same code. Expect only pass-to-pass.
snapshot-tool capture  benchmarks
snapshot-tool baseline benchmarks
snapshot-tool verify   benchmarks
jq '."pass-to-fail"' summary.json   # expect 0
```

If `pass-to-fail` is greater than 0 here, the benchmark reads entropy from a source that `RNGPatcher` does not seed. Common causes: an unseeded `numpy.random.Generator`, `os.urandom`, `secrets`, network or filesystem timestamps, and the order of sets of strings when `PYTHONHASHSEED` is not set.

## Next steps

- [Tracing](tracing.md) — Why deterministic execution matters for what the tracer captures.
- [Baseline & Verify](baseline-and-verify.md) — How `pass-to-fail` cells are produced.
