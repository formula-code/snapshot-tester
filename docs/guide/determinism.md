# Determinism

Snapshot testing is only useful if the same code produces the same output twice. `BenchmarkRunner` enforces this by reseeding every known RNG before each benchmark invocation. The component responsible is `RNGPatcher`.

## What gets reseeded

`RNGPatcher.patch_all()` reseeds the following, in order, on every invocation:

| Library | Calls | Notes |
|---------|-------|-------|
| **Python stdlib** | `random.seed(seed)` | Always — `random` is in the stdlib. |
| **NumPy** (legacy API) | `np.random.seed(seed)` | If numpy is importable. Affects `np.random.rand`, `np.random.randn`, etc. Compatible with numpy 1.12+ (2017). |
| **PyTorch** | `torch.manual_seed(seed)`, plus `torch.cuda.manual_seed_all(seed)` if CUDA is available | If torch is importable. |
| **TensorFlow** | `tf.random.set_seed(seed)` | If tensorflow is importable. |

Default `seed=42`. `BenchmarkRunner` instantiates `RNGPatcher(seed=42)` and calls `patch_all()` before:

- Every benchmark run (via `_reset_random_state()` inside `_run_benchmark_internal`).
- Every `setup_cache()` call (once per class).
- Every `setup()` call (before each parameter combination).

## Numpy Generator API

`RNGPatcher` reseeds the **legacy** numpy API (`np.random.seed`). The newer `numpy.random.Generator` API (`np.random.default_rng()`) is **not** patched automatically — each `Generator` instance carries its own state and is created independently.

If a benchmark uses `default_rng()`, the same call inside `setup` or the benchmark method will produce the same output **as long as no other code is producing entropy in between**. This is usually fine because `patch_all` runs immediately before `setup`/method execution, so the import-time random state is irrelevant.

If you have a benchmark that creates a `Generator` *outside* the setup path (e.g., at module import), pin its seed explicitly in the benchmark code — `RNGPatcher` can't reach module-level state that was already instantiated.

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

The whole point of running it three times is to guarantee that the RNG state immediately before any user-code path is identical between capture and verify. **Do not bypass these calls** — bypassing them is exactly equivalent to making the snapshots non-reproducible.

## Module-level constants vs runtime state

`RNGPatcher` only handles RNG state. Other sources of nondeterminism are out of scope:

- **Dict/set iteration order** — Python ≥ 3.7 dicts iterate insertion order; sets do not. If a benchmark returns a `set` or iterates one, the result may vary between runs.
- **Hash randomization** — `PYTHONHASHSEED` affects `hash()` for strings/bytes. Snapshots survive this if the benchmark doesn't expose hashes; if it does, set `PYTHONHASHSEED=0` (or any constant) in your environment.
- **Wall-clock time** — `datetime.now()`, `time.time()` and similar are not patched.
- **OS / hardware** — floating-point semantics can differ across BLAS implementations and CPU vector widths. `--tolerance` is the primary mitigation; if your `pass-to-fail` transitions are tiny float drifts, loosen `rtol`/`atol` before assuming a real regression.

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

There's also `patch_all_rngs` / `unpatch_all_rngs` — these wrap a single global `_global_patcher`. The "unpatch" path doesn't actually restore the prior RNG state (you can't; it's already been consumed) — it just allows another `patch_all_rngs(...)` call to take effect.

## Quick determinism check

If you suspect a benchmark is non-deterministic despite the patcher:

```bash
# Capture and verify twice. The second verify should be 100% pass-to-pass.
snapshot-tool capture  benchmarks
snapshot-tool baseline benchmarks
snapshot-tool verify   benchmarks
jq '."pass-to-fail"' summary.json   # expect 0
```

If you see `pass-to-fail` > 0 on this self-comparison, the benchmark is reading entropy from a source `RNGPatcher` doesn't cover. Common culprits: a `Generator` instantiated at import time, `os.urandom`, `secrets`, network/filesystem timestamps, dict ordering of stringly-keyed maps with `PYTHONHASHSEED` unset.

## Next steps

- [Tracing](tracing.md) — Why deterministic execution matters for what the tracer captures.
- [Baseline & Verify](baseline-and-verify.md) — How `pass-to-fail` cells are produced.
