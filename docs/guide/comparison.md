# Comparison

`Comparator` decides whether an actual value matches an expected snapshot. It is type-aware, tolerance-aware, and **never** depends on numpy at import time — numpy is detected lazily and probed by module/type-name so the package works in pure-Python environments.

## Configuration

```python
from snapshot_tool import Comparator, ComparisonConfig

config = ComparisonConfig(
    rtol=1e-5,           # relative tolerance
    atol=1e-8,           # absolute tolerance
    equal_nan=False,     # if True, NaN == NaN
    strict_types=True,   # error on dtype mismatches for numpy arrays
    strict_shapes=True,  # error on shape mismatches for numpy arrays
    ignore_order=False,  # (reserved; not currently honored by built-in strategies)
)

comparator = Comparator(config)
result = comparator.compare(actual, expected)
```

| Field | Default | What it does |
|-------|---------|--------------|
| `rtol` | `1e-5` | Relative tolerance applied to `\|a - b\| <= atol + rtol * \|b\|` |
| `atol` | `1e-8` | Absolute tolerance, same formula |
| `equal_nan` | `False` | If `True`, two NaN values compare equal |
| `strict_types` | `True` | Numpy array `dtype` mismatch is a failure |
| `strict_shapes` | `True` | Numpy array `shape` mismatch is a failure |

The CLI exposes `rtol` and `atol` via `--tolerance RTOL ATOL`. The remaining fields are configurable only through `snapshot_config.json` or programmatic use.

## The result

```python
@dataclass
class ComparisonResult:
    match: bool
    skipped: bool = False
    tolerance_used: Optional[dict[str, float]] = None
    error_message: Optional[str] = None
    details: Optional[dict[str, Any]] = None
```

Three outcomes:

| Outcome | `match` | `skipped` | When |
|---------|---------|-----------|------|
| Pass | `True` | `False` | Values match within tolerance |
| Skip | `True` | `True` | The expected value was a placeholder that can't be compared (a generator, callable, or unpicklable marker), or the type has no usable `__eq__` |
| Fail | `False` | `False` | Concrete mismatch — `error_message` and `details` describe what differs |

`verify` treats `skipped=True` results as `[SKIP]` (not a failure) and only fails the run on `match=False, skipped=False`.

## Dispatch order

`compare()` runs through a fixed strategy chain and returns the first result that isn't `None`:

1. **Serialized placeholders** — If `expected` is a dict tagged `__generator__`, `__callable__`, or `__unpicklable__`, the comparison is skipped. These tags come from [`SnapshotManager._serialize_value`](storage.md#serialization-of-unpicklable-values), which writes them when the captured value can't round-trip through pickle.
2. **Serialized class instance** — If `expected` is tagged `__class_instance__`, the comparator checks `__class__.__name__`, then recursively compares each attribute in the saved `__dict__` against `actual.__dict__`. Extra or missing attributes fail.
3. **`None` handling** — Both `None` → match. One `None` → fail.
4. **Numpy arrays** — If both sides are arrays:
    - Shapes are compared (failure under `strict_shapes`).
    - Dtypes are compared (failure under `strict_types`).
    - `object` dtype arrays go through `_compare_object_arrays`, which recursively compares each element with `compare()` (so a shapely-array-of-polygons works).
    - Numeric arrays are flattened and compared element-wise with the pure-Python `_py_isclose`. The result includes `max_difference`, `mean_difference`, `shape`, and `dtype` for diagnostics.
5. **Numeric scalars** — `int`/`float` and numpy scalar types. Compared with `_py_isclose`.
6. **Objects with `__eq__`** — If `type(actual) == type(expected)` and `__eq__` is defined somewhere on the MRO (not just the default `object.__eq__`), `==` is called. The result may itself be a numpy array (e.g., `astropy.coordinates.SkyCoord` returns a bool array from `__eq__`), in which case `.all()` reduces it to a single bool. Lists/tuples of arrays are handled similarly.
7. **Sequences** (list, tuple) — Length match, then element-wise recursion. The first ten mismatch indices and their messages are retained in `details["mismatches"]`.
8. **Dicts** — Key-set match, then per-key recursion. Like sequences, up to ten mismatches are retained.
9. **Fallback `==`** — Type-check first, then `actual == expected`. If the comparison itself raises (e.g., `ValueError("ambiguous truth value")` from accidentally calling `bool()` on an array), the failure is converted to a **skip** with `details["reason"]` explaining why.

The order matters. `_compare_objects` runs before `_compare_sequences` because many "object-like" types (pandas `Series`, astropy `SkyCoord`) have `__len__` and `__getitem__` but want to be compared via their own `__eq__`, not element-by-element.

## Scalar comparison

Numeric scalars use `_py_isclose` — a pure-Python mirror of `numpy.isclose`:

```python
def _py_isclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False):
    # NaN handling
    if equal_nan and isnan(a) and isnan(b): return True
    if isnan(a) or isnan(b): return False
    # Infinity: inf == inf, -inf == -inf, inf != -inf
    if isinf(a) or isinf(b): return a == b
    return abs(a - b) <= atol + rtol * abs(b)
```

The formula is **asymmetric** in `b` — that's intentional; it matches numpy's semantics. In `snapshot-tool`, `b` is always the *expected* (snapshot) value, so the tolerance scales with the magnitude of what you originally captured.

## Numpy array comparison without importing numpy

`Comparator` never imports numpy unconditionally. Detection is done with:

```python
def _is_numpy_array(obj):
    if HAS_NUMPY:
        return isinstance(obj, np.ndarray)
    obj_type = type(obj)
    return obj_type.__module__ == 'numpy' and obj_type.__name__ == 'ndarray'
```

If numpy isn't installed, array snapshots round-trip as the placeholder representations produced by `SnapshotManager` (e.g., `__class_instance__` for object arrays) and are compared accordingly. In practice, if you have numpy-array snapshots you almost certainly have numpy installed — but the harness doesn't assume it.

## Failure diagnostics

Failed comparisons populate `error_message` and, where useful, `details`:

- **Numpy arrays**: `details = {"max_difference": ..., "mean_difference": ..., "shape": ..., "dtype": ...}`
- **Sequences/dicts**: `details = {"mismatches": [(index_or_key, message), ...]}` (capped at 10)
- **Scalars**: `details = {"difference": abs(actual - expected)}`
- **Classes via fallback**: `details = {"type": ..., "reason": ..., "skipped": True}` when comparison is unsupported

`verify` prints `error_message` on `[FAIL]` and the `details` dict when `--verbose` is set.

## Configuring tolerance per-run

The CLI accepts the two most important knobs directly:

```bash
snapshot-tool verify benchmarks --tolerance 1e-4 1e-6
```

For everything else (`equal_nan`, `strict_types`, `strict_shapes`), set them in `snapshot_config.json` — see [Configuration](configuration.md).

## Next steps

- [Snapshot Storage](storage.md) — How values are serialized and what placeholders look like on disk.
- [Baseline & Verify](baseline-and-verify.md) — How comparison outcomes roll up into the 3×3 transition matrix.
