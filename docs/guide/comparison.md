# Comparison

`Comparator` decides whether an actual value matches an expected snapshot. It checks types and applies numerical tolerances. Numpy is optional: `comparator.py` imports it if it is installed, and otherwise detects numpy types by module and type name.

## Configuration

```python
from snapshot_tool import Comparator, ComparisonConfig

config = ComparisonConfig(
    rtol=1e-5,           # relative tolerance
    atol=1e-8,           # absolute tolerance
    equal_nan=True,      # NaN at same position == NaN (snapshot default)
    strict_types=True,   # error on dtype mismatches for numpy arrays
    strict_shapes=True,  # error on shape mismatches for numpy arrays
    ignore_order=False,  # not used by any strategy
)

comparator = Comparator(config)
result = comparator.compare(actual, expected)
```

| Field | Default | What it does |
|-------|---------|--------------|
| `rtol` | `1e-5` | Relative tolerance applied to `\|a - b\| <= atol + rtol * \|b\|` |
| `atol` | `1e-8` | Absolute tolerance, same formula |
| `equal_nan` | `True` | Two NaN values at the same position compare equal. Defaults `True` because, for snapshot testing, a deterministic NaN that reappears unchanged is not a regression. Set `False` for `numpy.isclose` semantics. |
| `strict_types` | `True` | Numpy array `dtype` mismatch is a failure |
| `strict_shapes` | `True` | Numpy array `shape` mismatch is a failure |

The CLI sets `rtol` and `atol` from `--tolerance RTOL ATOL` or from `snapshot_config.json`, and `equal_nan` from the file. `strict_types`, `strict_shapes` and `ignore_order` can be set only from Python.

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
| Skip | `True` | `True` | The expected value is a placeholder that cannot be compared (a generator, a callable, or an unpicklable marker) |
| Fail | `False` | `False` | A real mismatch; `error_message` and `details` describe what differs |

`verify` reports `skipped=True` results as `[SKIP]` and fails the run only on `match=False`.

Some results have `match=True` and `skipped=False` but set `details["skipped"] = True`: a type with no custom `__eq__`, an `__eq__` that returns `NotImplemented`, or an `==` that raises in the fallback strategy. `verify` counts these as `[PASS]`.

## Dispatch order

`compare()` runs through a fixed strategy chain and returns the first result that isn't `None`:

1. **Serialized placeholders** — If `expected` is a dict tagged `__generator__`, `__callable__`, or `__unpicklable__`, the comparison is skipped. These tags come from [`serialize_value`](storage.md#serialization-of-unpicklable-values), which writes them when the captured value cannot round-trip through pickle.
2. **Serialized class instance** — If `expected` is tagged `__class_instance__`, the comparator checks `__class__.__name__`, then recursively compares each attribute in the saved `__dict__` against `actual.__dict__`. Extra or missing attributes fail.
3. **`None` handling** — Both `None` → match. One `None` → fail.
4. **Numpy arrays** — If both sides are arrays:
    - Shapes are compared (failure under `strict_shapes`).
    - Dtypes are compared (failure under `strict_types`).
    - `object` dtype arrays go through `_compare_object_arrays`, which recursively compares each element with `compare()` (so a shapely-array-of-polygons works).
    - Numeric arrays are flattened and compared element-wise with the pure-Python `_py_isclose`. The result includes `max_difference`, `mean_difference`, `shape`, and `dtype` for diagnostics.
5. **Numeric scalars** — `int`/`float` and numpy scalar types. Compared with `_py_isclose`.
6. **pandas `DataFrame` / `Series`** — Both sides must be the same pandas type. They are converted with `to_numpy()` and compared as numpy arrays (step 4).
7. **Objects** — For any value that is not a `list`, `tuple` or `dict`: if the two types differ, this is a failure. If the type has no custom `__eq__` on its MRO (the default `object.__eq__` does not count), the result is a match with `details["skipped"] = True`. Otherwise `==` is called. The result may itself be a numpy array (e.g., `astropy.coordinates.SkyCoord` returns a bool array from `__eq__`), in which case `.all()` reduces it to a single bool. Lists/tuples of arrays are handled similarly.
8. **Sequences** (list, tuple) — Length match, then element-wise recursion. The first ten mismatch indices and their messages are kept in `details["mismatches"]`.
9. **Dicts** — Key-set match, then per-key recursion. Like sequences, up to ten mismatches are kept.
10. **Fallback `==`** — Type check first, then `actual == expected`. If the comparison raises, the result is a match with `details["skipped"] = True` and a `details["reason"]`.

The order matters. `_compare_pandas` runs before `_compare_objects` because `bool(df == other)` raises for pandas objects. `_compare_objects` runs before `_compare_sequences` because types such as astropy `SkyCoord` have `__len__` and `__getitem__` but must be compared with their own `__eq__`, not element by element. In practice step 7 handles every value that is not a `list`, `tuple` or `dict`, so steps 8 to 10 see only those built-in types.

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

The helper's own `equal_nan` parameter defaults to `False` (mirroring `numpy.isclose`), but `Comparator` always passes it explicitly from `ComparisonConfig.equal_nan`, which **defaults to `True`** — so in practice scalar and array NaNs compare equal unless you opt into strict semantics.

The formula is **asymmetric** in `b`, as in numpy. In `snapshot-tool`, `b` is always the *expected* (snapshot) value, so the tolerance scales with the magnitude of what you originally captured.

## Numpy detection

`comparator.py` tries `import numpy` once at import time. Detection then works with or without numpy:

```python
def _is_numpy_array(obj):
    if HAS_NUMPY:
        return isinstance(obj, np.ndarray)
    obj_type = type(obj)
    return obj_type.__module__ == 'numpy' and obj_type.__name__ == 'ndarray'
```

A snapshot that holds numpy arrays cannot be unpickled without numpy. `load_snapshot` then logs a warning and returns `None`, and `verify` counts the test as skipped.

## Failure diagnostics

Failed comparisons populate `error_message` and, where useful, `details`:

- **Numpy arrays**: `details = {"max_difference": ..., "mean_difference": ..., "shape": ..., "dtype": ...}`
- **Sequences/dicts**: `details = {"mismatches": [(index_or_key, message), ...]}` (capped at 10)
- **Scalars**: `details = {"difference": abs(actual - expected)}`
- **Comparisons that cannot run** (steps 7 and 10): `details = {"type": ..., "reason": ..., "skipped": True}` on a result that counts as a match

`verify` prints `error_message` on `[FAIL]`. With `--verbose` it also logs `details` at DEBUG level, which the default log setup does not show.

## Configuring tolerance per-run

The CLI accepts `rtol` and `atol` directly:

```bash
snapshot-tool verify benchmarks --tolerance 1e-4 1e-6
```

Set `equal_nan` in `snapshot_config.json` (see [Configuration](configuration.md)). `strict_types` and `strict_shapes` can be set only from Python.

## Next steps

- [Snapshot Storage](storage.md) — How values are serialized and what placeholders look like on disk.
- [Baseline & Verify](baseline-and-verify.md) — How comparison outcomes roll up into the 3×3 transition matrix.
