"""Compare captured outputs with stored snapshots, with numerical tolerances in pure Python."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

# numpy is optional; it is used only to detect numpy types
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False


def _is_numpy_array(obj: Any) -> bool:
    """Check if object is a numpy array without requiring numpy import."""
    if HAS_NUMPY and np is not None:
        return isinstance(obj, np.ndarray)
    obj_type = type(obj)
    return obj_type.__module__ == "numpy" and obj_type.__name__ == "ndarray"


def _is_pandas_dataframe(obj: Any) -> bool:
    """Detect a pandas DataFrame without importing pandas (it's optional)."""
    t = type(obj)
    return t.__name__ == "DataFrame" and t.__module__.split(".")[0] == "pandas"


def _is_pandas_series(obj: Any) -> bool:
    """Detect a pandas Series without importing pandas (it's optional)."""
    t = type(obj)
    return t.__name__ == "Series" and t.__module__.split(".")[0] == "pandas"


def _is_numpy_scalar(obj: Any) -> bool:
    """Check if object is a numpy scalar without requiring numpy import."""
    if HAS_NUMPY and np is not None:
        return isinstance(obj, np.number)
    obj_type = type(obj)
    return obj_type.__module__ == "numpy"


def _py_isclose(
    a: float, b: float, rtol: float = 1e-5, atol: float = 1e-8, equal_nan: bool = False
) -> bool:
    """Pure Python implementation of numpy.isclose for scalars."""
    try:
        a_float = float(a)
        b_float = float(b)
        a_is_nan = math.isnan(a_float)
        b_is_nan = math.isnan(b_float)
        a_is_inf = math.isinf(a_float)
        b_is_inf = math.isinf(b_float)
    except (TypeError, ValueError):
        return False

    if equal_nan and a_is_nan and b_is_nan:
        return True
    if a_is_nan or b_is_nan:
        return False

    # Handle infinity: inf == inf, -inf == -inf, but inf != -inf
    if a_is_inf or b_is_inf:
        return a_float == b_float

    # |a - b| <= atol + rtol * |b|, as in numpy.isclose
    return abs(a_float - b_float) <= atol + rtol * abs(b_float)


@dataclass
class ComparisonResult:
    """Result of comparing two values."""

    match: bool
    skipped: bool = False  # placeholder value (generator, callable, unpicklable)
    tolerance_used: Optional[dict[str, float]] = None
    error_message: Optional[str] = None
    details: Optional[dict[str, Any]] = None


@dataclass
class ComparisonConfig:
    """Configuration for comparison operations."""

    rtol: float = 1e-5
    atol: float = 1e-8
    # A NaN that appears again at the same position is not a change (numpy.isclose uses False)
    equal_nan: bool = True
    strict_types: bool = True
    strict_shapes: bool = True
    ignore_order: bool = False  # not used by any strategy


class Comparator:
    """Compares values with configurable tolerances."""

    def __init__(self, config: Optional[ComparisonConfig] = None):
        self.config = config or ComparisonConfig()

    def compare(self, actual: Any, expected: Any) -> ComparisonResult:
        """Compare actual value with expected value."""
        try:
            if isinstance(expected, dict) and expected.get("__generator__"):
                return ComparisonResult(
                    match=True,
                    skipped=True,
                    details="Skipped comparison for generator (cannot be pickled)",
                )

            if isinstance(expected, dict) and expected.get("__callable__"):
                return ComparisonResult(
                    match=True,
                    skipped=True,
                    details="Skipped comparison for callable (cannot be pickled reliably)",
                )

            if isinstance(expected, dict) and expected.get("__unpicklable__"):
                return ComparisonResult(
                    match=True,
                    skipped=True,
                    details="Skipped comparison for unpicklable value (stored as placeholder)",
                )

            if isinstance(expected, dict) and expected.get("__class_instance__"):
                return self._compare_class_instance(actual, expected)

            if actual is None and expected is None:
                return ComparisonResult(match=True)
            elif actual is None or expected is None:
                return ComparisonResult(
                    match=False,
                    error_message=f"One value is None: actual={actual}, expected={expected}",
                )

            # _compare_objects comes before _compare_sequences: SkyCoord and similar types
            # look like sequences but must use their own __eq__
            strategies = [
                self._compare_numpy_arrays,
                self._compare_scalars,
                self._compare_pandas,
                self._compare_objects,
                self._compare_sequences,
                self._compare_dicts,
                self._compare_fallback,
            ]

            for strategy in strategies:
                result = strategy(actual, expected)
                if result is not None:
                    return result

            return ComparisonResult(
                match=False, error_message="No comparison strategy could handle these types"
            )

        except Exception as e:
            return ComparisonResult(
                match=False, error_message=f"Comparison failed with exception: {e}"
            )

    def _compare_class_instance(self, actual: Any, expected: dict[str, Any]) -> ComparisonResult:
        """Compare a class instance with a serialized class instance."""
        if not hasattr(actual, "__class__"):
            return ComparisonResult(
                match=False, error_message=f"Expected class instance but got {type(actual)}"
            )

        # Only the class name is compared, not the module
        expected_class = expected["__class_name__"]
        actual_class = actual.__class__.__name__

        if actual_class != expected_class:
            return ComparisonResult(
                match=False,
                error_message=f"Class name mismatch: expected {expected_class}, got {actual_class}",
            )

        expected_dict = expected["__dict__"]
        actual_dict = actual.__dict__

        for key, expected_value in expected_dict.items():
            if key not in actual_dict:
                return ComparisonResult(
                    match=False, error_message=f"Missing attribute '{key}' in actual instance"
                )

            actual_value = actual_dict[key]
            attr_comparison = self.compare(actual_value, expected_value)
            if not attr_comparison.match:
                return ComparisonResult(
                    match=False,
                    error_message=f"Attribute '{key}' mismatch: {attr_comparison.error_message}",
                )

        for key in actual_dict:
            if key not in expected_dict:
                return ComparisonResult(
                    match=False, error_message=f"Extra attribute '{key}' in actual instance"
                )

        return ComparisonResult(match=True)

    def _compare_numpy_arrays(self, actual: Any, expected: Any) -> Optional[ComparisonResult]:
        """Compare numpy arrays using pure Python iteration."""
        if not (_is_numpy_array(actual) and _is_numpy_array(expected)):
            return None

        actual_shape = getattr(actual, "shape", None)
        expected_shape = getattr(expected, "shape", None)
        actual_dtype = getattr(actual, "dtype", None)
        expected_dtype = getattr(expected, "dtype", None)

        if self.config.strict_shapes and actual_shape != expected_shape:
            return ComparisonResult(
                match=False,
                error_message=f"Array shapes differ: {actual_shape} vs {expected_shape}",
            )

        if self.config.strict_types and actual_dtype != expected_dtype:
            return ComparisonResult(
                match=False,
                error_message=f"Array dtypes differ: {actual_dtype} vs {expected_dtype}",
            )

        # Handle object arrays (like Shapely geometry arrays) - compare element-wise.
        # `dtype == object` is the idiomatic numpy object-dtype check; `is` would
        # be wrong here (a dtype is not the `object` type itself).
        if actual_dtype == object or expected_dtype == object:  # noqa: E721
            return self._compare_object_arrays(actual, expected)

        try:
            actual_flat = actual.flatten()
            expected_flat = expected.flatten()

            differences = []
            all_close = True

            for a_val, e_val in zip(actual_flat, expected_flat):
                try:
                    a_float = float(a_val)
                    e_float = float(e_val)
                except (TypeError, ValueError):
                    if a_val != e_val:
                        all_close = False
                        break
                    continue

                if not _py_isclose(
                    a_float, e_float, self.config.rtol, self.config.atol, self.config.equal_nan
                ):
                    all_close = False
                    differences.append(abs(a_float - e_float))

            tolerance_used = {
                "rtol": self.config.rtol,
                "atol": self.config.atol,
                "equal_nan": self.config.equal_nan,
            }

            if not all_close:
                max_diff = max(differences) if differences else 0
                mean_diff = sum(differences) / len(differences) if differences else 0

                details = {
                    "max_difference": float(max_diff),
                    "mean_difference": float(mean_diff),
                    "shape": actual_shape,
                    "dtype": str(actual_dtype),
                }

                return ComparisonResult(
                    match=False,
                    tolerance_used=tolerance_used,
                    error_message=f"Arrays not close: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}",
                    details=details,
                )
            else:
                return ComparisonResult(match=True, tolerance_used=tolerance_used)

        except Exception as e:
            return ComparisonResult(match=False, error_message=f"Array comparison failed: {e}")

    def _compare_scalars(self, actual: Any, expected: Any) -> Optional[ComparisonResult]:
        """Compare scalar values using pure Python."""
        if not (self._is_numeric_scalar(actual) and self._is_numeric_scalar(expected)):
            return None

        try:
            match = _py_isclose(
                float(actual),
                float(expected),
                rtol=self.config.rtol,
                atol=self.config.atol,
                equal_nan=self.config.equal_nan,
            )

            tolerance_used = {
                "rtol": self.config.rtol,
                "atol": self.config.atol,
                "equal_nan": self.config.equal_nan,
            }

            if not match:
                diff = abs(float(actual) - float(expected))
                return ComparisonResult(
                    match=False,
                    tolerance_used=tolerance_used,
                    error_message=f"Scalars not close: {actual} vs {expected}, diff={diff:.2e}",
                    details={"difference": diff},
                )
            else:
                return ComparisonResult(match=True, tolerance_used=tolerance_used)

        except Exception as e:
            return ComparisonResult(match=False, error_message=f"Scalar comparison failed: {e}")

    def _compare_pandas(self, actual: Any, expected: Any) -> Optional[ComparisonResult]:
        """Compare pandas DataFrame / Series with tolerance.

        Runs before _compare_objects, where ``bool(actual == expected)`` raises for pandas.
        Both sides are converted with to_numpy() and compared as arrays.
        """
        a_df, a_sr = _is_pandas_dataframe(actual), _is_pandas_series(actual)
        e_df, e_sr = _is_pandas_dataframe(expected), _is_pandas_series(expected)
        if not ((a_df or a_sr) and (e_df or e_sr)):
            return None
        if a_df != e_df or a_sr != e_sr:
            return ComparisonResult(
                match=False,
                error_message=(
                    f"pandas type mismatch: {type(actual).__name__} vs {type(expected).__name__}"
                ),
            )
        try:
            actual_np = actual.to_numpy()
            expected_np = expected.to_numpy()
        except Exception as e:
            return ComparisonResult(match=False, error_message=f"pandas comparison failed: {e}")
        np_result = self._compare_numpy_arrays(actual_np, expected_np)
        if np_result is None:
            return ComparisonResult(
                match=False,
                error_message="pandas values could not be compared as arrays",
            )
        return np_result

    def _compare_sequences(self, actual: Any, expected: Any) -> Optional[ComparisonResult]:
        """Compare sequences (lists, tuples, etc.)."""
        if not (self._is_sequence(actual) and self._is_sequence(expected)):
            return None

        if len(actual) != len(expected):
            return ComparisonResult(
                match=False,
                error_message=f"Sequence lengths differ: {len(actual)} vs {len(expected)}",
            )

        mismatches = []
        for i, (a, e) in enumerate(zip(actual, expected)):
            result = self.compare(a, e)
            if not result.match:
                mismatches.append((i, result.error_message))

        if mismatches:
            return ComparisonResult(
                match=False,
                error_message=f"Sequence elements differ at indices: {[i for i, _ in mismatches[:5]]}",
                details={"mismatches": mismatches[:10]},
            )
        else:
            return ComparisonResult(match=True)

    def _compare_dicts(self, actual: Any, expected: Any) -> Optional[ComparisonResult]:
        """Compare dictionaries."""
        if not (isinstance(actual, dict) and isinstance(expected, dict)):
            return None

        actual_keys = set(actual.keys())
        expected_keys = set(expected.keys())

        if actual_keys != expected_keys:
            missing = expected_keys - actual_keys
            extra = actual_keys - expected_keys
            return ComparisonResult(
                match=False, error_message=f"Dict keys differ: missing={missing}, extra={extra}"
            )

        mismatches = []
        for key in actual_keys:
            result = self.compare(actual[key], expected[key])
            if not result.match:
                mismatches.append((key, result.error_message))

        if mismatches:
            return ComparisonResult(
                match=False,
                error_message=f"Dict values differ for keys: {[k for k, _ in mismatches[:5]]}",
                details={"mismatches": mismatches[:10]},
            )
        else:
            return ComparisonResult(match=True)

    def _compare_objects(self, actual: Any, expected: Any) -> Optional[ComparisonResult]:
        """Compare any value that is not a list, tuple or dict, using its custom __eq__."""
        if isinstance(actual, (list, tuple, dict)):
            return None

        actual_type = type(actual)
        expected_type = type(expected)

        if actual_type != expected_type:
            return ComparisonResult(
                match=False,
                error_message=f"Type mismatch: {actual_type.__name__} vs {expected_type.__name__}",
            )

        # object.__eq__ does not count as a custom __eq__
        has_custom_eq = False
        for cls in actual_type.__mro__:
            if "__eq__" in cls.__dict__:
                has_custom_eq = True
                break

        if not has_custom_eq:
            # Counts as a match; details records the skip
            return ComparisonResult(
                match=True,
                details={
                    "skipped": True,
                    "reason": f"Type {actual_type.__name__} has no custom __eq__ method",
                    "type": actual_type.__name__,
                },
                error_message=f"Skipped comparison: {actual_type.__name__} has no custom __eq__",
            )

        try:
            match = actual == expected
            if match is NotImplemented:
                return ComparisonResult(
                    match=True,
                    details={
                        "skipped": True,
                        "reason": f"__eq__ returned NotImplemented for {actual_type.__name__}",
                        "type": actual_type.__name__,
                    },
                    error_message=f"Skipped: __eq__ not implemented for {actual_type.__name__}",
                )

            # __eq__ can return an array (e.g. SkyCoord) or a list of arrays
            if _is_numpy_array(match):
                if hasattr(match, "all"):
                    match = bool(match.all())
                else:
                    match = all(bool(x) for x in match.flatten())
            elif isinstance(match, (list, tuple)) and len(match) > 0:
                if _is_numpy_array(match[0]):
                    match = all(
                        (arr.all() if hasattr(arr, "all") else bool(arr))
                        if _is_numpy_array(arr)
                        else arr
                        for arr in match
                    )

            return ComparisonResult(
                match=match,
                error_message=None if match else f"Objects not equal: {actual} vs {expected}",
            )
        except Exception as e:
            return ComparisonResult(
                match=False,
                error_message=f"Object comparison failed: {e}",
                details={"type": actual_type.__name__},
            )

    def _compare_fallback(self, actual: Any, expected: Any) -> Optional[ComparisonResult]:
        """Fallback comparison using == operator."""
        try:
            if type(actual) is not type(expected):
                return ComparisonResult(
                    match=False,
                    error_message=f"Type mismatch: {type(actual).__name__} vs {type(expected).__name__}",
                )

            match = actual == expected

            if match is NotImplemented:
                return ComparisonResult(
                    match=True,
                    details={
                        "skipped": True,
                        "reason": f"Comparison not supported for {type(actual).__name__}",
                        "type": type(actual).__name__,
                    },
                    error_message=f"Skipped: comparison not supported for {type(actual).__name__}",
                )

            return ComparisonResult(
                match=match,
                error_message=None if match else f"Values not equal: {actual} vs {expected}",
            )
        except Exception as e:
            # Counts as a match; details records the skip
            return ComparisonResult(
                match=True,
                details={
                    "skipped": True,
                    "reason": f"Comparison failed: {e}",
                    "type": type(actual).__name__,
                },
                error_message=f"Skipped: comparison failed for {type(actual).__name__}: {e}",
            )

    def _compare_object_arrays(self, actual: Any, expected: Any) -> ComparisonResult:
        """Compare numpy object arrays element-wise."""
        actual_flat = actual.flatten()
        expected_flat = expected.flatten()

        mismatches = []
        for i, (a, e) in enumerate(zip(actual_flat, expected_flat)):
            result = self.compare(a, e)
            if not result.match:
                mismatches.append((i, result.error_message))
                if len(mismatches) >= 10:
                    break

        if mismatches:
            return ComparisonResult(
                match=False,
                error_message=f"Object array elements differ at indices: {[i for i, _ in mismatches[:5]]}",
                details={"mismatches": mismatches, "total_elements": len(actual_flat)},
            )
        else:
            return ComparisonResult(
                match=True,
                details={"array_type": "object", "total_elements": len(actual_flat)},
            )

    def _is_numeric_scalar(self, value: Any) -> bool:
        """Check if value is a numeric scalar."""
        if isinstance(value, (int, float)):
            return True
        if _is_numpy_scalar(value):
            return not hasattr(value, "shape") or value.shape == ()
        return False

    def _is_sequence(self, value: Any) -> bool:
        """Check if value is a sequence (but not a string or dict)."""
        return (
            hasattr(value, "__len__")
            and hasattr(value, "__getitem__")
            and not isinstance(value, (str, bytes, dict))
        )

    def compare_multiple(
        self, actual_values: list[Any], expected_values: list[Any]
    ) -> list[ComparisonResult]:
        """Compare multiple pairs of values."""
        if len(actual_values) != len(expected_values):
            raise ValueError("Lists must have the same length")

        return [
            self.compare(actual, expected)
            for actual, expected in zip(actual_values, expected_values)
        ]

    def get_summary_stats(self, results: list[ComparisonResult]) -> dict[str, Any]:
        """Get summary statistics for a list of comparison results."""
        total = len(results)
        matches = sum(1 for r in results if r.match)
        failures = total - matches

        return {
            "total": total,
            "matches": matches,
            "failures": failures,
            "success_rate": matches / total if total > 0 else 0,
            "failure_rate": failures / total if total > 0 else 0,
        }
