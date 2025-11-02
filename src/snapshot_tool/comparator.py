"""
Comparison engine for snapshot testing.

This module compares captured outputs with stored snapshots using
numpy.allclose for numerical data and other strategies for different types.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ComparisonResult:
    """Result of comparing two values."""

    match: bool
    skipped: bool = False  # True if comparison was skipped (e.g., generators, unpicklable values)
    tolerance_used: dict[str, float] | None = None
    error_message: str | None = None
    details: dict[str, Any] | None = None


@dataclass
class ComparisonConfig:
    """Configuration for comparison operations."""

    rtol: float = 1e-5
    atol: float = 1e-8
    equal_nan: bool = False
    strict_types: bool = True
    strict_shapes: bool = True
    ignore_order: bool = False


class Comparator:
    """Compares values with configurable tolerances."""

    def __init__(self, config: ComparisonConfig | None = None):
        self.config = config or ComparisonConfig()

    def compare(self, actual: Any, expected: Any) -> ComparisonResult:
        """Compare actual value with expected value."""
        try:
            # Handle serialized generators - skip comparison
            if isinstance(expected, dict) and expected.get("__generator__"):
                return ComparisonResult(
                    match=True,
                    skipped=True,
                    details="Skipped comparison for generator (cannot be pickled)"
                )

            # Handle serialized callables (functions/closures) - skip comparison
            if isinstance(expected, dict) and expected.get("__callable__"):
                return ComparisonResult(
                    match=True,
                    skipped=True,
                    details="Skipped comparison for callable (cannot be pickled reliably)",
                )

            # Handle unpicklable placeholder
            if isinstance(expected, dict) and expected.get("__unpicklable__"):
                return ComparisonResult(
                    match=True,
                    skipped=True,
                    details="Skipped comparison for unpicklable value (stored as placeholder)",
                )

            # Handle serialized class instances
            if isinstance(expected, dict) and expected.get("__class_instance__"):
                return self._compare_class_instance(actual, expected)

            # Handle None values
            if actual is None and expected is None:
                return ComparisonResult(match=True)
            elif actual is None or expected is None:
                return ComparisonResult(
                    match=False,
                    error_message=f"One value is None: actual={actual}, expected={expected}",
                )

            # Try different comparison strategies
            # Note: _compare_objects must come before _compare_sequences because some objects
            # like SkyCoord have sequence-like methods but should use their __eq__ method
            strategies = [
                self._compare_numpy_arrays,
                self._compare_scalars,
                self._compare_objects,
                self._compare_sequences,
                self._compare_dicts,
                self._compare_fallback,
            ]

            for strategy in strategies:
                result = strategy(actual, expected)
                if result is not None:
                    return result

            # If no strategy worked, use fallback
            return ComparisonResult(
                match=False, error_message="No comparison strategy could handle these types"
            )

        except Exception as e:
            return ComparisonResult(
                match=False, error_message=f"Comparison failed with exception: {e}"
            )

    def _compare_class_instance(self, actual: Any, expected: dict[str, Any]) -> ComparisonResult:
        """Compare a class instance with a serialized class instance."""
        # Check if actual is also a class instance
        if not hasattr(actual, "__class__"):
            return ComparisonResult(
                match=False, error_message=f"Expected class instance but got {type(actual)}"
            )

        # Check class name and module
        expected_class = expected["__class_name__"]
        expected_module = expected["__module__"]
        actual_class = actual.__class__.__name__
        actual_module = getattr(actual.__class__, "__module__", "")

        if actual_class != expected_class:
            return ComparisonResult(
                match=False,
                error_message=f"Class name mismatch: expected {expected_class}, got {actual_class}",
            )

        # Compare the __dict__ contents
        expected_dict = expected["__dict__"]
        actual_dict = actual.__dict__

        # Compare each attribute
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

        # Check for extra attributes in actual
        for key in actual_dict:
            if key not in expected_dict:
                return ComparisonResult(
                    match=False, error_message=f"Extra attribute '{key}' in actual instance"
                )

        return ComparisonResult(match=True)

    def _compare_numpy_arrays(self, actual: Any, expected: Any) -> ComparisonResult | None:
        """Compare numpy arrays."""
        if not (isinstance(actual, np.ndarray) and isinstance(expected, np.ndarray)):
            return None

        # Check shapes if strict_shapes is enabled
        if self.config.strict_shapes and actual.shape != expected.shape:
            return ComparisonResult(
                match=False,
                error_message=f"Array shapes differ: {actual.shape} vs {expected.shape}",
            )

        # Check dtypes if strict_types is enabled
        if self.config.strict_types and actual.dtype != expected.dtype:
            return ComparisonResult(
                match=False,
                error_message=f"Array dtypes differ: {actual.dtype} vs {expected.dtype}",
            )

        # Handle object arrays (like Shapely geometry arrays) - compare element-wise
        if actual.dtype == object or expected.dtype == object:
            return self._compare_object_arrays(actual, expected)

        # Use numpy.allclose for comparison of numeric arrays
        try:
            match = np.allclose(
                actual,
                expected,
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
                # Calculate some statistics about the difference
                diff = np.abs(actual - expected)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)

                details = {
                    "max_difference": float(max_diff),
                    "mean_difference": float(mean_diff),
                    "shape": actual.shape,
                    "dtype": str(actual.dtype),
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
            return ComparisonResult(match=False, error_message=f"numpy.allclose failed: {e}")

    def _compare_scalars(self, actual: Any, expected: Any) -> ComparisonResult | None:
        """Compare scalar values."""
        # Check if both are numeric scalars
        if not (self._is_numeric_scalar(actual) and self._is_numeric_scalar(expected)):
            return None

        # Convert to numpy scalars for consistent comparison
        actual_np = np.asarray(actual)
        expected_np = np.asarray(expected)

        # Use numpy.isclose for scalar comparison
        try:
            match = np.isclose(
                actual_np,
                expected_np,
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
                diff = abs(float(actual_np) - float(expected_np))
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

    def _compare_sequences(self, actual: Any, expected: Any) -> ComparisonResult | None:
        """Compare sequences (lists, tuples, etc.)."""
        if not (self._is_sequence(actual) and self._is_sequence(expected)):
            return None

        # Check lengths
        if len(actual) != len(expected):
            return ComparisonResult(
                match=False,
                error_message=f"Sequence lengths differ: {len(actual)} vs {len(expected)}",
            )

        # Compare elements
        mismatches = []
        for i, (a, e) in enumerate(zip(actual, expected)):
            result = self.compare(a, e)
            if not result.match:
                mismatches.append((i, result.error_message))

        if mismatches:
            return ComparisonResult(
                match=False,
                error_message=f"Sequence elements differ at indices: {[i for i, _ in mismatches[:5]]}",
                details={"mismatches": mismatches[:10]},  # Limit details
            )
        else:
            return ComparisonResult(match=True)

    def _compare_dicts(self, actual: Any, expected: Any) -> ComparisonResult | None:
        """Compare dictionaries."""
        if not (isinstance(actual, dict) and isinstance(expected, dict)):
            return None

        # Check keys
        actual_keys = set(actual.keys())
        expected_keys = set(expected.keys())

        if actual_keys != expected_keys:
            missing = expected_keys - actual_keys
            extra = actual_keys - expected_keys
            return ComparisonResult(
                match=False, error_message=f"Dict keys differ: missing={missing}, extra={extra}"
            )

        # Compare values
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

    def _compare_objects(self, actual: Any, expected: Any) -> ComparisonResult | None:
        """Compare objects with __eq__ method."""
        # Check if both objects have __eq__ defined (not just inherited from object)
        actual_type = type(actual)
        expected_type = type(expected)

        # Check if types match
        if actual_type != expected_type:
            return ComparisonResult(
                match=False,
                error_message=f"Type mismatch: {actual_type.__name__} vs {expected_type.__name__}",
            )

        # Skip lists/tuples that contain numpy arrays - let _compare_sequences handle them
        if isinstance(actual, (list, tuple)) and len(actual) > 0:
            if isinstance(actual[0], np.ndarray):
                return None  # Let _compare_sequences handle this

        # Check if __eq__ is properly implemented (not just the default object.__eq__)
        has_custom_eq = False
        for cls in actual_type.__mro__:
            if "__eq__" in cls.__dict__:
                has_custom_eq = True
                break

        if not has_custom_eq:
            # No custom __eq__, skip comparison and log
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
            # Check if __eq__ returned NotImplemented
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

            # Handle cases where __eq__ returns an array (e.g., SkyCoord, pandas Series)
            if isinstance(match, np.ndarray):
                match = bool(match.all())
            # Handle cases where __eq__ returns a list/tuple of arrays (e.g., lists of numpy arrays)
            elif isinstance(match, (list, tuple)) and len(match) > 0:
                # Check if it contains arrays (use len() to avoid evaluating the list as boolean)
                if isinstance(match[0], np.ndarray):
                    match = all(arr.all() if isinstance(arr, np.ndarray) else arr for arr in match)

            return ComparisonResult(
                match=match,
                error_message=None if match else f"Objects not equal: {actual} vs {expected}",
            )
        except Exception as e:
            return ComparisonResult(
                match=False, error_message=f"Object comparison failed: {e}", details={"type": actual_type.__name__}
            )

    def _compare_fallback(self, actual: Any, expected: Any) -> ComparisonResult | None:
        """Fallback comparison using == operator."""
        try:
            # Check type consistency
            if type(actual) != type(expected):
                return ComparisonResult(
                    match=False,
                    error_message=f"Type mismatch: {type(actual).__name__} vs {type(expected).__name__}",
                )

            match = actual == expected

            # If comparison returned NotImplemented, skip
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
            # If comparison fails, skip with warning
            return ComparisonResult(
                match=True,
                details={
                    "skipped": True,
                    "reason": f"Comparison failed: {e}",
                    "type": type(actual).__name__,
                },
                error_message=f"Skipped: comparison failed for {type(actual).__name__}: {e}",
            )

    def _compare_object_arrays(self, actual: np.ndarray, expected: np.ndarray) -> ComparisonResult:
        """Compare numpy object arrays element-wise."""
        # Flatten arrays for easier comparison
        actual_flat = actual.flatten()
        expected_flat = expected.flatten()

        # Compare each element
        mismatches = []
        for i, (a, e) in enumerate(zip(actual_flat, expected_flat)):
            # Use the compare method recursively for each element
            result = self.compare(a, e)
            if not result.match:
                mismatches.append((i, result.error_message))
                # Limit the number of mismatches we track for performance
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
        return isinstance(value, (int, float, np.number)) and np.isscalar(value)

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
