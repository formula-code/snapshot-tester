"""
Comparison engine for snapshot testing.

This module compares captured outputs with stored snapshots using
numpy.allclose for numerical data and other strategies for different types.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings


@dataclass
class ComparisonResult:
    """Result of comparing two values."""

    match: bool
    tolerance_used: Optional[Dict[str, float]] = None
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


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

    def __init__(self, config: Optional[ComparisonConfig] = None):
        self.config = config or ComparisonConfig()

    def compare(self, actual: Any, expected: Any) -> ComparisonResult:
        """Compare actual value with expected value."""
        try:
            # Handle serialized generators - skip comparison
            if isinstance(expected, dict) and expected.get("__generator__"):
                return ComparisonResult(
                    match=True, details="Skipped comparison for generator (cannot be pickled)"
                )

            # Handle serialized callables (functions/closures) - skip comparison
            if isinstance(expected, dict) and expected.get("__callable__"):
                return ComparisonResult(
                    match=True,
                    details="Skipped comparison for callable (cannot be pickled reliably)",
                )

            # Handle unpicklable placeholder
            if isinstance(expected, dict) and expected.get("__unpicklable__"):
                return ComparisonResult(
                    match=True,
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
            strategies = [
                self._compare_numpy_arrays,
                self._compare_scalars,
                self._compare_sequences,
                self._compare_dicts,
                self._compare_objects,
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

    def _compare_class_instance(self, actual: Any, expected: Dict[str, Any]) -> ComparisonResult:
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

    def _compare_numpy_arrays(self, actual: Any, expected: Any) -> Optional[ComparisonResult]:
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

        # Use numpy.allclose for comparison
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

    def _compare_scalars(self, actual: Any, expected: Any) -> Optional[ComparisonResult]:
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

    def _compare_sequences(self, actual: Any, expected: Any) -> Optional[ComparisonResult]:
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

    def _compare_dicts(self, actual: Any, expected: Any) -> Optional[ComparisonResult]:
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

    def _compare_objects(self, actual: Any, expected: Any) -> Optional[ComparisonResult]:
        """Compare objects with __eq__ method."""
        if not (hasattr(actual, "__eq__") and hasattr(expected, "__eq__")):
            return None

        try:
            match = actual == expected
            return ComparisonResult(
                match=match,
                error_message=None if match else f"Objects not equal: {actual} vs {expected}",
            )
        except Exception as e:
            return ComparisonResult(match=False, error_message=f"Object comparison failed: {e}")

    def _compare_fallback(self, actual: Any, expected: Any) -> Optional[ComparisonResult]:
        """Fallback comparison using == operator."""
        try:
            match = actual == expected
            return ComparisonResult(
                match=match,
                error_message=None if match else f"Values not equal: {actual} vs {expected}",
            )
        except Exception as e:
            return ComparisonResult(match=False, error_message=f"Fallback comparison failed: {e}")

    def _is_numeric_scalar(self, value: Any) -> bool:
        """Check if value is a numeric scalar."""
        return isinstance(value, (int, float, np.number)) and np.isscalar(value)

    def _is_sequence(self, value: Any) -> bool:
        """Check if value is a sequence (but not a string)."""
        return (
            hasattr(value, "__len__")
            and hasattr(value, "__getitem__")
            and not isinstance(value, (str, bytes))
        )

    def compare_multiple(
        self, actual_values: List[Any], expected_values: List[Any]
    ) -> List[ComparisonResult]:
        """Compare multiple pairs of values."""
        if len(actual_values) != len(expected_values):
            raise ValueError("Lists must have the same length")

        return [
            self.compare(actual, expected)
            for actual, expected in zip(actual_values, expected_values)
        ]

    def get_summary_stats(self, results: List[ComparisonResult]) -> Dict[str, Any]:
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
