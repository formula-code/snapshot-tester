"""Capture a benchmark's return value with sys.settrace.

The tracer keeps the shallowest meaningful return value from user code. The attribute is named
``deepest_call`` but holds the shallowest call.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class TraceResult:
    """Result of tracing execution."""

    return_value: Any
    function_name: str
    module_name: str
    depth: int
    success: bool
    error: Optional[Exception] = None


class ExecutionTracer:
    """Trace function calls and keep the shallowest meaningful return value."""

    def __init__(self, max_depth: int = 100):
        self.max_depth = max_depth
        self.deepest_call: Optional[TraceResult] = None
        self.current_depth = 0
        self.max_depth_reached = 0
        self.tracing = False
        self.current_frame: Optional[types.FrameType] = None

        # Modules never traced; also covers stdlib on Python < 3.10
        self.excluded_modules = {
            "builtins",
            "sys",
            "os",
            "pathlib",
            "typing",
            "dataclasses",
            "collections",
            "itertools",
            "functools",
            "operator",
            "re",
            "json",
            "pickle",
            "io",
            "abc",
            "copy",
            "copyreg",
            "enum",
            "importlib",
            "inspect",
            "ast",
            "codecs",
            "encodings",
            "weakref",
            "gc",
            "threading",
            "multiprocessing",
            "concurrent",
            "queue",
            "socket",
            "select",
            "selectors",
            "signal",
            "subprocess",
            "time",
            "datetime",
            "calendar",
            "contextlib",
            "traceback",
            "warnings",
            "logging",
            "argparse",
            "unittest",
            "pytest",
            "tokenize",
            "token",
            "keyword",
            "linecache",
            "sre",
            "_sre",
            "string",
            "textwrap",
            "pprint",
            "reprlib",
            "_pytest",
            "genericpath",
            "posixpath",
            "ntpath",
            "macpath",
            "stat",
            "shutil",
            "glob",
            "fnmatch",
            "tempfile",
            "atexit",
            "platform",
            "site",
        }

        self.excluded_methods = {
            "__init__",
            "__new__",
            "__del__",
            "__repr__",
            "__str__",
            "__eq__",
            "__ne__",
            "__lt__",
            "__le__",
            "__gt__",
            "__ge__",
            "__hash__",
            "__bool__",
            "__len__",
            "__iter__",
            "__next__",
            "__getitem__",
            "__setitem__",
            "__delitem__",
            "__contains__",
        }

    def trace_execution(self, func: Callable, *args, **kwargs) -> TraceResult:
        """
        Convenience method to trace a function's execution.

        Args:
            func: The function to trace
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            TraceResult containing the captured return value
        """
        self.start_tracing()
        try:
            func(*args, **kwargs)
        except Exception:
            # A value captured before the exception is still returned
            pass
        finally:
            result = self.stop_tracing()

        if result is None:
            return TraceResult(
                return_value=None,
                function_name=func.__name__,
                module_name=getattr(func, "__module__", "<unknown>"),
                depth=0,
                success=False,
                error=None,
            )

        return result

    def start_tracing(self) -> None:
        """Start tracing function calls."""
        self.tracing = True
        self.current_depth = 0
        self.max_depth_reached = 0
        self.deepest_call = None
        sys.settrace(self._trace_calls)

    def stop_tracing(self) -> Optional[TraceResult]:
        """Stop tracing and return the captured call, or None."""
        sys.settrace(None)
        self.tracing = False
        return self.deepest_call

    def _trace_calls(self, frame: types.FrameType, event: str, arg: Any) -> Optional[Callable]:
        """Trace function calls and returns."""
        if not self.tracing:
            return None

        if event == "call":
            return self._handle_call(frame)
        elif event == "return":
            return self._handle_return(frame, arg)
        elif event == "exception":
            return self._handle_exception(frame, arg)

        return self._trace_calls

    def _handle_call(self, frame: types.FrameType) -> Optional[Callable]:
        """Handle function call events."""
        if not self._should_trace_frame(frame):
            return None

        self.current_depth += 1
        self.max_depth_reached = max(self.max_depth_reached, self.current_depth)

        # Frames deeper than max_depth are not traced
        if self.current_depth > self.max_depth:
            return None

        self.current_frame = frame

        return self._trace_calls

    def _handle_return(self, frame: types.FrameType, arg: Any) -> Optional[Callable]:
        """Handle function return events."""
        if not self._should_trace_frame(frame):
            return None

        # Keep the shallowest meaningful value; at equal depth the later call wins
        if arg is not None and not self._is_meaningless_return(arg):
            if self.deepest_call is None or self.current_depth <= self.deepest_call.depth:
                self.deepest_call = TraceResult(
                    return_value=arg,
                    function_name=frame.f_code.co_name,
                    module_name=frame.f_globals.get("__name__", "<unknown>"),
                    depth=self.current_depth,
                    success=True,
                )

        self.current_depth -= 1
        return self._trace_calls

    def _handle_exception(self, frame: types.FrameType, arg: Any) -> Optional[Callable]:
        """Handle exception events."""
        if not self._should_trace_frame(frame):
            return None

        # Record the exception only if nothing was captured yet
        if self.deepest_call is None:
            self.deepest_call = TraceResult(
                return_value=None,
                function_name=frame.f_code.co_name,
                module_name=frame.f_globals.get("__name__", "<unknown>"),
                depth=self.current_depth,
                success=False,
                error=arg[1] if isinstance(arg, tuple) and len(arg) > 1 else arg,
            )

        self.current_depth -= 1
        return self._trace_calls

    def _should_trace_frame(self, frame: types.FrameType) -> bool:
        """Determine if we should trace this frame."""
        if not frame.f_globals or "__name__" not in frame.f_globals:
            return False

        module_name = frame.f_globals["__name__"]
        if module_name is None:
            return False

        function_name = frame.f_code.co_name

        if any(
            module_name == excluded or module_name.startswith(excluded + ".")
            for excluded in self.excluded_modules
        ):
            return False

        # Python 3.10+: skip a module if it or a parent package is in the stdlib
        if hasattr(sys, "stdlib_module_names"):
            module_parts = module_name.split(".")
            for i in range(len(module_parts)):
                partial_name = ".".join(module_parts[: i + 1])
                if partial_name in sys.stdlib_module_names:
                    return False

        # numpy internals
        numpy_internal_patterns = [
            "numpy._core",
            "numpy.core",
            "numpy.lib",
            "numpy.ma",
            "numpy.array_api",
            "numpy.f2py",
            "numpy.fft",
            "numpy.linalg",
            "numpy.random",
            "numpy.testing",
            "numpy._",
            "numpy.compat",
            "numpy.matrixlib",
        ]
        if any(module_name.startswith(pattern) for pattern in numpy_internal_patterns):
            return False

        # shapely internals; public shapely functions are still traced
        shapely_internal_patterns = [
            "shapely.lib",
            "shapely._",
            "shapely.geos",
            "shapely.geometry.base",
        ]
        if any(module_name.startswith(pattern) for pattern in shapely_internal_patterns):
            return False

        if function_name in self.excluded_methods:
            return False

        # Lambdas, comprehensions and other synthetic code names
        if function_name == "<lambda>" or function_name.startswith("<"):
            return False

        return True

    def _is_meaningless_return(self, value: Any) -> bool:
        """Check if a return value is meaningless for snapshot testing."""
        if value is None:
            return True

        try:
            if hasattr(value, "__len__") and len(value) == 0:
                return True
        except (TypeError, AttributeError):
            pass

        # Zero scalars (0, 0.0, "", False) are not meaningful
        if isinstance(value, (int, float, str, bool)) and not self._is_significant_value(value):
            return True

        if self._is_class_instance(value):
            return False

        return False

    def _is_significant_value(self, value: Any) -> bool:
        """Check if a simple value is significant enough to capture."""
        if isinstance(value, (int, float)):
            return value != 0
        elif isinstance(value, str):
            return len(value) > 0
        elif isinstance(value, bool):
            return True

        return True

    def _is_class_instance(self, value: Any) -> bool:
        """Check if a value is a class instance (not a built-in type)."""
        if hasattr(value, "__class__"):
            class_name = value.__class__.__name__
            module_name = getattr(value.__class__, "__module__", "") or ""

            if module_name in ("builtins", "types", "collections", "typing"):
                return False

            if class_name in ("int", "float", "str", "bool", "list", "dict", "tuple", "set"):
                return False

            if module_name and module_name.startswith("numpy"):
                return False

            return True

        return False

    def get_trace_stats(self) -> dict:
        """Get statistics about the tracing session."""
        return {
            "max_depth_reached": self.max_depth_reached,
            "deepest_call_depth": self.deepest_call.depth if self.deepest_call else 0,
            "deepest_function": self.deepest_call.function_name if self.deepest_call else None,
            "deepest_module": self.deepest_call.module_name if self.deepest_call else None,
            "captured_return": self.deepest_call is not None and self.deepest_call.success,
            "had_exception": self.deepest_call is not None and not self.deepest_call.success,
        }
