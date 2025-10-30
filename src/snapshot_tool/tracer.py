"""
Execution tracer for capturing function return values.

This module implements a tracing mechanism using sys.settrace to capture
the deepest function call's return value during benchmark execution.
"""

import sys
import types
from typing import Any, Optional, Set, Callable
from dataclasses import dataclass


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
    """Traces function execution to capture the deepest return value."""

    def __init__(self, max_depth: int = 100):
        self.max_depth = max_depth
        self.deepest_call: Optional[TraceResult] = None
        self.current_depth = 0
        self.max_depth_reached = 0
        self.tracing = False
        self.current_frame: Optional[types.FrameType] = None

        # Functions to exclude from tracing
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
        }

        # Method names to exclude
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

    def start_tracing(self) -> None:
        """Start tracing function calls."""
        self.tracing = True
        self.current_depth = 0
        self.max_depth_reached = 0
        self.deepest_call = None
        sys.settrace(self._trace_calls)

    def stop_tracing(self) -> Optional[TraceResult]:
        """Stop tracing and return the deepest captured call."""
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

        # Limit depth to prevent infinite recursion
        if self.current_depth > self.max_depth:
            return None

        # Store the frame for potential result capture
        self.current_frame = frame

        # No more debug print here
        return self._trace_calls

    def _handle_return(self, frame: types.FrameType, arg: Any) -> Optional[Callable]:
        """Handle function return events."""
        if not self._should_trace_frame(frame):
            return None

        # Only capture if this is deeper than our current deepest call
        if self.deepest_call is None or self.current_depth > self.deepest_call.depth:
            # Skip None returns and some common non-meaningful returns
            if arg is not None and not self._is_meaningless_return(arg):
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

        # Record exception as deepest call if it's the deepest we've seen
        if self.deepest_call is None or self.current_depth > self.deepest_call.depth:
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
        # Skip frames without proper module info
        if not frame.f_globals or "__name__" not in frame.f_globals:
            return False

        module_name = frame.f_globals["__name__"]
        function_name = frame.f_code.co_name

        # Skip excluded modules
        if any(module_name.startswith(excluded) for excluded in self.excluded_modules):
            return False

        # Skip excluded methods
        if function_name in self.excluded_methods:
            return False

        # Skip lambda functions and anonymous functions
        if function_name == "<lambda>" or function_name.startswith("<"):
            return False

        return True

    def _is_meaningless_return(self, value: Any) -> bool:
        """Check if a return value is meaningless for snapshot testing."""
        # Skip common meaningless returns
        if value is None:
            return True

        # Skip empty containers (but be careful with types)
        try:
            if hasattr(value, "__len__") and len(value) == 0:
                return True
        except (TypeError, AttributeError):
            # Some objects have __len__ but can't be called with len()
            pass

        # Skip simple types that are likely not the main computation result
        if isinstance(value, (int, float, str, bool)) and not self._is_significant_value(value):
            return True

        # Always capture class instances - they might be important return values
        if self._is_class_instance(value):
            return False

        return False

    def _is_significant_value(self, value: Any) -> bool:
        """Check if a simple value is significant enough to capture."""
        # Capture non-zero numbers, non-empty strings, etc.
        if isinstance(value, (int, float)):
            return value != 0
        elif isinstance(value, str):
            return len(value) > 0
        elif isinstance(value, bool):
            return True  # Always capture booleans

        return True

    def _is_class_instance(self, value: Any) -> bool:
        """Check if a value is a class instance (not a built-in type)."""
        # Check if it's an instance of a user-defined class
        if hasattr(value, "__class__"):
            class_name = value.__class__.__name__
            module_name = getattr(value.__class__, "__module__", "")

            # Skip built-in types and common library types
            if module_name in ("builtins", "types", "collections", "typing"):
                return False

            # Skip if it's a basic type
            if class_name in ("int", "float", "str", "bool", "list", "dict", "tuple", "set"):
                return False

            # Skip numpy arrays and other common scientific computing types
            if module_name.startswith("numpy"):
                return False

            # It's likely a user-defined class instance
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
