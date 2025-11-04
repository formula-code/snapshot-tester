"""Comprehensive tests for ExecutionTracer."""
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

import numpy as np

from snapshot_tool.tracer import ExecutionTracer


class TestBasicTracing:
    """Tests for basic tracing functionality."""

    def test_simple_function_trace(self):
        """Test tracing a simple function."""
        def simple_func():
            return 42

        tracer = ExecutionTracer()
        result = tracer.trace_execution(simple_func)

        assert result.success is True
        assert result.return_value == 42
        assert result.function_name == "simple_func"
        assert result.depth >= 0

    def test_nested_function_trace(self):
        """Test that tracer captures deepest function call."""
        def inner():
            return "inner_result"

        def outer():
            return inner()

        tracer = ExecutionTracer()
        result = tracer.trace_execution(outer)

        assert result.success is True
        # Should capture the inner function's return value
        assert result.return_value == "inner_result"
        assert result.function_name == "inner"
        assert result.depth > 0

    def test_deeply_nested_calls(self):
        """Test tracing with multiple levels of nesting."""
        def level3():
            return [1, 2, 3]

        def level2():
            return level3()

        def level1():
            return level2()

        tracer = ExecutionTracer()
        result = tracer.trace_execution(level1)

        assert result.success is True
        assert result.return_value == [1, 2, 3]
        assert result.function_name == "level3"

    def test_function_with_arguments(self):
        """Test tracing function with arguments."""
        def add(a, b):
            return a + b

        tracer = ExecutionTracer()
        result = tracer.trace_execution(lambda: add(5, 3))

        assert result.success is True
        assert result.return_value == 8

    def test_function_with_kwargs(self):
        """Test tracing function with keyword arguments."""
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        tracer = ExecutionTracer()
        result = tracer.trace_execution(lambda: greet("World", greeting="Hi"))

        assert result.success is True
        assert result.return_value == "Hi, World!"


class TestReturnValueTypes:
    """Tests for various return value types."""

    def test_none_return(self):
        """Test function returning None."""
        def returns_none():
            pass

        tracer = ExecutionTracer()
        result = tracer.trace_execution(returns_none)

        # Tracer should skip None returns and look deeper
        assert result.success is True

    def test_numeric_return(self):
        """Test function returning numbers."""
        def returns_int():
            return 42

        def returns_float():
            return 3.14

        tracer = ExecutionTracer()

        result = tracer.trace_execution(returns_int)
        assert result.return_value == 42
        assert isinstance(result.return_value, int)

        result = tracer.trace_execution(returns_float)
        assert result.return_value == 3.14
        assert isinstance(result.return_value, float)

    def test_string_return(self):
        """Test function returning strings."""
        def returns_string():
            return "test string"

        tracer = ExecutionTracer()
        result = tracer.trace_execution(returns_string)

        assert result.return_value == "test string"
        assert isinstance(result.return_value, str)

    def test_list_return(self):
        """Test function returning lists."""
        def returns_list():
            return [1, 2, 3, 4, 5]

        tracer = ExecutionTracer()
        result = tracer.trace_execution(returns_list)

        assert result.return_value == [1, 2, 3, 4, 5]
        assert isinstance(result.return_value, list)

    def test_dict_return(self):
        """Test function returning dictionaries."""
        def returns_dict():
            return {"key": "value", "number": 42}

        tracer = ExecutionTracer()
        result = tracer.trace_execution(returns_dict)

        assert result.return_value == {"key": "value", "number": 42}
        assert isinstance(result.return_value, dict)

    def test_numpy_array_return(self):
        """Test function returning numpy arrays."""
        def returns_array():
            return np.array([1, 2, 3, 4, 5])

        tracer = ExecutionTracer()
        result = tracer.trace_execution(returns_array)

        assert isinstance(result.return_value, np.ndarray)
        assert np.array_equal(result.return_value, np.array([1, 2, 3, 4, 5]))

    def test_tuple_return(self):
        """Test function returning tuples."""
        def returns_tuple():
            return (1, 2, 3)

        tracer = ExecutionTracer()
        result = tracer.trace_execution(returns_tuple)

        assert result.return_value == (1, 2, 3)
        assert isinstance(result.return_value, tuple)

    def test_set_return(self):
        """Test function returning sets."""
        def returns_set():
            return {1, 2, 3}

        tracer = ExecutionTracer()
        result = tracer.trace_execution(returns_set)

        assert result.return_value == {1, 2, 3}
        assert isinstance(result.return_value, set)

    def test_class_instance_return(self):
        """Test function returning class instances."""
        class TestClass:
            def __init__(self, value):
                self.value = value

        def returns_instance():
            return TestClass(42)

        tracer = ExecutionTracer()
        result = tracer.trace_execution(returns_instance)

        assert isinstance(result.return_value, TestClass)
        assert result.return_value.value == 42


class TestBuiltinFiltering:
    """Tests for filtering built-in functions."""

    def test_builtin_functions_skipped(self):
        """Test that built-in functions are skipped."""
        def uses_builtins():
            # These should be skipped by the tracer
            length = len([1, 2, 3])
            max_val = max([1, 2, 3])
            return max_val + length

        tracer = ExecutionTracer()
        result = tracer.trace_execution(uses_builtins)

        assert result.success is True
        assert result.return_value == 6
        # Should not trace into built-ins like len() or max()
        assert result.function_name == "uses_builtins"

    def test_stdlib_functions_skipped(self):
        """Test that standard library functions are skipped."""
        import math

        def uses_stdlib():
            return math.sqrt(16)

        tracer = ExecutionTracer()
        result = tracer.trace_execution(uses_stdlib)

        assert result.success is True
        assert result.return_value == 4.0
        # Should not trace into math.sqrt
        assert result.function_name == "uses_stdlib"

    def test_dunder_methods_skipped(self):
        """Test that dunder methods are filtered."""
        class MyClass:
            def __init__(self):
                self.value = 42

            def get_value(self):
                return self.value

        def creates_instance():
            obj = MyClass()
            return obj.get_value()

        tracer = ExecutionTracer()
        result = tracer.trace_execution(creates_instance)

        assert result.success is True
        assert result.return_value == 42
        # Should skip __init__ and capture get_value
        assert result.function_name == "get_value"


class TestDepthTracking:
    """Tests for call depth tracking."""

    def test_depth_increases_with_nesting(self):
        """Test that depth increases with nested calls."""
        depths = []

        def level1():
            return level2()

        def level2():
            return level3()

        def level3():
            return "result"

        tracer = ExecutionTracer()
        result = tracer.trace_execution(level1)

        # Should have tracked multiple depths
        assert result.depth > 0

    def test_max_depth_not_exceeded(self):
        """Test that tracer respects max depth to avoid infinite recursion."""
        def recursive(n):
            if n <= 0:
                return 0
            return recursive(n - 1) + 1

        tracer = ExecutionTracer()
        result = tracer.trace_execution(lambda: recursive(100))

        # Should complete without stack overflow
        assert result.success is True


class TestErrorHandling:
    """Tests for error handling in tracer."""

    def test_exception_in_traced_function(self):
        """Test handling of exceptions during tracing."""
        def raises_error():
            raise ValueError("Test error")

        tracer = ExecutionTracer()
        result = tracer.trace_execution(raises_error)

        assert result.success is False
        assert "ValueError" in result.error

    def test_exception_in_nested_function(self):
        """Test handling of exceptions in nested calls."""
        def inner():
            raise RuntimeError("Inner error")

        def outer():
            return inner()

        tracer = ExecutionTracer()
        result = tracer.trace_execution(outer)

        assert result.success is False
        assert "RuntimeError" in result.error

    def test_type_error_handling(self):
        """Test handling of type errors."""
        def bad_types():
            return "string" + 42  # Will raise TypeError

        tracer = ExecutionTracer()
        result = tracer.trace_execution(bad_types)

        assert result.success is False
        assert "TypeError" in result.error

    def test_attribute_error_handling(self):
        """Test handling of attribute errors."""
        def missing_attr():
            obj = object()
            return obj.nonexistent_attribute

        tracer = ExecutionTracer()
        result = tracer.trace_execution(missing_attr)

        assert result.success is False
        assert "AttributeError" in result.error


class TestCallableTypes:
    """Tests for different callable types."""

    def test_lambda_function(self):
        """Test tracing lambda functions."""
        tracer = ExecutionTracer()
        result = tracer.trace_execution(lambda: 42)

        assert result.success is True
        assert result.return_value == 42

    def test_class_method(self):
        """Test tracing class methods."""
        class MyClass:
            def method(self):
                return "method_result"

        obj = MyClass()
        tracer = ExecutionTracer()
        result = tracer.trace_execution(obj.method)

        assert result.success is True
        assert result.return_value == "method_result"

    def test_static_method(self):
        """Test tracing static methods."""
        class MyClass:
            @staticmethod
            def static():
                return "static_result"

        tracer = ExecutionTracer()
        result = tracer.trace_execution(MyClass.static)

        assert result.success is True
        assert result.return_value == "static_result"

    def test_class_method_decorator(self):
        """Test tracing class methods with @classmethod."""
        class MyClass:
            @classmethod
            def class_method(cls):
                return "classmethod_result"

        tracer = ExecutionTracer()
        result = tracer.trace_execution(MyClass.class_method)

        assert result.success is True
        assert result.return_value == "classmethod_result"


class TestComplexScenarios:
    """Tests for complex tracing scenarios."""

    def test_recursive_function(self):
        """Test tracing recursive functions."""
        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n - 1) + fibonacci(n - 2)

        tracer = ExecutionTracer()
        result = tracer.trace_execution(lambda: fibonacci(5))

        assert result.success is True
        assert result.return_value == 5

    def test_generator_function(self):
        """Test tracing generators."""
        def gen():
            yield 1
            yield 2
            yield 3

        tracer = ExecutionTracer()
        result = tracer.trace_execution(gen)

        assert result.success is True
        # Should capture the generator object
        assert hasattr(result.return_value, '__next__')

    def test_list_comprehension(self):
        """Test tracing with list comprehensions."""
        def uses_comprehension():
            return [x * 2 for x in range(5)]

        tracer = ExecutionTracer()
        result = tracer.trace_execution(uses_comprehension)

        assert result.success is True
        assert result.return_value == [0, 2, 4, 6, 8]

    def test_multiple_return_paths(self):
        """Test function with multiple return paths."""
        def multi_return(x):
            if x > 0:
                return "positive"
            elif x < 0:
                return "negative"
            else:
                return "zero"

        tracer = ExecutionTracer()

        result = tracer.trace_execution(lambda: multi_return(5))
        assert result.return_value == "positive"

        result = tracer.trace_execution(lambda: multi_return(-5))
        assert result.return_value == "negative"

        result = tracer.trace_execution(lambda: multi_return(0))
        assert result.return_value == "zero"

    def test_context_manager(self):
        """Test tracing with context managers."""
        def uses_context_manager():
            result = []
            with open(__file__) as f:
                # Just check we can read
                result.append(f.readline())
            return len(result)

        tracer = ExecutionTracer()
        result = tracer.trace_execution(uses_context_manager)

        assert result.success is True
        assert result.return_value == 1

    def test_numpy_operations(self):
        """Test tracing numpy operations."""
        def numpy_calc():
            arr = np.array([1, 2, 3, 4, 5])
            return np.sum(arr) * 2

        tracer = ExecutionTracer()
        result = tracer.trace_execution(numpy_calc)

        assert result.success is True
        assert result.return_value == 30

    def test_nested_data_structures(self):
        """Test with nested data structures."""
        def nested_structures():
            return {
                'list': [1, 2, [3, 4]],
                'dict': {'a': {'b': 'c'}},
                'tuple': (1, (2, 3)),
                'mixed': [{'key': [1, 2, 3]}]
            }

        tracer = ExecutionTracer()
        result = tracer.trace_execution(nested_structures)

        assert result.success is True
        assert isinstance(result.return_value, dict)
        assert result.return_value['list'] == [1, 2, [3, 4]]


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_function(self):
        """Test function that does nothing."""
        def empty():
            pass

        tracer = ExecutionTracer()
        result = tracer.trace_execution(empty)

        assert result.success is True
        # Empty function returns None, which might be skipped

    def test_function_returning_function(self):
        """Test function that returns another function."""
        def outer():
            def inner():
                return 42
            return inner

        tracer = ExecutionTracer()
        result = tracer.trace_execution(outer)

        assert result.success is True
        assert callable(result.return_value)

    def test_circular_reference(self):
        """Test handling of circular references."""
        def circular():
            a = []
            b = [a]
            a.append(b)
            return a

        tracer = ExecutionTracer()
        result = tracer.trace_execution(circular)

        assert result.success is True
        # Should handle circular structure without infinite loop

    def test_very_large_return_value(self):
        """Test with very large return values."""
        def large_data():
            return list(range(10000))

        tracer = ExecutionTracer()
        result = tracer.trace_execution(large_data)

        assert result.success is True
        assert len(result.return_value) == 10000

    def test_unicode_strings(self):
        """Test with unicode strings."""
        def unicode_func():
            return "Hello 世界 🌍"

        tracer = ExecutionTracer()
        result = tracer.trace_execution(unicode_func)

        assert result.success is True
        assert result.return_value == "Hello 世界 🌍"
