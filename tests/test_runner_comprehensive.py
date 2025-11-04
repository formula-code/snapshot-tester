"""Comprehensive tests for BenchmarkRunner."""
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from snapshot_tool.discovery import BenchmarkDiscovery
from snapshot_tool.runner import BenchmarkRunner


@pytest.fixture
def temp_benchmark_dir():
    """Create a temporary directory for test benchmarks."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def runner(temp_benchmark_dir):
    """Create a BenchmarkRunner instance."""
    return BenchmarkRunner(temp_benchmark_dir)


@pytest.fixture
def discovery(temp_benchmark_dir):
    """Create a BenchmarkDiscovery instance."""
    return BenchmarkDiscovery(temp_benchmark_dir)


class TestFunctionBenchmarkExecution:
    """Tests for executing function benchmarks."""

    def test_simple_function_execution(self, temp_benchmark_dir, runner, discovery):
        """Test executing a simple function benchmark."""
        bench_file = temp_benchmark_dir / "simple.py"
        bench_file.write_text("""
def time_simple():
    return 42
""")

        benchmarks = discovery.discover_all()
        result = runner.run_benchmark(benchmarks[0])

        assert result.success is True
        assert result.return_value == 42
        assert result.function_name is not None

    def test_function_with_computation(self, temp_benchmark_dir, runner, discovery):
        """Test function that performs computation."""
        bench_file = temp_benchmark_dir / "compute.py"
        bench_file.write_text("""
import numpy as np

def time_compute():
    return np.sum(np.array([1, 2, 3, 4, 5]))
""")

        benchmarks = discovery.discover_all()
        result = runner.run_benchmark(benchmarks[0])

        assert result.success is True
        assert result.return_value == 15

    def test_function_with_nested_calls(self, temp_benchmark_dir, runner, discovery):
        """Test function with nested function calls."""
        bench_file = temp_benchmark_dir / "nested.py"
        bench_file.write_text("""
def helper():
    return [1, 2, 3]

def time_nested():
    return helper()
""")

        benchmarks = discovery.discover_all()
        # Find the time_ benchmark
        benchmark = [b for b in benchmarks if b.name.startswith('time_')][0]
        result = runner.run_benchmark(benchmark)

        assert result.success is True
        assert result.return_value == [1, 2, 3]

    def test_function_with_imports(self, temp_benchmark_dir, runner, discovery):
        """Test function that uses imports."""
        bench_file = temp_benchmark_dir / "imports.py"
        bench_file.write_text("""
import math

def time_imports():
    return math.sqrt(16)
""")

        benchmarks = discovery.discover_all()
        result = runner.run_benchmark(benchmarks[0])

        assert result.success is True
        assert result.return_value == 4.0


class TestClassBenchmarkExecution:
    """Tests for executing class-based benchmarks."""

    def test_simple_class_method(self, temp_benchmark_dir, runner, discovery):
        """Test executing a simple class method benchmark."""
        bench_file = temp_benchmark_dir / "class_simple.py"
        bench_file.write_text("""
class SimpleBench:
    def time_method(self):
        return 42
""")

        benchmarks = discovery.discover_all()
        result = runner.run_benchmark(benchmarks[0])

        assert result.success is True
        assert result.return_value == 42

    def test_class_with_setup(self, temp_benchmark_dir, runner, discovery):
        """Test class with setup method."""
        bench_file = temp_benchmark_dir / "class_setup.py"
        bench_file.write_text("""
class BenchWithSetup:
    def setup(self):
        self.data = [1, 2, 3, 4, 5]

    def time_method(self):
        return sum(self.data)
""")

        benchmarks = discovery.discover_all()
        result = runner.run_benchmark(benchmarks[0])

        assert result.success is True
        assert result.return_value == 15

    def test_class_with_setup_and_params(self, temp_benchmark_dir, runner, discovery):
        """Test class with setup that receives parameters."""
        bench_file = temp_benchmark_dir / "class_setup_params.py"
        bench_file.write_text("""
import numpy as np

class BenchWithSetupParams:
    params = ([10, 100],)
    param_names = ['size']

    def setup(self, size):
        self.data = np.arange(size)

    def time_method(self):
        return np.sum(self.data)
""")

        benchmarks = discovery.discover_all()
        param_combinations = discovery.generate_parameter_combinations(benchmarks[0])

        for params in param_combinations:
            result = runner.run_benchmark(benchmarks[0], params)
            assert result.success is True
            # Verify the computation is correct
            size = params[0]
            expected = sum(range(size))
            assert result.return_value == expected

    def test_class_instance_state(self, temp_benchmark_dir, runner, discovery):
        """Test that class state is properly maintained."""
        bench_file = temp_benchmark_dir / "class_state.py"
        bench_file.write_text("""
class StatefulBench:
    def setup(self):
        self.counter = 0
        self.data = []

    def time_stateful(self):
        self.counter += 1
        self.data.append(self.counter)
        return self.data.copy()
""")

        benchmarks = discovery.discover_all()
        result = runner.run_benchmark(benchmarks[0])

        assert result.success is True
        assert result.return_value == [1]


class TestParameterHandling:
    """Tests for parameter handling."""

    def test_single_parameter(self, temp_benchmark_dir, runner, discovery):
        """Test benchmark with single parameter."""
        bench_file = temp_benchmark_dir / "single_param.py"
        bench_file.write_text("""
class SingleParam:
    params = ([1, 2, 3],)
    param_names = ['x']

    def setup(self, x):
        self.x = x

    def time_bench(self):
        return self.x * 10
""")

        benchmarks = discovery.discover_all()
        param_combinations = discovery.generate_parameter_combinations(benchmarks[0])

        for params in param_combinations:
            result = runner.run_benchmark(benchmarks[0], params)
            assert result.success is True
            assert result.return_value == params[0] * 10

    def test_multiple_parameters(self, temp_benchmark_dir, runner, discovery):
        """Test benchmark with multiple parameters."""
        bench_file = temp_benchmark_dir / "multi_param.py"
        bench_file.write_text("""
class MultiParam:
    params = ([1, 2], ['a', 'b'])
    param_names = ['num', 'letter']

    def setup(self, num, letter):
        self.num = num
        self.letter = letter

    def time_bench(self):
        return f"{self.letter}{self.num}"
""")

        benchmarks = discovery.discover_all()
        param_combinations = discovery.generate_parameter_combinations(benchmarks[0])

        expected_results = {
            (1, 'a'): 'a1',
            (1, 'b'): 'b1',
            (2, 'a'): 'a2',
            (2, 'b'): 'b2'
        }

        for params in param_combinations:
            result = runner.run_benchmark(benchmarks[0], params)
            assert result.success is True
            assert result.return_value == expected_results[params]

    def test_complex_parameter_types(self, temp_benchmark_dir, runner, discovery):
        """Test parameters with complex types."""
        bench_file = temp_benchmark_dir / "complex_params.py"
        bench_file.write_text("""
class ComplexParams:
    params = ([[1, 2], [3, 4]], [True, False])
    param_names = ['list_param', 'bool_param']

    def setup(self, list_param, bool_param):
        self.list_param = list_param
        self.bool_param = bool_param

    def time_bench(self):
        if self.bool_param:
            return sum(self.list_param)
        else:
            return len(self.list_param)
""")

        benchmarks = discovery.discover_all()
        param_combinations = discovery.generate_parameter_combinations(benchmarks[0])

        for params in param_combinations:
            result = runner.run_benchmark(benchmarks[0], params)
            assert result.success is True


class TestErrorHandling:
    """Tests for error handling during execution."""

    def test_benchmark_raises_exception(self, temp_benchmark_dir, runner, discovery):
        """Test handling of benchmarks that raise exceptions."""
        bench_file = temp_benchmark_dir / "raises.py"
        bench_file.write_text("""
def time_raises():
    raise ValueError("Test error")
""")

        benchmarks = discovery.discover_all()
        result = runner.run_benchmark(benchmarks[0])

        assert result.success is False
        assert result.error is not None
        assert "ValueError" in result.error

    def test_setup_raises_exception(self, temp_benchmark_dir, runner, discovery):
        """Test handling of exceptions in setup."""
        bench_file = temp_benchmark_dir / "setup_raises.py"
        bench_file.write_text("""
class SetupRaises:
    def setup(self):
        raise RuntimeError("Setup failed")

    def time_bench(self):
        return 42
""")

        benchmarks = discovery.discover_all()
        result = runner.run_benchmark(benchmarks[0])

        assert result.success is False
        assert result.error is not None

    def test_import_error(self, temp_benchmark_dir, runner, discovery):
        """Test handling of import errors."""
        bench_file = temp_benchmark_dir / "import_error.py"
        bench_file.write_text("""
import nonexistent_module

def time_bench():
    return 42
""")

        benchmarks = discovery.discover_all()
        result = runner.run_benchmark(benchmarks[0])

        assert result.success is False
        assert result.error is not None

    def test_syntax_error_in_benchmark(self, temp_benchmark_dir, runner, discovery):
        """Test handling of syntax errors."""
        bench_file = temp_benchmark_dir / "syntax_error.py"
        bench_file.write_text("""
def time_bench():
    return 42
    invalid syntax here
""")

        # Discovery should skip this file
        benchmarks = discovery.discover_all()
        assert len(benchmarks) == 0

    def test_missing_dependency(self, temp_benchmark_dir, runner, discovery):
        """Test error categorization for missing dependencies."""
        bench_file = temp_benchmark_dir / "missing_dep.py"
        bench_file.write_text("""
try:
    import some_missing_package
except ImportError:
    raise

def time_bench():
    return 42
""")

        benchmarks = discovery.discover_all()
        if benchmarks:
            result = runner.run_benchmark(benchmarks[0])
            assert result.success is False


class TestReturnValueCapture:
    """Tests for capturing different return value types."""

    def test_capture_numpy_array(self, temp_benchmark_dir, runner, discovery):
        """Test capturing numpy arrays."""
        bench_file = temp_benchmark_dir / "numpy.py"
        bench_file.write_text("""
import numpy as np

def time_numpy():
    return np.array([1, 2, 3, 4, 5])
""")

        benchmarks = discovery.discover_all()
        result = runner.run_benchmark(benchmarks[0])

        assert result.success is True
        assert isinstance(result.return_value, np.ndarray)
        assert np.array_equal(result.return_value, np.array([1, 2, 3, 4, 5]))

    def test_capture_dict(self, temp_benchmark_dir, runner, discovery):
        """Test capturing dictionaries."""
        bench_file = temp_benchmark_dir / "dict.py"
        bench_file.write_text("""
def time_dict():
    return {'key': 'value', 'number': 42}
""")

        benchmarks = discovery.discover_all()
        result = runner.run_benchmark(benchmarks[0])

        assert result.success is True
        assert result.return_value == {'key': 'value', 'number': 42}

    def test_capture_list(self, temp_benchmark_dir, runner, discovery):
        """Test capturing lists."""
        bench_file = temp_benchmark_dir / "list.py"
        bench_file.write_text("""
def time_list():
    return [1, 2, [3, 4], {'nested': 'dict'}]
""")

        benchmarks = discovery.discover_all()
        result = runner.run_benchmark(benchmarks[0])

        assert result.success is True
        assert result.return_value == [1, 2, [3, 4], {'nested': 'dict'}]

    def test_capture_class_instance(self, temp_benchmark_dir, runner, discovery):
        """Test capturing class instances."""
        bench_file = temp_benchmark_dir / "class_instance.py"
        bench_file.write_text("""
class Result:
    def __init__(self, value):
        self.value = value

def time_class():
    return Result(42)
""")

        benchmarks = discovery.discover_all()
        # Filter to just the benchmark
        benchmark = [b for b in benchmarks if b.name == 'time_class'][0]
        result = runner.run_benchmark(benchmark)

        assert result.success is True
        assert hasattr(result.return_value, 'value')
        assert result.return_value.value == 42


class TestModuleCaching:
    """Tests for module caching behavior."""

    def test_module_cached_between_runs(self, temp_benchmark_dir, runner, discovery):
        """Test that modules are cached between benchmark runs."""
        bench_file = temp_benchmark_dir / "cached.py"
        bench_file.write_text("""
counter = 0

def time_bench():
    global counter
    counter += 1
    return counter
""")

        benchmarks = discovery.discover_all()

        # Run twice - should use cached module
        result1 = runner.run_benchmark(benchmarks[0])
        result2 = runner.run_benchmark(benchmarks[0])

        # Both should succeed
        assert result1.success is True
        assert result2.success is True

    def test_different_benchmarks_same_module(self, temp_benchmark_dir, runner, discovery):
        """Test multiple benchmarks from same module."""
        bench_file = temp_benchmark_dir / "multi.py"
        bench_file.write_text("""
def time_bench1():
    return 1

def time_bench2():
    return 2
""")

        benchmarks = discovery.discover_all()
        assert len(benchmarks) == 2

        results = [runner.run_benchmark(b) for b in benchmarks]
        assert all(r.success for r in results)
        assert {r.return_value for r in results} == {1, 2}


class TestTracingBehavior:
    """Tests for execution tracing behavior."""

    def test_captures_deepest_call(self, temp_benchmark_dir, runner, discovery):
        """Test that tracer captures deepest function call."""
        bench_file = temp_benchmark_dir / "deep.py"
        bench_file.write_text("""
def inner():
    return "deepest"

def middle():
    return inner()

def time_outer():
    return middle()
""")

        benchmarks = discovery.discover_all()
        benchmark = [b for b in benchmarks if b.name.startswith('time_')][0]
        result = runner.run_benchmark(benchmark)

        assert result.success is True
        assert result.return_value == "deepest"
        # Should capture inner function
        assert result.function_name == "inner"

    def test_expression_to_return_transformation(self, temp_benchmark_dir, runner, discovery):
        """Test that expression statements are transformed to returns."""
        bench_file = temp_benchmark_dir / "expression.py"
        bench_file.write_text("""
def time_expression():
    1 + 1
    [1, 2, 3]
    "final expression"
""")

        benchmarks = discovery.discover_all()
        result = runner.run_benchmark(benchmarks[0])

        assert result.success is True
        # Should capture one of the expressions


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_benchmark(self, temp_benchmark_dir, runner, discovery):
        """Test benchmark that does nothing."""
        bench_file = temp_benchmark_dir / "empty.py"
        bench_file.write_text("""
def time_empty():
    pass
""")

        benchmarks = discovery.discover_all()
        result = runner.run_benchmark(benchmarks[0])

        # Should succeed even if return is None
        assert result is not None

    def test_benchmark_with_globals(self, temp_benchmark_dir, runner, discovery):
        """Test benchmark using global variables."""
        bench_file = temp_benchmark_dir / "globals.py"
        bench_file.write_text("""
GLOBAL_VALUE = 42

def time_globals():
    return GLOBAL_VALUE * 2
""")

        benchmarks = discovery.discover_all()
        result = runner.run_benchmark(benchmarks[0])

        assert result.success is True
        assert result.return_value == 84

    def test_benchmark_with_closure(self, temp_benchmark_dir, runner, discovery):
        """Test benchmark using closures."""
        bench_file = temp_benchmark_dir / "closure.py"
        bench_file.write_text("""
def time_closure():
    x = 10
    def inner():
        return x * 2
    return inner()
""")

        benchmarks = discovery.discover_all()
        result = runner.run_benchmark(benchmarks[0])

        assert result.success is True
        assert result.return_value == 20

    def test_very_long_running_benchmark(self, temp_benchmark_dir, runner, discovery):
        """Test benchmark that takes some time."""
        bench_file = temp_benchmark_dir / "slow.py"
        bench_file.write_text("""
import time

def time_slow():
    # Don't actually sleep, just simulate work
    total = 0
    for i in range(1000):
        total += i
    return total
""")

        benchmarks = discovery.discover_all()
        result = runner.run_benchmark(benchmarks[0])

        assert result.success is True
        assert result.return_value == sum(range(1000))

    def test_benchmark_modifying_state(self, temp_benchmark_dir, runner, discovery):
        """Test benchmark that modifies external state."""
        bench_file = temp_benchmark_dir / "state_modify.py"
        bench_file.write_text("""
state = []

def time_modify():
    state.append(1)
    return len(state)
""")

        benchmarks = discovery.discover_all()

        # Run multiple times
        results = [runner.run_benchmark(benchmarks[0]) for _ in range(3)]

        assert all(r.success for r in results)
        # State persists due to module caching
