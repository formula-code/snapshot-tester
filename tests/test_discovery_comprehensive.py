"""Comprehensive tests for BenchmarkDiscovery."""
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

import shutil
import tempfile
from pathlib import Path

import pytest

from snapshot_tool.discovery import BenchmarkDiscovery


@pytest.fixture
def temp_benchmark_dir():
    """Create a temporary directory for test benchmarks."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def discovery(temp_benchmark_dir):
    """Create a BenchmarkDiscovery instance."""
    return BenchmarkDiscovery(temp_benchmark_dir)


class TestFunctionBenchmarkDiscovery:
    """Tests for discovering function-level benchmarks."""

    def test_simple_function_benchmark(self, temp_benchmark_dir, discovery):
        """Test discovery of simple function benchmarks."""
        benchmark_file = temp_benchmark_dir / "test_simple.py"
        benchmark_file.write_text("""
def time_simple():
    return 42

def time_another():
    return [1, 2, 3]

def not_a_benchmark():
    return "ignored"
""")

        benchmarks = discovery.discover_all()
        assert len(benchmarks) == 2
        assert all(b.benchmark_type == "function" for b in benchmarks)
        assert {b.name for b in benchmarks} == {"time_simple", "time_another"}

    def test_different_benchmark_prefixes(self, temp_benchmark_dir, discovery):
        """Test that all ASV prefixes are recognized."""
        benchmark_file = temp_benchmark_dir / "test_prefixes.py"
        benchmark_file.write_text("""
def time_timing():
    pass

def timeraw_raw_timing():
    pass

def mem_memory():
    pass

def peakmem_peak_memory():
    pass

def track_tracker():
    pass
""")

        benchmarks = discovery.discover_all()
        assert len(benchmarks) == 5
        prefixes = {b.name.split('_')[0] for b in benchmarks}
        assert prefixes == {"time", "timeraw", "mem", "peakmem", "track"}

    def test_function_with_docstring(self, temp_benchmark_dir, discovery):
        """Test that benchmarks with docstrings are discovered."""
        benchmark_file = temp_benchmark_dir / "test_docstring.py"
        benchmark_file.write_text('''
def time_with_docstring():
    """This is a benchmark with a docstring."""
    return 42
''')

        benchmarks = discovery.discover_all()
        assert len(benchmarks) == 1
        assert benchmarks[0].name == "time_with_docstring"


class TestClassBenchmarkDiscovery:
    """Tests for discovering class-based benchmarks."""

    def test_simple_class_benchmark(self, temp_benchmark_dir, discovery):
        """Test discovery of simple class benchmarks."""
        benchmark_file = temp_benchmark_dir / "test_class.py"
        benchmark_file.write_text("""
class SimpleBenchmark:
    def time_method(self):
        return 42

    def time_another_method(self):
        return [1, 2, 3]

    def not_a_benchmark(self):
        return "ignored"
""")

        benchmarks = discovery.discover_all()
        assert len(benchmarks) == 2
        assert all(b.benchmark_type == "method" for b in benchmarks)
        assert {b.name for b in benchmarks} == {"time_method", "time_another_method"}

    def test_class_with_setup(self, temp_benchmark_dir, discovery):
        """Test discovery of classes with setup methods."""
        benchmark_file = temp_benchmark_dir / "test_setup.py"
        benchmark_file.write_text("""
class BenchmarkWithSetup:
    def setup(self):
        self.data = [1, 2, 3]

    def time_benchmark(self):
        return sum(self.data)
""")

        benchmarks = discovery.discover_all()
        assert len(benchmarks) == 1
        assert benchmarks[0].has_setup is True
        assert benchmarks[0].name == "time_benchmark"

    def test_class_with_teardown(self, temp_benchmark_dir, discovery):
        """Test discovery of classes with teardown methods."""
        benchmark_file = temp_benchmark_dir / "test_teardown.py"
        benchmark_file.write_text("""
class BenchmarkWithTeardown:
    def teardown(self):
        pass

    def time_benchmark(self):
        return 42
""")

        benchmarks = discovery.discover_all()
        assert len(benchmarks) == 1


class TestParameterizedBenchmarks:
    """Tests for discovering parameterized benchmarks."""

    def test_simple_params(self, temp_benchmark_dir, discovery):
        """Test discovery of simple parameterized benchmarks."""
        benchmark_file = temp_benchmark_dir / "test_params.py"
        benchmark_file.write_text("""
class ParameterizedBenchmark:
    params = ([1, 2, 3], ['a', 'b'])
    param_names = ['number', 'letter']

    def setup(self, number, letter):
        self.number = number
        self.letter = letter

    def time_benchmark(self):
        return f"{self.letter}{self.number}"
""")

        benchmarks = discovery.discover_all()
        assert len(benchmarks) == 1
        benchmark = benchmarks[0]
        assert benchmark.params is not None
        assert benchmark.param_names == ['number', 'letter']

        # Test parameter combination generation
        combinations = discovery.generate_parameter_combinations(benchmark)
        assert len(combinations) == 6  # 3 * 2 = 6 combinations
        assert (1, 'a') in combinations
        assert (3, 'b') in combinations

    def test_params_without_names(self, temp_benchmark_dir, discovery):
        """Test parameters without explicit names."""
        benchmark_file = temp_benchmark_dir / "test_unnamed_params.py"
        benchmark_file.write_text("""
class UnnamedParams:
    params = ([1, 2, 3],)

    def time_benchmark(self):
        return 42
""")

        benchmarks = discovery.discover_all()
        assert len(benchmarks) == 1
        benchmark = benchmarks[0]
        assert benchmark.params is not None
        assert benchmark.param_names is None

        combinations = discovery.generate_parameter_combinations(benchmark)
        assert len(combinations) == 3

    def test_runtime_eval_detection(self, temp_benchmark_dir, discovery):
        """Test detection of parameters that need runtime evaluation."""
        benchmark_file = temp_benchmark_dir / "test_runtime.py"
        benchmark_file.write_text("""
import numpy as np

class RuntimeEvalBenchmark:
    params = [list(range(10)), np.arange(5)]

    def time_benchmark(self):
        return 42
""")

        benchmarks = discovery.discover_all()
        assert len(benchmarks) == 1
        # Should detect that params need runtime evaluation
        assert benchmarks[0].needs_runtime_eval is True

    def test_nested_params(self, temp_benchmark_dir, discovery):
        """Test complex nested parameter structures."""
        benchmark_file = temp_benchmark_dir / "test_nested.py"
        benchmark_file.write_text("""
class NestedParams:
    params = ([[1, 2], [3, 4]], ['x', 'y', 'z'])
    param_names = ['nested_list', 'letter']

    def time_benchmark(self):
        return 42
""")

        benchmarks = discovery.discover_all()
        assert len(benchmarks) == 1
        benchmark = benchmarks[0]

        combinations = discovery.generate_parameter_combinations(benchmark)
        assert len(combinations) == 6  # 2 * 3 = 6
        assert ([1, 2], 'x') in combinations


class TestModuleStructure:
    """Tests for module-level discovery."""

    def test_multiple_files(self, temp_benchmark_dir, discovery):
        """Test discovery across multiple files."""
        (temp_benchmark_dir / "bench1.py").write_text("""
def time_bench1():
    return 1
""")
        (temp_benchmark_dir / "bench2.py").write_text("""
def time_bench2():
    return 2
""")

        benchmarks = discovery.discover_all()
        assert len(benchmarks) == 2
        modules = {b.module_path for b in benchmarks}
        assert modules == {"bench1", "bench2"}

    def test_nested_directories(self, temp_benchmark_dir, discovery):
        """Test discovery in nested directories."""
        subdir = temp_benchmark_dir / "subdir"
        subdir.mkdir()
        (subdir / "nested_bench.py").write_text("""
def time_nested():
    return 42
""")

        benchmarks = discovery.discover_all()
        assert len(benchmarks) == 1
        assert "subdir" in benchmarks[0].module_path

    def test_ignore_non_benchmark_files(self, temp_benchmark_dir, discovery):
        """Test that non-benchmark files are ignored."""
        (temp_benchmark_dir / "config.py").write_text("""
# Configuration file
CONFIG = {'key': 'value'}
""")
        (temp_benchmark_dir / "utils.py").write_text("""
def helper_function():
    return "helper"
""")

        benchmarks = discovery.discover_all()
        assert len(benchmarks) == 0

    def test_ignore_init_files(self, temp_benchmark_dir, discovery):
        """Test that __init__.py files don't cause issues."""
        (temp_benchmark_dir / "__init__.py").write_text("")
        (temp_benchmark_dir / "bench.py").write_text("""
def time_benchmark():
    return 42
""")

        benchmarks = discovery.discover_all()
        assert len(benchmarks) == 1


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_directory(self, temp_benchmark_dir, discovery):
        """Test discovery in empty directory."""
        benchmarks = discovery.discover_all()
        assert len(benchmarks) == 0

    def test_syntax_error_in_file(self, temp_benchmark_dir, discovery):
        """Test handling of files with syntax errors."""
        (temp_benchmark_dir / "bad_syntax.py").write_text("""
def time_benchmark()  # Missing colon
    return 42
""")

        # Should not crash, should skip the file
        benchmarks = discovery.discover_all()
        assert len(benchmarks) == 0

    def test_class_and_function_in_same_file(self, temp_benchmark_dir, discovery):
        """Test file with both class and function benchmarks."""
        (temp_benchmark_dir / "mixed.py").write_text("""
def time_function():
    return 1

class BenchmarkClass:
    def time_method(self):
        return 2
""")

        benchmarks = discovery.discover_all()
        assert len(benchmarks) == 2
        types = {b.benchmark_type for b in benchmarks}
        assert types == {"function", "method"}

    def test_multiple_classes_in_file(self, temp_benchmark_dir, discovery):
        """Test file with multiple benchmark classes."""
        (temp_benchmark_dir / "multi_class.py").write_text("""
class FirstBenchmark:
    def time_first(self):
        return 1

class SecondBenchmark:
    def time_second(self):
        return 2
""")

        benchmarks = discovery.discover_all()
        assert len(benchmarks) == 2
        classes = {b.class_name for b in benchmarks}
        assert classes == {"FirstBenchmark", "SecondBenchmark"}

    def test_empty_class(self, temp_benchmark_dir, discovery):
        """Test that classes without benchmarks are ignored."""
        (temp_benchmark_dir / "empty_class.py").write_text("""
class EmptyClass:
    pass

class OnlySetup:
    def setup(self):
        pass
""")

        benchmarks = discovery.discover_all()
        assert len(benchmarks) == 0

    def test_benchmark_with_decorators(self, temp_benchmark_dir, discovery):
        """Test that decorated benchmarks are discovered."""
        (temp_benchmark_dir / "decorated.py").write_text("""
def my_decorator(func):
    return func

@my_decorator
def time_decorated():
    return 42
""")

        benchmarks = discovery.discover_all()
        assert len(benchmarks) == 1

    def test_params_as_single_list(self, temp_benchmark_dir, discovery):
        """Test params defined as a single list (not tuple of lists)."""
        (temp_benchmark_dir / "single_param.py").write_text("""
class SingleParam:
    params = [1, 2, 3]

    def time_benchmark(self):
        return 42
""")

        benchmarks = discovery.discover_all()
        assert len(benchmarks) == 1

        combinations = discovery.generate_parameter_combinations(benchmarks[0])
        # Should handle as single parameter dimension
        assert len(combinations) == 3


class TestBenchmarkInfoAttributes:
    """Test BenchmarkInfo dataclass attributes."""

    def test_function_benchmark_attributes(self, temp_benchmark_dir, discovery):
        """Test that function benchmarks have correct attributes."""
        (temp_benchmark_dir / "func.py").write_text("""
def time_test():
    return 42
""")

        benchmarks = discovery.discover_all()
        benchmark = benchmarks[0]

        assert benchmark.name == "time_test"
        assert benchmark.benchmark_type == "function"
        assert benchmark.class_name is None
        assert benchmark.has_setup is False
        assert benchmark.params is None
        assert benchmark.param_names is None
        assert benchmark.module_path == "func"

    def test_class_benchmark_attributes(self, temp_benchmark_dir, discovery):
        """Test that class benchmarks have correct attributes."""
        (temp_benchmark_dir / "cls.py").write_text("""
class MyBenchmark:
    params = ([1, 2],)
    param_names = ['x']

    def setup(self, x):
        self.x = x

    def time_test(self):
        return self.x
""")

        benchmarks = discovery.discover_all()
        benchmark = benchmarks[0]

        assert benchmark.name == "time_test"
        assert benchmark.benchmark_type == "method"
        assert benchmark.class_name == "MyBenchmark"
        assert benchmark.has_setup is True
        assert benchmark.params is not None
        assert benchmark.param_names == ['x']
