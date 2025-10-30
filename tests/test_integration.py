"""Integration tests for end-to-end workflows."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
from snapshot_tool import (
    BenchmarkDiscovery,
    BenchmarkRunner,
    SnapshotManager,
    Comparator,
)


@pytest.fixture
def workspace():
    """Create a temporary workspace with benchmark and snapshot dirs."""
    temp_dir = tempfile.mkdtemp()
    workspace_path = Path(temp_dir)

    benchmark_dir = workspace_path / "benchmarks"
    benchmark_dir.mkdir()

    snapshot_dir = workspace_path / ".snapshots"
    snapshot_dir.mkdir()

    yield {
        'root': workspace_path,
        'benchmarks': benchmark_dir,
        'snapshots': snapshot_dir
    }

    shutil.rmtree(temp_dir, ignore_errors=True)


class TestCaptureWorkflow:
    """Tests for the complete capture workflow."""

    def test_capture_simple_function_benchmark(self, workspace):
        """Test capturing a simple function benchmark."""
        # Create benchmark file
        bench_file = workspace['benchmarks'] / "simple.py"
        bench_file.write_text("""
import numpy as np

def time_simple():
    return np.array([1, 2, 3, 4, 5])
""")

        # Discover benchmarks
        discovery = BenchmarkDiscovery(workspace['benchmarks'])
        benchmarks = discovery.discover_all()
        assert len(benchmarks) == 1

        # Run benchmark
        runner = BenchmarkRunner(workspace['benchmarks'])
        result = runner.run_benchmark(benchmarks[0])
        assert result.success is True
        assert isinstance(result.return_value, np.ndarray)

        # Store snapshot
        storage = SnapshotManager(workspace['snapshots'])
        snapshot_path = storage.store_snapshot(
            benchmark_name=benchmarks[0].name,
            module_path=benchmarks[0].module_path,
            parameters=(),
            param_names=None,
            return_value=result.return_value
        )
        assert snapshot_path.exists()

    def test_capture_parameterized_benchmark(self, workspace):
        """Test capturing parameterized benchmarks."""
        bench_file = workspace['benchmarks'] / "parameterized.py"
        bench_file.write_text("""
import numpy as np

class ParameterizedBench:
    params = ([1, 2], [10, 100])
    param_names = ['multiplier', 'size']

    def setup(self, multiplier, size):
        self.multiplier = multiplier
        self.data = np.arange(size)

    def time_compute(self):
        return self.data * self.multiplier
""")

        discovery = BenchmarkDiscovery(workspace['benchmarks'])
        benchmarks = discovery.discover_all()
        assert len(benchmarks) == 1

        # Generate parameter combinations
        param_combinations = discovery.generate_parameter_combinations(benchmarks[0])
        assert len(param_combinations) == 4  # 2 * 2

        # Run and capture all combinations
        runner = BenchmarkRunner(workspace['benchmarks'])
        storage = SnapshotManager(workspace['snapshots'])

        for params in param_combinations:
            result = runner.run_benchmark(benchmarks[0], params)
            assert result.success is True

            storage.store_snapshot(
                benchmark_name=benchmarks[0].name,
                module_path=benchmarks[0].module_path,
                parameters=params,
                param_names=benchmarks[0].param_names,
                return_value=result.return_value
            )

        # Verify all snapshots were created
        snapshots = storage.list_snapshots()
        assert len(snapshots) == 4

    def test_capture_multiple_benchmarks(self, workspace):
        """Test capturing multiple benchmarks in one file."""
        bench_file = workspace['benchmarks'] / "multiple.py"
        bench_file.write_text("""
def time_bench1():
    return 42

def time_bench2():
    return [1, 2, 3]

def time_bench3():
    return {'key': 'value'}
""")

        discovery = BenchmarkDiscovery(workspace['benchmarks'])
        benchmarks = discovery.discover_all()
        assert len(benchmarks) == 3

        runner = BenchmarkRunner(workspace['benchmarks'])
        storage = SnapshotManager(workspace['snapshots'])

        # Capture all benchmarks
        for benchmark in benchmarks:
            result = runner.run_benchmark(benchmark)
            assert result.success is True

            storage.store_snapshot(
                benchmark_name=benchmark.name,
                module_path=benchmark.module_path,
                parameters=(),
                param_names=None,
                return_value=result.return_value
            )

        # Verify
        snapshots = storage.list_snapshots()
        assert len(snapshots) == 3


class TestVerifyWorkflow:
    """Tests for the verify workflow."""

    def test_verify_unchanged_benchmark(self, workspace):
        """Test verifying a benchmark that hasn't changed."""
        bench_file = workspace['benchmarks'] / "verify.py"
        bench_file.write_text("""
import numpy as np

def time_verify():
    return np.array([1, 2, 3])
""")

        # Initial capture
        discovery = BenchmarkDiscovery(workspace['benchmarks'])
        benchmarks = discovery.discover_all()

        runner = BenchmarkRunner(workspace['benchmarks'])
        result = runner.run_benchmark(benchmarks[0])

        storage = SnapshotManager(workspace['snapshots'])
        storage.store_snapshot(
            benchmark_name=benchmarks[0].name,
            module_path=benchmarks[0].module_path,
            parameters=(),
            param_names=None,
            return_value=result.return_value
        )

        # Re-run and verify
        result2 = runner.run_benchmark(benchmarks[0])
        loaded_value, _ = storage.load_snapshot(
            benchmark_name=benchmarks[0].name,
            module_path=benchmarks[0].module_path,
            parameters=()
        )

        comparator = Comparator()
        comparison = comparator.compare(result2.return_value, loaded_value)
        assert comparison.match is True

    def test_detect_changed_benchmark(self, workspace):
        """Test detecting when a benchmark output changes."""
        bench_file = workspace['benchmarks'] / "changed.py"
        bench_file.write_text("""
def time_changed():
    return 42
""")

        # Initial capture
        discovery = BenchmarkDiscovery(workspace['benchmarks'])
        benchmarks = discovery.discover_all()

        runner = BenchmarkRunner(workspace['benchmarks'])
        result = runner.run_benchmark(benchmarks[0])

        storage = SnapshotManager(workspace['snapshots'])
        storage.store_snapshot(
            benchmark_name=benchmarks[0].name,
            module_path=benchmarks[0].module_path,
            parameters=(),
            param_names=None,
            return_value=result.return_value
        )

        # Modify benchmark
        bench_file.write_text("""
def time_changed():
    return 43  # Changed value!
""")

        # Re-discover and run
        discovery = BenchmarkDiscovery(workspace['benchmarks'])
        benchmarks = discovery.discover_all()

        # Need to reload module
        runner = BenchmarkRunner(workspace['benchmarks'])
        result2 = runner.run_benchmark(benchmarks[0])

        loaded_value, _ = storage.load_snapshot(
            benchmark_name=benchmarks[0].name,
            module_path=benchmarks[0].module_path,
            parameters=()
        )

        comparator = Comparator()
        comparison = comparator.compare(result2.return_value, loaded_value)
        # Should detect the change
        assert comparison.match is False

    def test_verify_with_tolerance(self, workspace):
        """Test verification with numerical tolerance."""
        bench_file = workspace['benchmarks'] / "tolerance.py"
        bench_file.write_text("""
import numpy as np

def time_tolerance():
    # Simulate small numerical differences
    return np.array([1.0, 2.0, 3.0])
""")

        # Initial capture
        discovery = BenchmarkDiscovery(workspace['benchmarks'])
        benchmarks = discovery.discover_all()

        runner = BenchmarkRunner(workspace['benchmarks'])
        result = runner.run_benchmark(benchmarks[0])

        storage = SnapshotManager(workspace['snapshots'])
        storage.store_snapshot(
            benchmark_name=benchmarks[0].name,
            module_path=benchmarks[0].module_path,
            parameters=(),
            param_names=None,
            return_value=result.return_value
        )

        # Create slightly different value
        new_value = np.array([1.0000001, 2.0000001, 3.0000001])

        loaded_value, _ = storage.load_snapshot(
            benchmark_name=benchmarks[0].name,
            module_path=benchmarks[0].module_path,
            parameters=()
        )

        # Should pass with default tolerance
        comparator = Comparator(rtol=1e-5, atol=1e-7)
        comparison = comparator.compare(new_value, loaded_value)
        assert comparison.match is True


class TestComplexScenarios:
    """Tests for complex real-world scenarios."""

    def test_mixed_benchmark_types(self, workspace):
        """Test file with both functions and classes."""
        bench_file = workspace['benchmarks'] / "mixed.py"
        bench_file.write_text("""
import numpy as np

def time_function():
    return np.array([1, 2, 3])

class BenchmarkClass:
    params = ([10, 100],)
    param_names = ['size']

    def setup(self, size):
        self.data = np.arange(size)

    def time_method(self):
        return np.sum(self.data)
""")

        discovery = BenchmarkDiscovery(workspace['benchmarks'])
        benchmarks = discovery.discover_all()

        # Should find both function and methods
        assert len(benchmarks) == 2
        types = {b.benchmark_type for b in benchmarks}
        assert types == {'function', 'method'}

        runner = BenchmarkRunner(workspace['benchmarks'])
        storage = SnapshotManager(workspace['snapshots'])

        # Capture all
        for benchmark in benchmarks:
            if benchmark.params:
                param_combinations = discovery.generate_parameter_combinations(benchmark)
                for params in param_combinations:
                    result = runner.run_benchmark(benchmark, params)
                    if result.success:
                        storage.store_snapshot(
                            benchmark_name=benchmark.name,
                            module_path=benchmark.module_path,
                            parameters=params,
                            param_names=benchmark.param_names,
                            return_value=result.return_value
                        )
            else:
                result = runner.run_benchmark(benchmark)
                if result.success:
                    storage.store_snapshot(
                        benchmark_name=benchmark.name,
                        module_path=benchmark.module_path,
                        parameters=(),
                        param_names=None,
                        return_value=result.return_value
                    )

    def test_nested_module_structure(self, workspace):
        """Test benchmarks in nested directories."""
        # Create nested structure
        subdir = workspace['benchmarks'] / "submodule"
        subdir.mkdir()

        (subdir / "nested_bench.py").write_text("""
def time_nested():
    return "nested_result"
""")

        discovery = BenchmarkDiscovery(workspace['benchmarks'])
        benchmarks = discovery.discover_all()

        assert len(benchmarks) == 1
        assert "submodule" in benchmarks[0].module_path

        runner = BenchmarkRunner(workspace['benchmarks'])
        result = runner.run_benchmark(benchmarks[0])
        assert result.success is True

    def test_failed_benchmark_handling(self, workspace):
        """Test handling of benchmarks that fail."""
        bench_file = workspace['benchmarks'] / "failing.py"
        bench_file.write_text("""
def time_failing():
    raise ValueError("This benchmark fails")
""")

        discovery = BenchmarkDiscovery(workspace['benchmarks'])
        benchmarks = discovery.discover_all()

        runner = BenchmarkRunner(workspace['benchmarks'])
        result = runner.run_benchmark(benchmarks[0])

        assert result.success is False
        assert result.error is not None

        # Store failed capture
        storage = SnapshotManager(workspace['snapshots'])
        storage.store_failed_capture(
            benchmark_name=benchmarks[0].name,
            module_path=benchmarks[0].module_path,
            parameters=(),
            param_names=None,
            failure_reason=result.error
        )

        # Verify it's marked as failed
        assert storage.is_failed_capture(
            benchmark_name=benchmarks[0].name,
            module_path=benchmarks[0].module_path,
            parameters=()
        )


class TestDataTypeRoundTrip:
    """Tests for round-trip data integrity."""

    def test_numpy_array_round_trip(self, workspace):
        """Test numpy array survives capture and load."""
        bench_file = workspace['benchmarks'] / "numpy_test.py"
        bench_file.write_text("""
import numpy as np

def time_numpy():
    return np.array([[1, 2, 3], [4, 5, 6]])
""")

        discovery = BenchmarkDiscovery(workspace['benchmarks'])
        benchmarks = discovery.discover_all()

        runner = BenchmarkRunner(workspace['benchmarks'])
        result = runner.run_benchmark(benchmarks[0])

        storage = SnapshotManager(workspace['snapshots'])
        storage.store_snapshot(
            benchmark_name=benchmarks[0].name,
            module_path=benchmarks[0].module_path,
            parameters=(),
            param_names=None,
            return_value=result.return_value
        )

        loaded, _ = storage.load_snapshot(
            benchmark_name=benchmarks[0].name,
            module_path=benchmarks[0].module_path,
            parameters=()
        )

        assert isinstance(loaded, np.ndarray)
        assert np.array_equal(loaded, result.return_value)
        assert loaded.dtype == result.return_value.dtype
        assert loaded.shape == result.return_value.shape

    def test_complex_dict_round_trip(self, workspace):
        """Test complex nested structure round trip."""
        bench_file = workspace['benchmarks'] / "dict_test.py"
        bench_file.write_text("""
import numpy as np

def time_complex():
    return {
        'arrays': [np.array([1, 2, 3]), np.array([4, 5, 6])],
        'nested': {'a': [1, 2], 'b': {'c': 3}},
        'number': 42,
        'string': 'test'
    }
""")

        discovery = BenchmarkDiscovery(workspace['benchmarks'])
        benchmarks = discovery.discover_all()

        runner = BenchmarkRunner(workspace['benchmarks'])
        result = runner.run_benchmark(benchmarks[0])

        storage = SnapshotManager(workspace['snapshots'])
        storage.store_snapshot(
            benchmark_name=benchmarks[0].name,
            module_path=benchmarks[0].module_path,
            parameters=(),
            param_names=None,
            return_value=result.return_value
        )

        loaded, _ = storage.load_snapshot(
            benchmark_name=benchmarks[0].name,
            module_path=benchmarks[0].module_path,
            parameters=()
        )

        comparator = Comparator()
        comparison = comparator.compare(loaded, result.return_value)
        assert comparison.match is True


class TestParameterCombinations:
    """Tests for complex parameter scenarios."""

    def test_multiple_parameter_dimensions(self, workspace):
        """Test benchmarks with multiple parameter dimensions."""
        bench_file = workspace['benchmarks'] / "multi_param.py"
        bench_file.write_text("""
import numpy as np

class MultiParam:
    params = ([1, 2, 3], ['a', 'b'], [True, False])
    param_names = ['num', 'letter', 'flag']

    def setup(self, num, letter, flag):
        self.num = num
        self.letter = letter
        self.flag = flag

    def time_multi(self):
        return f"{self.letter}{self.num}_{self.flag}"
""")

        discovery = BenchmarkDiscovery(workspace['benchmarks'])
        benchmarks = discovery.discover_all()

        param_combinations = discovery.generate_parameter_combinations(benchmarks[0])
        # 3 * 2 * 2 = 12 combinations
        assert len(param_combinations) == 12

        runner = BenchmarkRunner(workspace['benchmarks'])
        storage = SnapshotManager(workspace['snapshots'])

        # Capture all combinations
        for params in param_combinations:
            result = runner.run_benchmark(benchmarks[0], params)
            assert result.success is True

            storage.store_snapshot(
                benchmark_name=benchmarks[0].name,
                module_path=benchmarks[0].module_path,
                parameters=params,
                param_names=benchmarks[0].param_names,
                return_value=result.return_value
            )

        # Verify all were stored
        snapshots = storage.list_snapshots()
        assert len(snapshots) == 12

    def test_parameter_uniqueness(self, workspace):
        """Test that different parameters create unique snapshots."""
        bench_file = workspace['benchmarks'] / "unique_params.py"
        bench_file.write_text("""
class UniqueParams:
    params = ([1, 2],)
    param_names = ['x']

    def setup(self, x):
        self.x = x

    def time_unique(self):
        return self.x * 10
""")

        discovery = BenchmarkDiscovery(workspace['benchmarks'])
        benchmarks = discovery.discover_all()

        runner = BenchmarkRunner(workspace['benchmarks'])
        storage = SnapshotManager(workspace['snapshots'])

        param_combinations = discovery.generate_parameter_combinations(benchmarks[0])

        results = {}
        for params in param_combinations:
            result = runner.run_benchmark(benchmarks[0], params)
            results[params] = result.return_value

            storage.store_snapshot(
                benchmark_name=benchmarks[0].name,
                module_path=benchmarks[0].module_path,
                parameters=params,
                param_names=benchmarks[0].param_names,
                return_value=result.return_value
            )

        # Verify each parameter has unique result
        for params, expected_value in results.items():
            loaded, _ = storage.load_snapshot(
                benchmark_name=benchmarks[0].name,
                module_path=benchmarks[0].module_path,
                parameters=params
            )
            assert loaded == expected_value


class TestEndToEnd:
    """Complete end-to-end workflow tests."""

    def test_full_capture_verify_cycle(self, workspace):
        """Test a complete capture and verify cycle."""
        # Create benchmarks
        bench_file = workspace['benchmarks'] / "full_test.py"
        bench_file.write_text("""
import numpy as np

def time_simple():
    return np.array([1, 2, 3])

class ParamBench:
    params = ([10, 100],)
    param_names = ['size']

    def setup(self, size):
        self.size = size

    def time_param(self):
        return np.arange(self.size)
""")

        # Phase 1: Discovery
        discovery = BenchmarkDiscovery(workspace['benchmarks'])
        benchmarks = discovery.discover_all()
        assert len(benchmarks) == 2

        # Phase 2: Capture
        runner = BenchmarkRunner(workspace['benchmarks'])
        storage = SnapshotManager(workspace['snapshots'])

        captured_count = 0
        for benchmark in benchmarks:
            if benchmark.params:
                param_combinations = discovery.generate_parameter_combinations(benchmark)
                for params in param_combinations:
                    result = runner.run_benchmark(benchmark, params)
                    if result.success:
                        storage.store_snapshot(
                            benchmark_name=benchmark.name,
                            module_path=benchmark.module_path,
                            parameters=params,
                            param_names=benchmark.param_names,
                            return_value=result.return_value
                        )
                        captured_count += 1
            else:
                result = runner.run_benchmark(benchmark)
                if result.success:
                    storage.store_snapshot(
                        benchmark_name=benchmark.name,
                        module_path=benchmark.module_path,
                        parameters=(),
                        param_names=None,
                        return_value=result.return_value
                    )
                    captured_count += 1

        assert captured_count == 3  # 1 simple + 2 parameterized

        # Phase 3: Verify
        comparator = Comparator()
        verify_count = 0

        for benchmark in benchmarks:
            if benchmark.params:
                param_combinations = discovery.generate_parameter_combinations(benchmark)
                for params in param_combinations:
                    result = runner.run_benchmark(benchmark, params)
                    if result.success:
                        loaded_value, _ = storage.load_snapshot(
                            benchmark_name=benchmark.name,
                            module_path=benchmark.module_path,
                            parameters=params
                        )
                        comparison = comparator.compare(result.return_value, loaded_value)
                        assert comparison.match is True
                        verify_count += 1
            else:
                result = runner.run_benchmark(benchmark)
                if result.success:
                    loaded_value, _ = storage.load_snapshot(
                        benchmark_name=benchmark.name,
                        module_path=benchmark.module_path,
                        parameters=()
                    )
                    comparison = comparator.compare(result.return_value, loaded_value)
                    assert comparison.match is True
                    verify_count += 1

        assert verify_count == 3
