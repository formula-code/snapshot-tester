"""Integration tests for end-to-end workflows."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from snapshot_tool import (
    BenchmarkDiscovery,
    BenchmarkRunner,
    Comparator,
    SnapshotManager,
)
from snapshot_tool.comparator import ComparisonConfig


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
        config = ComparisonConfig(rtol=1e-5, atol=1e-7)
        comparator = Comparator(config)
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


class TestShapelyRepo:
    """Integration tests using the real shapely repository."""

    @pytest.fixture
    def shapely_workspace(self):
        """Set up workspace for shapely benchmarks."""
        # Use the shapely benchmarks (isolated from source to avoid import conflicts)
        benchmark_dir = Path(__file__).parent / "test_repos" / "shapely_benchmarks"

        # Create a temporary snapshot directory
        temp_dir = tempfile.mkdtemp()
        snapshot_dir = Path(temp_dir) / ".snapshots"
        snapshot_dir.mkdir()

        yield {
            'benchmarks': benchmark_dir,
            'snapshots': snapshot_dir
        }

        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_shapely_benchmark_discovery(self, shapely_workspace):
        """Test that we can discover shapely benchmarks."""
        discovery = BenchmarkDiscovery(shapely_workspace['benchmarks'])
        benchmarks = discovery.discover_all()

        # Shapely has multiple benchmark classes
        assert len(benchmarks) > 0

        # Check we found some expected benchmarks
        benchmark_names = {b.name for b in benchmarks}

        # These are classes from the shapely benchmarks
        expected_classes = {'PointPolygonTimeSuite', 'IOSuite', 'ConstructorsSuite'}
        found_classes = {b.class_name for b in benchmarks if b.class_name}

        # At least some of these should be found
        assert len(expected_classes.intersection(found_classes)) > 0

    def test_shapely_capture_and_verify_subset(self, shapely_workspace):
        """Test capturing and verifying a subset of shapely benchmarks."""
        discovery = BenchmarkDiscovery(shapely_workspace['benchmarks'])
        benchmarks = discovery.discover_all()

        # Filter to a small, fast benchmark for testing
        # ConstructorsSuite.time_point is a simple microbenchmark
        test_benchmark = None
        for b in benchmarks:
            if b.class_name == 'ConstructorsSuite' and b.name == 'time_point':
                test_benchmark = b
                break

        if test_benchmark is None:
            pytest.skip("Could not find ConstructorsSuite.time_point benchmark")

        runner = BenchmarkRunner(shapely_workspace['benchmarks'])
        storage = SnapshotManager(shapely_workspace['snapshots'])

        # Capture the benchmark
        result = runner.run_benchmark(test_benchmark)

        # The benchmark should run successfully
        assert result.success is True
        assert result.return_value is not None

        # Store the snapshot
        storage.store_snapshot(
            benchmark_name=test_benchmark.name,
            module_path=test_benchmark.module_path,
            parameters=(),
            param_names=None,
            return_value=result.return_value
        )

        # Verify the snapshot
        result2 = runner.run_benchmark(test_benchmark)
        assert result2.success is True

        loaded_value, _ = storage.load_snapshot(
            benchmark_name=test_benchmark.name,
            module_path=test_benchmark.module_path,
            parameters=()
        )

        comparator = Comparator()
        comparison = comparator.compare(result2.return_value, loaded_value)
        assert comparison.match is True

    def test_shapely_determinism_multiple_verifies(self, shapely_workspace):
        """Test that verification is deterministic - running verify twice should always pass."""
        discovery = BenchmarkDiscovery(shapely_workspace['benchmarks'])
        benchmarks = discovery.discover_all()

        # Find a simple benchmark to test
        test_benchmark = None
        for b in benchmarks:
            if b.class_name == 'ConstructorsSuite' and b.name == 'time_point':
                test_benchmark = b
                break

        if test_benchmark is None:
            pytest.skip("Could not find ConstructorsSuite.time_point benchmark")

        runner = BenchmarkRunner(shapely_workspace['benchmarks'])
        storage = SnapshotManager(shapely_workspace['snapshots'])
        comparator = Comparator()

        # Initial capture
        capture_result = runner.run_benchmark(test_benchmark)
        assert capture_result.success is True

        storage.store_snapshot(
            benchmark_name=test_benchmark.name,
            module_path=test_benchmark.module_path,
            parameters=(),
            param_names=None,
            return_value=capture_result.return_value
        )

        # First verification
        verify1_result = runner.run_benchmark(test_benchmark)
        assert verify1_result.success is True

        loaded_value1, _ = storage.load_snapshot(
            benchmark_name=test_benchmark.name,
            module_path=test_benchmark.module_path,
            parameters=()
        )

        comparison1 = comparator.compare(verify1_result.return_value, loaded_value1)
        assert comparison1.match is True, "First verification should pass"

        # Second verification (testing determinism)
        verify2_result = runner.run_benchmark(test_benchmark)
        assert verify2_result.success is True

        loaded_value2, _ = storage.load_snapshot(
            benchmark_name=test_benchmark.name,
            module_path=test_benchmark.module_path,
            parameters=()
        )

        comparison2 = comparator.compare(verify2_result.return_value, loaded_value2)
        assert comparison2.match is True, "Second verification should pass (determinism check)"

        # Third verification for good measure
        verify3_result = runner.run_benchmark(test_benchmark)
        assert verify3_result.success is True

        loaded_value3, _ = storage.load_snapshot(
            benchmark_name=test_benchmark.name,
            module_path=test_benchmark.module_path,
            parameters=()
        )

        comparison3 = comparator.compare(verify3_result.return_value, loaded_value3)
        assert comparison3.match is True, "Third verification should pass (determinism check)"

    def test_shapely_with_setup_method(self, shapely_workspace):
        """Test a shapely benchmark that has a setup method - uses time_distance which returns numpy arrays."""
        discovery = BenchmarkDiscovery(shapely_workspace['benchmarks'])
        benchmarks = discovery.discover_all()

        # Find PointPolygonTimeSuite.time_distance which has setup and returns numpy array
        test_benchmark = None
        for b in benchmarks:
            if b.class_name == 'PointPolygonTimeSuite' and b.name == 'time_distance':
                test_benchmark = b
                break

        if test_benchmark is None:
            pytest.skip("Could not find PointPolygonTimeSuite.time_distance benchmark")

        runner = BenchmarkRunner(shapely_workspace['benchmarks'])
        storage = SnapshotManager(shapely_workspace['snapshots'])

        # Run the benchmark (it should handle setup internally)
        result = runner.run_benchmark(test_benchmark)

        # Should succeed and return a value
        assert result.success is True
        assert result.return_value is not None

        # Store and verify
        storage.store_snapshot(
            benchmark_name=test_benchmark.name,
            module_path=test_benchmark.module_path,
            parameters=(),
            param_names=None,
            return_value=result.return_value
        )

        # Verify determinism - important for benchmarks with random data in setup
        result2 = runner.run_benchmark(test_benchmark)
        assert result2.success is True

        loaded_value, _ = storage.load_snapshot(
            benchmark_name=test_benchmark.name,
            module_path=test_benchmark.module_path,
            parameters=()
        )

        comparator = Comparator()
        comparison = comparator.compare(result2.return_value, loaded_value)
        assert comparison.match is True

    def test_shapely_full_capture_and_verify_cycle(self, shapely_workspace):
        """Full integration test: capture shapely benchmarks and verify determinism."""
        discovery = BenchmarkDiscovery(shapely_workspace['benchmarks'])
        benchmarks = discovery.discover_all()

        # Filter to fast benchmarks
        fast_benchmarks = [
            b for b in benchmarks
            if b.class_name == 'ConstructorsSuite' and b.name in ['time_point', 'time_linestring_from_numpy']
        ]

        if len(fast_benchmarks) == 0:
            pytest.skip("Could not find ConstructorsSuite benchmarks")

        runner = BenchmarkRunner(shapely_workspace['benchmarks'])
        storage = SnapshotManager(shapely_workspace['snapshots'])
        comparator = Comparator()

        # Phase 1: Capture
        captured_benchmarks = []
        for benchmark in fast_benchmarks:
            result = runner.run_benchmark(benchmark)
            if result.success:
                storage.store_snapshot(
                    benchmark_name=benchmark.name,
                    module_path=benchmark.module_path,
                    parameters=(),
                    param_names=None,
                    return_value=result.return_value
                )
                captured_benchmarks.append(benchmark)

        assert len(captured_benchmarks) > 0, "Should capture at least one benchmark"

        # Phase 2: Verify determinism - run verify 3 times
        for verify_round in range(3):
            all_passed = True
            for benchmark in captured_benchmarks:
                result = runner.run_benchmark(benchmark)
                assert result.success, f"Benchmark {benchmark.name} failed on verify round {verify_round + 1}"

                loaded_value, _ = storage.load_snapshot(
                    benchmark_name=benchmark.name,
                    module_path=benchmark.module_path,
                    parameters=()
                )

                comparison = comparator.compare(result.return_value, loaded_value)
                if not comparison.match:
                    all_passed = False
                    print(f"Round {verify_round + 1} failed for {benchmark.name}: {comparison.error_message}")

            assert all_passed, f"Verification round {verify_round + 1} should pass (determinism check)"

    def test_shapely_with_random_data_determinism(self, shapely_workspace):
        """Test that benchmarks using np.random produce deterministic results with seed management."""
        discovery = BenchmarkDiscovery(shapely_workspace['benchmarks'])
        benchmarks = discovery.discover_all()

        # Find PointPolygonTimeSuite.time_distance which uses random data
        test_benchmark = None
        for b in benchmarks:
            if b.class_name == 'PointPolygonTimeSuite' and b.name == 'time_distance':
                test_benchmark = b
                break

        if test_benchmark is None:
            pytest.skip("Could not find PointPolygonTimeSuite.time_distance benchmark")

        runner = BenchmarkRunner(shapely_workspace['benchmarks'])
        comparator = Comparator()

        # Run 3 times - should be identical due to seed reset
        result1 = runner.run_benchmark(test_benchmark)
        assert result1.success is True
        value1 = result1.return_value

        result2 = runner.run_benchmark(test_benchmark)
        assert result2.success is True
        value2 = result2.return_value

        result3 = runner.run_benchmark(test_benchmark)
        assert result3.success is True
        value3 = result3.return_value

        # All three runs should produce identical results
        comparison_1_2 = comparator.compare(value1, value2)
        comparison_2_3 = comparator.compare(value2, value3)

        assert comparison_1_2.match is True, "Runs 1 and 2 should match (seed reset working)"
        assert comparison_2_3.match is True, "Runs 2 and 3 should match (seed reset working)"

    def test_shapely_multiple_benchmarks_verify(self, shapely_workspace):
        """Test running multiple shapely benchmarks and verifying all."""
        discovery = BenchmarkDiscovery(shapely_workspace['benchmarks'])
        benchmarks = discovery.discover_all()

        target_benchmarks = [
            ('ConstructorsSuite', 'time_point'),
            ('ConstructorsSuite', 'time_linestring_from_numpy'),
            ('ConstructorsSuite', 'time_linearring_from_numpy'),
        ]

        selected_benchmarks = []
        for class_name, bench_name in target_benchmarks:
            for b in benchmarks:
                if b.class_name == class_name and b.name == bench_name:
                    selected_benchmarks.append(b)
                    break

        if len(selected_benchmarks) < 2:
            pytest.skip("Could not find enough benchmarks")

        runner = BenchmarkRunner(shapely_workspace['benchmarks'])
        storage = SnapshotManager(shapely_workspace['snapshots'])
        comparator = Comparator()

        # Capture all
        capture_results = {}
        for benchmark in selected_benchmarks:
            result = runner.run_benchmark(benchmark)
            if result.success:
                capture_results[benchmark.name] = result.return_value
                storage.store_snapshot(
                    benchmark_name=benchmark.name,
                    module_path=benchmark.module_path,
                    parameters=(),
                    param_names=None,
                    return_value=result.return_value
                )

        assert len(capture_results) >= 2, "Should capture at least 2 benchmarks"

        # Verify all
        verify_results = {}
        for benchmark in selected_benchmarks:
            if benchmark.name in capture_results:
                result = runner.run_benchmark(benchmark)
                assert result.success is True

                loaded_value, _ = storage.load_snapshot(
                    benchmark_name=benchmark.name,
                    module_path=benchmark.module_path,
                    parameters=()
                )

                comparison = comparator.compare(result.return_value, loaded_value)
                verify_results[benchmark.name] = comparison.match

        all_passed = all(verify_results.values())
        failed_benchmarks = [name for name, passed in verify_results.items() if not passed]

        assert all_passed, f"All benchmarks should pass verification. Failed: {failed_benchmarks}"
