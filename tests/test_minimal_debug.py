#!/usr/bin/env python3
"""
Test script using minimal benchmarks to verify the snapshot testing tool fixes.
"""

import sys
from pathlib import Path

from snapshot_tool import BenchmarkDiscovery, BenchmarkRunner, SnapshotManager


def test_minimal_benchmarks():
    """Test with minimal benchmarks that don't require external dependencies."""
    print("Testing minimal benchmarks...")

    # Create a temporary benchmark directory
    benchmark_dir = Path("/tmp/test_benchmarks")
    benchmark_dir.mkdir(exist_ok=True)

    # Copy our test benchmark file
    import shutil

    shutil.copy("tests/test_benchmarks.py", benchmark_dir / "test_benchmarks.py")

    discovery = BenchmarkDiscovery(benchmark_dir)
    benchmarks = discovery.discover_all()

    print(f"Found {len(benchmarks)} benchmarks:")
    for benchmark in benchmarks:
        print(f"  {benchmark.module_path}.{benchmark.name} ({benchmark.benchmark_type})")
        if benchmark.params:
            param_combinations = discovery.generate_parameter_combinations(benchmark)
            print(f"    Parameters: {len(param_combinations)} combinations")

    runner = BenchmarkRunner(benchmark_dir)
    success_count = 0

    for benchmark in benchmarks:
        print(f"\nTesting benchmark: {benchmark.name}")

        if benchmark.params:
            # Test parameterized benchmark
            param_combinations = discovery.generate_parameter_combinations(benchmark)
            for i, params in enumerate(param_combinations[:2]):  # Test first 2 combinations
                print(f"  Testing with params: {params}")
                result = runner.run_benchmark(benchmark, params)

                if result and result.success:
                    print(f"  ✓ Successfully captured return value: {type(result.return_value)}")
                    print(f"    Function: {result.function_name}")
                    print(f"    Module: {result.module_name}")
                    print(f"    Depth: {result.depth}")
                    success_count += 1
                else:
                    print("  ✗ Failed to capture return value")
                    if result:
                        print(f"    Error: {result.error}")
        else:
            # Test simple benchmark
            result = runner.run_benchmark(benchmark)

            if result and result.success:
                print(f"  ✓ Successfully captured return value: {type(result.return_value)}")
                print(f"    Function: {result.function_name}")
                print(f"    Module: {result.module_name}")
                print(f"    Depth: {result.depth}")
                success_count += 1
            else:
                print("  ✗ Failed to capture return value")
                if result:
                    print(f"    Error: {result.error}")

    # Clean up
    shutil.rmtree(benchmark_dir)

    return success_count, len(benchmarks)


def test_storage():
    """Test snapshot storage."""
    print("\nTesting snapshot storage...")

    snapshot_dir = Path("/tmp/test_snapshots_debug")
    storage = SnapshotManager(snapshot_dir)

    # Test storing a snapshot
    test_data = {"test": "data", "number": 42}
    snapshot_path = storage.store_snapshot(
        benchmark_name="test_benchmark",
        module_path="test_module",
        parameters=(1, 2, 3),
        param_names=["a", "b", "c"],
        return_value=test_data,
    )

    print(f"✓ Stored snapshot at: {snapshot_path}")

    # Test loading the snapshot
    loaded_data = storage.load_snapshot(
        benchmark_name="test_benchmark", module_path="test_module", parameters=(1, 2, 3)
    )

    if loaded_data:
        return_value, metadata = loaded_data
        print(f"✓ Loaded return value: {return_value}")
        print(f"✓ Metadata: {metadata.benchmark_name}, {metadata.timestamp}")

        # Clean up
        import shutil

        if snapshot_dir.exists():
            shutil.rmtree(snapshot_dir)

        return True
    else:
        print("✗ Failed to load snapshot")
        return False


def main():
    """Run all tests."""
    print("Testing snapshot testing tool fixes with minimal benchmarks...")

    try:
        success_count, total_benchmarks = test_minimal_benchmarks()
        storage_success = test_storage()

        print("\nTest Results:")
        print(f"  Benchmarks: {success_count}/{total_benchmarks} passed")
        print(f"  Storage: {'✓' if storage_success else '✗'}")

        if success_count == total_benchmarks and storage_success:
            print("✓ All tests passed!")
            return 0
        else:
            print("✗ Some tests failed")
            return 1

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
