#!/usr/bin/env python3
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')
"""
Simple test script to verify the snapshot testing tool fixes.
"""

import sys
from pathlib import Path

from snapshot_tool import BenchmarkDiscovery, BenchmarkRunner, SnapshotManager


def test_simple_benchmark():
    """Test with a simple benchmark that should work."""
    logger.info("Testing simple benchmark...")

    benchmark_dir = Path(
        "/mnt/sdd1/jamesh/formulacode/diff-tester/examples/astropy-benchmarks/benchmarks"
    )

    discovery = BenchmarkDiscovery(benchmark_dir)
    benchmarks = discovery.discover_all()

    # Find a simple function benchmark
    simple_benchmark = None
    for benchmark in benchmarks:
        if (
            benchmark.benchmark_type == "function"
            and benchmark.module_path == "coordinates"
            and benchmark.name
            in [
                "time_latitude",
                "time_angle_array_repr",
                "time_angle_array_str",
                "time_angle_array_repr_latex",
            ]
        ):
            simple_benchmark = benchmark
            break

    if not simple_benchmark:
        logger.info("No simple benchmark found")
        return False

    logger.info(f"Testing benchmark: {simple_benchmark.name}")

    runner = BenchmarkRunner(benchmark_dir)
    result = runner.run_benchmark(simple_benchmark)

    if result and result.success:
        logger.info(f"✓ Successfully captured return value: {type(result.return_value)}")
        logger.info(f"  Function: {result.function_name}")
        logger.info(f"  Module: {result.module_name}")
        logger.info(f"  Depth: {result.depth}")
        return True
    else:
        logger.info("✗ Failed to capture return value")
        if result:
            logger.info(f"  Error: {result.error}")
        return False


def test_parameterized_benchmark():
    """Test with a parameterized benchmark."""
    logger.info("\nTesting parameterized benchmark...")

    benchmark_dir = Path(
        "/mnt/sdd1/jamesh/formulacode/diff-tester/examples/astropy-benchmarks/benchmarks"
    )

    discovery = BenchmarkDiscovery(benchmark_dir)
    benchmarks = discovery.discover_all()

    # Find a parameterized benchmark
    param_benchmark = None
    for benchmark in benchmarks:
        if (
            benchmark.benchmark_type == "method"
            and benchmark.params
            and benchmark.module_path == "convolve"
        ):
            param_benchmark = benchmark
            break

    if not param_benchmark:
        logger.info("No parameterized benchmark found")
        return False

    logger.info(f"Testing benchmark: {param_benchmark.name}")

    runner = BenchmarkRunner(benchmark_dir)
    param_combinations = discovery.generate_parameter_combinations(param_benchmark)

    # Test with first parameter combination
    if param_combinations:
        params = param_combinations[0]
        logger.info(f"  Testing with params: {params}")
        result = runner.run_benchmark(param_benchmark, params)

        if result and result.success:
            logger.info(f"✓ Successfully captured return value: {type(result.return_value)}")
            logger.info(f"  Function: {result.function_name}")
            logger.info(f"  Module: {result.module_name}")
            logger.info(f"  Depth: {result.depth}")
            return True
        else:
            logger.info("✗ Failed to capture return value")
            if result:
                logger.info(f"  Error: {result.error}")
            return False
    else:
        logger.info("No parameter combinations found")
        return False


def test_storage():
    """Test snapshot storage."""
    logger.info("\nTesting snapshot storage...")

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

    logger.info(f"✓ Stored snapshot at: {snapshot_path}")

    # Test loading the snapshot
    loaded_data = storage.load_snapshot(
        benchmark_name="test_benchmark", module_path="test_module", parameters=(1, 2, 3)
    )

    if loaded_data:
        return_value, metadata = loaded_data
        logger.info(f"✓ Loaded return value: {return_value}")
        logger.info(f"✓ Metadata: {metadata.benchmark_name}, {metadata.timestamp}")

        # Clean up
        import shutil

        if snapshot_dir.exists():
            shutil.rmtree(snapshot_dir)

        return True
    else:
        logger.info("✗ Failed to load snapshot")
        return False


def main():
    """Run all tests."""
    logger.info("Testing snapshot testing tool fixes...")

    success_count = 0
    total_tests = 3

    try:
        if test_simple_benchmark():
            success_count += 1

        if test_parameterized_benchmark():
            success_count += 1

        if test_storage():
            success_count += 1

        logger.info(f"\nTest Results: {success_count}/{total_tests} tests passed")

        if success_count == total_tests:
            logger.info("✓ All tests passed!")
            return 0
        else:
            logger.info("✗ Some tests failed")
            return 1

    except Exception as e:
        logger.info(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
