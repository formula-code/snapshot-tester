#!/usr/bin/env python3
"""
Test script for the snapshot testing tool.

This script tests the tool with the provided astropy benchmark examples.
"""

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

import sys
from pathlib import Path

from snapshot_tool import BenchmarkDiscovery, BenchmarkRunner, Comparator, SnapshotManager


def test_discovery():
    """Test benchmark discovery."""
    logger.info("Testing benchmark discovery...")

    benchmark_dir = Path(
        "/mnt/sdd1/jamesh/formulacode/diff-tester/examples/astropy-benchmarks/benchmarks"
    )

    discovery = BenchmarkDiscovery(benchmark_dir)
    benchmarks = discovery.discover_all()

    logger.info(f"Found {len(benchmarks)} benchmarks:")
    for benchmark in benchmarks:
        logger.info(f"  {benchmark.module_path}.{benchmark.name} ({benchmark.benchmark_type})")
        if benchmark.params:
            param_combinations = discovery.generate_parameter_combinations(benchmark)
            logger.info(f"    Parameters: {len(param_combinations)} combinations")

    return benchmarks


def test_runner():
    """Test benchmark runner."""
    logger.info("\nTesting benchmark runner...")

    benchmark_dir = Path(
        "/mnt/sdd1/jamesh/formulacode/diff-tester/examples/astropy-benchmarks/benchmarks"
    )

    runner = BenchmarkRunner(benchmark_dir)

    # Test with a simple function benchmark
    discovery = BenchmarkDiscovery(benchmark_dir)
    benchmarks = discovery.discover_all()

    # Find a simple function benchmark
    simple_benchmark = None
    for benchmark in benchmarks:
        if benchmark.benchmark_type == "function" and not benchmark.params:
            simple_benchmark = benchmark
            break

    if simple_benchmark:
        logger.info(f"Testing simple benchmark: {simple_benchmark.name}")
        result = runner.run_benchmark(simple_benchmark)

        if result:
            logger.info(f"  Captured return value: {type(result.return_value)}")
            logger.info(f"  Function: {result.function_name}")
            logger.info(f"  Module: {result.module_name}")
            logger.info(f"  Depth: {result.depth}")
            logger.info(f"  Success: {result.success}")
        else:
            logger.info("  Failed to capture return value")

    # Test with a parameterized benchmark
    param_benchmark = None
    for benchmark in benchmarks:
        if benchmark.benchmark_type == "method" and benchmark.params:
            param_benchmark = benchmark
            break

    if param_benchmark:
        logger.info(f"\nTesting parameterized benchmark: {param_benchmark.name}")
        param_combinations = discovery.generate_parameter_combinations(param_benchmark)

        # Test with first parameter combination
        if param_combinations:
            params = param_combinations[0]
            logger.info(f"  Testing with params: {params}")
            result = runner.run_benchmark(param_benchmark, params)

            if result:
                logger.info(f"    Captured return value: {type(result.return_value)}")
                logger.info(f"    Function: {result.function_name}")
                logger.info(f"    Module: {result.module_name}")
                logger.info(f"    Depth: {result.depth}")
                logger.info(f"    Success: {result.success}")
            else:
                logger.info("    Failed to capture return value")


def test_storage():
    """Test snapshot storage."""
    logger.info("\nTesting snapshot storage...")

    snapshot_dir = Path("/tmp/test_snapshots")
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

    logger.info(f"Stored snapshot at: {snapshot_path}")

    # Test loading the snapshot
    loaded_data = storage.load_snapshot(
        benchmark_name="test_benchmark", module_path="test_module", parameters=(1, 2, 3)
    )

    if loaded_data:
        return_value, metadata = loaded_data
        logger.info(f"Loaded return value: {return_value}")
        logger.info(f"Metadata: {metadata.benchmark_name}, {metadata.timestamp}")

    # Test listing snapshots
    snapshots = storage.list_snapshots()
    logger.info(f"Total snapshots: {len(snapshots)}")

    # Clean up
    import shutil

    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)


def test_comparator():
    """Test comparison engine."""
    logger.info("\nTesting comparison engine...")

    comparator = Comparator()

    # Test numpy array comparison
    import numpy as np

    arr1 = np.array([1.0, 2.0, 3.0])
    arr2 = np.array([1.000001, 2.000001, 3.000001])

    result = comparator.compare(arr1, arr2)
    logger.info(f"Array comparison: {result.match}")
    if not result.match:
        logger.info(f"  Error: {result.error_message}")

    # Test scalar comparison
    result = comparator.compare(1.0, 1.000001)
    logger.info(f"Scalar comparison: {result.match}")

    # Test exact match
    result = comparator.compare(42, 42)
    logger.info(f"Exact match: {result.match}")

    # Test mismatch
    result = comparator.compare(1.0, 2.0)
    logger.info(f"Mismatch: {result.match}")
    if not result.match:
        logger.info(f"  Error: {result.error_message}")


def main():
    """Run all tests."""
    logger.info("Testing snapshot testing tool...")

    try:
        benchmarks = test_discovery()
        test_runner()
        test_storage()
        test_comparator()

        logger.info("\nAll tests completed!")

    except Exception as e:
        logger.info(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
