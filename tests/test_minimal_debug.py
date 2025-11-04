#!/usr/bin/env python3
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')
"""
Test script using minimal benchmarks to verify the snapshot testing tool fixes.
"""

import sys
from pathlib import Path

from snapshot_tool import BenchmarkDiscovery, BenchmarkRunner, SnapshotManager


def test_minimal_benchmarks():
    """Test with minimal benchmarks that don't require external dependencies."""
    logger.info("Testing minimal benchmarks...")

    # Create a temporary benchmark directory
    benchmark_dir = Path("/tmp/test_benchmarks")
    benchmark_dir.mkdir(exist_ok=True)

    # Copy our test benchmark file
    import shutil

    shutil.copy("tests/test_benchmarks.py", benchmark_dir / "test_benchmarks.py")

    discovery = BenchmarkDiscovery(benchmark_dir)
    benchmarks = discovery.discover_all()

    logger.info(f"Found {len(benchmarks)} benchmarks:")
    for benchmark in benchmarks:
        logger.info(f"  {benchmark.module_path}.{benchmark.name} ({benchmark.benchmark_type})")
        if benchmark.params:
            param_combinations = discovery.generate_parameter_combinations(benchmark)
            logger.info(f"    Parameters: {len(param_combinations)} combinations")

    runner = BenchmarkRunner(benchmark_dir)
    success_count = 0

    for benchmark in benchmarks:
        logger.info(f"\nTesting benchmark: {benchmark.name}")

        if benchmark.params:
            # Test parameterized benchmark
            param_combinations = discovery.generate_parameter_combinations(benchmark)
            for i, params in enumerate(param_combinations[:2]):  # Test first 2 combinations
                logger.info(f"  Testing with params: {params}")
                result = runner.run_benchmark(benchmark, params)

                if result and result.success:
                    logger.info(f"  ✓ Successfully captured return value: {type(result.return_value)}")
                    logger.info(f"    Function: {result.function_name}")
                    logger.info(f"    Module: {result.module_name}")
                    logger.info(f"    Depth: {result.depth}")
                    success_count += 1
                else:
                    logger.info("  ✗ Failed to capture return value")
                    if result:
                        logger.info(f"    Error: {result.error}")
        else:
            # Test simple benchmark
            result = runner.run_benchmark(benchmark)

            if result and result.success:
                logger.info(f"  ✓ Successfully captured return value: {type(result.return_value)}")
                logger.info(f"    Function: {result.function_name}")
                logger.info(f"    Module: {result.module_name}")
                logger.info(f"    Depth: {result.depth}")
                success_count += 1
            else:
                logger.info("  ✗ Failed to capture return value")
                if result:
                    logger.info(f"    Error: {result.error}")

    # Clean up
    shutil.rmtree(benchmark_dir)

    return success_count, len(benchmarks)


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
    logger.info("Testing snapshot testing tool fixes with minimal benchmarks...")

    try:
        success_count, total_benchmarks = test_minimal_benchmarks()
        storage_success = test_storage()

        logger.info("\nTest Results:")
        logger.info(f"  Benchmarks: {success_count}/{total_benchmarks} passed")
        logger.info(f"  Storage: {'✓' if storage_success else '✗'}")

        if success_count == total_benchmarks and storage_success:
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
