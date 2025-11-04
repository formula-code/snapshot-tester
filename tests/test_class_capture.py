#!/usr/bin/env python3
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')
"""
Test script to verify class instance capture and failed capture handling.
"""

from pathlib import Path

from snapshot_tool import BenchmarkDiscovery, BenchmarkRunner, SnapshotManager


def test_class_instance_capture():
    """Test that class instances are properly captured."""
    logger.info("Testing class instance capture...")

    # Initialize components
    benchmark_dir = Path(__file__).parent
    runner = BenchmarkRunner(benchmark_dir)
    storage = SnapshotManager(benchmark_dir / ".snapshots")

    # Discover benchmarks
    discovery = BenchmarkDiscovery(benchmark_dir)
    benchmarks = discovery.discover_all()

    logger.info(f"Found {len(benchmarks)} benchmarks")

    for benchmark in benchmarks:
        logger.info(f"\nTesting benchmark: {benchmark.name}")

        # Run the benchmark
        result = runner.run_benchmark(benchmark)

        if result and result.success:
            logger.info("  ✓ Benchmark executed successfully")
            logger.info(f"  Return value type: {type(result.return_value)}")
            logger.info(f"  Return value: {result.return_value}")

            # Check if it's a class instance
            if hasattr(result.return_value, "__class__"):
                logger.info(f"  Class name: {result.return_value.__class__.__name__}")
                logger.info(f"  Module: {result.return_value.__class__.__module__}")

            # Store the snapshot
            storage.store_snapshot(
                benchmark_name=benchmark.name,
                module_path=benchmark.module_path,
                parameters=(),
                param_names=None,
                return_value=result.return_value,
            )
            logger.info("  ✓ Snapshot stored successfully")

        else:
            logger.info("  ✗ Benchmark failed")
            if result and result.error:
                logger.info(f"  Error: {result.error}")

    logger.info("\nTesting snapshot loading...")

    # Test loading snapshots
    for benchmark in benchmarks:
        snapshot_data = storage.load_snapshot(
            benchmark_name=benchmark.name, module_path=benchmark.module_path, parameters=()
        )

        if snapshot_data:
            return_value, metadata = snapshot_data
            logger.info(f"  ✓ Loaded snapshot for {benchmark.name}")
            logger.info(f"    Type: {type(return_value)}")
            logger.info(f"    Value: {return_value}")
        else:
            logger.info(f"  ✗ Failed to load snapshot for {benchmark.name}")


def test_failed_capture():
    """Test failed capture handling."""
    logger.info("\n\nTesting failed capture handling...")

    # Create a benchmark that will fail
    benchmark_code = '''
def time_failing_benchmark():
    """A benchmark that will fail."""
    raise ValueError("This benchmark intentionally fails")
'''

    # Write the failing benchmark
    failing_file = Path(__file__).parent / "test_failing.py"
    with open(failing_file, "w") as f:
        f.write(benchmark_code)

    try:
        # Initialize components
        benchmark_dir = Path(__file__).parent
        runner = BenchmarkRunner(benchmark_dir)
        storage = SnapshotManager(benchmark_dir / ".snapshots")

        # Discover benchmarks
        discovery = BenchmarkDiscovery(benchmark_dir)
        benchmarks = discovery.discover_all()

        # Find the failing benchmark
        failing_benchmark = None
        for benchmark in benchmarks:
            if benchmark.name == "time_failing_benchmark":
                failing_benchmark = benchmark
                break

        if failing_benchmark:
            logger.info(f"Found failing benchmark: {failing_benchmark.name}")

            # Run the benchmark (should fail)
            result = runner.run_benchmark(failing_benchmark)

            if result and not result.success:
                logger.info("  ✓ Benchmark failed as expected")
                logger.info(f"  Error: {result.error}")

                # Store failed capture
                storage.store_failed_capture(
                    benchmark_name=failing_benchmark.name,
                    module_path=failing_benchmark.module_path,
                    parameters=(),
                    param_names=None,
                    failure_reason=str(result.error),
                )
                logger.info("  ✓ Failed capture marker stored")

                # Test that we can detect failed captures
                is_failed = storage.is_failed_capture(
                    failing_benchmark.name, failing_benchmark.module_path, ()
                )
                logger.info(f"  ✓ Failed capture detection: {is_failed}")
            else:
                logger.info("  ✗ Benchmark should have failed but didn't")
        else:
            logger.info("  ✗ Failed benchmark not found")

    finally:
        # Clean up
        if failing_file.exists():
            failing_file.unlink()


if __name__ == "__main__":
    test_class_instance_capture()
    test_failed_capture()
    logger.info("\n✓ All tests completed!")
