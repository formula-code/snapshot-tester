#!/usr/bin/env python3
"""
Test script to verify class instance capture and failed capture handling.
"""

import sys
from pathlib import Path

# Add the diff_tester module to the path
sys.path.insert(0, str(Path(__file__).parent))

from diff_tester.snapshot import BenchmarkDiscovery, BenchmarkRunner, SnapshotManager


def test_class_instance_capture():
    """Test that class instances are properly captured."""
    print("Testing class instance capture...")
    
    # Initialize components
    benchmark_dir = Path(__file__).parent
    runner = BenchmarkRunner(benchmark_dir)
    storage = SnapshotManager(benchmark_dir / ".snapshots")
    
    # Discover benchmarks
    discovery = BenchmarkDiscovery(benchmark_dir)
    benchmarks = discovery.discover_all()
    
    print(f"Found {len(benchmarks)} benchmarks")
    
    for benchmark in benchmarks:
        print(f"\nTesting benchmark: {benchmark.name}")
        
        # Run the benchmark
        result = runner.run_benchmark(benchmark)
        
        if result and result.success:
            print(f"  ✓ Benchmark executed successfully")
            print(f"  Return value type: {type(result.return_value)}")
            print(f"  Return value: {result.return_value}")
            
            # Check if it's a class instance
            if hasattr(result.return_value, '__class__'):
                print(f"  Class name: {result.return_value.__class__.__name__}")
                print(f"  Module: {result.return_value.__class__.__module__}")
            
            # Store the snapshot
            storage.store_snapshot(
                benchmark_name=benchmark.name,
                module_path=benchmark.module_path,
                parameters=(),
                param_names=None,
                return_value=result.return_value
            )
            print(f"  ✓ Snapshot stored successfully")
            
        else:
            print(f"  ✗ Benchmark failed")
            if result and result.error:
                print(f"  Error: {result.error}")
    
    print("\nTesting snapshot loading...")
    
    # Test loading snapshots
    for benchmark in benchmarks:
        snapshot_data = storage.load_snapshot(
            benchmark_name=benchmark.name,
            module_path=benchmark.module_path,
            parameters=()
        )
        
        if snapshot_data:
            return_value, metadata = snapshot_data
            print(f"  ✓ Loaded snapshot for {benchmark.name}")
            print(f"    Type: {type(return_value)}")
            print(f"    Value: {return_value}")
        else:
            print(f"  ✗ Failed to load snapshot for {benchmark.name}")


def test_failed_capture():
    """Test failed capture handling."""
    print("\n\nTesting failed capture handling...")
    
    # Create a benchmark that will fail
    benchmark_code = '''
def time_failing_benchmark():
    """A benchmark that will fail."""
    raise ValueError("This benchmark intentionally fails")
'''
    
    # Write the failing benchmark
    failing_file = Path(__file__).parent / "test_failing.py"
    with open(failing_file, 'w') as f:
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
            print(f"Found failing benchmark: {failing_benchmark.name}")
            
            # Run the benchmark (should fail)
            result = runner.run_benchmark(failing_benchmark)
            
            if result and not result.success:
                print(f"  ✓ Benchmark failed as expected")
                print(f"  Error: {result.error}")
                
                # Store failed capture
                storage.store_failed_capture(
                    benchmark_name=failing_benchmark.name,
                    module_path=failing_benchmark.module_path,
                    parameters=(),
                    param_names=None,
                    failure_reason=str(result.error)
                )
                print(f"  ✓ Failed capture marker stored")
                
                # Test that we can detect failed captures
                is_failed = storage.is_failed_capture(
                    failing_benchmark.name,
                    failing_benchmark.module_path,
                    ()
                )
                print(f"  ✓ Failed capture detection: {is_failed}")
            else:
                print(f"  ✗ Benchmark should have failed but didn't")
        else:
            print(f"  ✗ Failed benchmark not found")
    
    finally:
        # Clean up
        if failing_file.exists():
            failing_file.unlink()


if __name__ == "__main__":
    test_class_instance_capture()
    test_failed_capture()
    print("\n✓ All tests completed!")
