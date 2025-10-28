# Snapshot Testing Tool for ASV Benchmarks

A tool for capturing and comparing function outputs from ASV benchmarks to verify correctness after optimizations.

## Overview

This tool integrates with ASV (airspeed-velocity) benchmarks to:
- Capture function return values before optimization
- Compare outputs after optimization to ensure correctness
- Use execution tracing to capture the deepest function call's return value
- Support both function-level and class-based parameterized benchmarks

## Installation

The tool is implemented as a Python package. Make sure you have the required dependencies:

```bash
pip install numpy
```

## Usage

### Command Line Interface

The main CLI script is `snapshot_test.py`. Here are the available commands:

#### List Benchmarks
```bash
python snapshot_test.py list <benchmark_dir> [--filter PATTERN]
```

Example:
```bash
python snapshot_test.py list examples/astropy-benchmarks/benchmarks --filter time_angle
```

#### Capture Snapshots
```bash
python snapshot_test.py capture <benchmark_dir> [--filter PATTERN] [--snapshot-dir DIR]
```

This captures baseline snapshots of all benchmark outputs.

#### Verify Against Snapshots
```bash
python snapshot_test.py verify <benchmark_dir> [--filter PATTERN] [--snapshot-dir DIR] [--tolerance RTOL ATOL]
```

This runs benchmarks and compares outputs against stored snapshots.

#### Configuration
```bash
python snapshot_test.py config --init    # Create default config
python snapshot_test.py config --show    # Show current config
```

### Programmatic Usage

```python
from diff_tester.snapshot import BenchmarkDiscovery, BenchmarkRunner, SnapshotManager, Comparator

# Discover benchmarks
discovery = BenchmarkDiscovery("benchmarks/")
benchmarks = discovery.discover_all()

# Run benchmarks with tracing
runner = BenchmarkRunner("benchmarks/")
result = runner.run_benchmark(benchmark)

# Store snapshots
storage = SnapshotManager(".snapshots/")
storage.store_snapshot(
    benchmark_name="my_benchmark",
    module_path="my_module", 
    parameters=(),
    param_names=None,
    return_value=result.return_value
)

# Compare outputs
comparator = Comparator()
comparison = comparator.compare(actual_value, expected_value)
```

## Architecture

### Core Components

1. **BenchmarkDiscovery**: Parses ASV benchmark files to find benchmark classes and functions
2. **ExecutionTracer**: Uses `sys.settrace()` to capture the deepest function call's return value
3. **BenchmarkRunner**: Executes benchmarks with tracing enabled
4. **SnapshotManager**: Stores and retrieves snapshots using pickle files
5. **Comparator**: Compares outputs using numpy.allclose for numerical data
6. **CLI**: Command-line interface for easy usage

### Key Features

- **Hierarchical Discovery**: Supports nested benchmark directories
- **Parameter Handling**: Automatically generates parameter combinations for parameterized benchmarks
- **Global Variables**: Handles benchmarks that use global variables
- **Setup Methods**: Supports class-based benchmarks with setup methods
- **Flexible Comparison**: Uses numpy.allclose for numerical data with configurable tolerances
- **Metadata Tracking**: Stores git commit, timestamp, and other metadata with snapshots

## Configuration

Create a `snapshot_config.json` file to customize behavior:

```json
{
  "benchmark_dir": "benchmarks/",
  "snapshot_dir": ".snapshots/",
  "tolerance": {
    "rtol": 1e-5,
    "atol": 1e-8,
    "equal_nan": false
  },
  "exclude_benchmarks": ["timeraw_*"],
  "trace_depth_limit": 100,
  "verbose": false,
  "quiet": false
}
```

## Example Workflow

1. **Initial Setup**: Capture baseline snapshots
   ```bash
   python snapshot_test.py capture benchmarks/
   ```

2. **Make Optimizations**: Modify your code to improve performance

3. **Verify Correctness**: Compare outputs against snapshots
   ```bash
   python snapshot_test.py verify benchmarks/
   ```

4. **Review Results**: The tool will report which benchmarks passed/failed

## Limitations

- Requires the benchmarked project to be importable
- Some benchmarks may not return meaningful values to capture
- Execution tracing adds overhead to benchmark runs
- Pickle format may not be compatible across Python versions

## Testing

Run the test script to verify the tool works:

```bash
python test_snapshot_tool.py
```

This will test discovery, storage, comparison, and basic benchmark execution.

## Contributing

The tool is designed to be extensible. Key areas for enhancement:
- Support for more data types in comparison
- Integration with CI/CD systems
- Web-based reporting interface
- Support for distributed snapshot storage
