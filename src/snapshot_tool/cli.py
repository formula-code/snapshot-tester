"""
Command-line interface for snapshot testing.

This module provides CLI commands for capturing and verifying snapshots
of ASV benchmark outputs.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .runner import BenchmarkRunner
from .storage import SnapshotManager
from .comparator import Comparator, ComparisonConfig
from .config import ConfigManager, SnapshotConfig
from .discovery import BenchmarkDiscovery


class SnapshotCLI:
    """Command-line interface for snapshot testing."""

    def __init__(self):
        self.config_manager = ConfigManager()
        self.config = self.config_manager.get_config()

    def run(self, args: Optional[List[str]] = None) -> int:
        """Run the CLI with given arguments."""
        parser = self._create_parser()
        parsed_args = parser.parse_args(args)

        try:
            return parsed_args.func(parsed_args)
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            return 1
        except Exception as e:
            print(f"Error: {e}")
            if self.config.verbose:
                import traceback

                traceback.print_exc()
            return 1

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser."""
        parser = argparse.ArgumentParser(
            description="Snapshot testing tool for ASV benchmarks",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        parser.add_argument("--config", "-c", type=Path, help="Configuration file path")

        parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

        parser.add_argument("--quiet", "-q", action="store_true", help="Quiet output")

        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # Capture command
        capture_parser = subparsers.add_parser("capture", help="Capture baseline snapshots")
        capture_parser.add_argument(
            "benchmark_dir", type=Path, help="Directory containing benchmark files"
        )
        capture_parser.add_argument("--filter", help="Filter benchmarks by name pattern")
        capture_parser.add_argument(
            "--snapshot-dir", type=Path, help="Directory to store snapshots"
        )
        capture_parser.set_defaults(func=self._capture_command)

        # Verify command
        verify_parser = subparsers.add_parser("verify", help="Verify against existing snapshots")
        verify_parser.add_argument(
            "benchmark_dir", type=Path, help="Directory containing benchmark files"
        )
        verify_parser.add_argument("--filter", help="Filter benchmarks by name pattern")
        verify_parser.add_argument(
            "--snapshot-dir", type=Path, help="Directory containing snapshots"
        )
        verify_parser.add_argument(
            "--tolerance",
            nargs=2,
            metavar=("RTOL", "ATOL"),
            type=float,
            help="Relative and absolute tolerance for comparison",
        )
        verify_parser.set_defaults(func=self._verify_command)

        # List command
        list_parser = subparsers.add_parser("list", help="List available benchmarks")
        list_parser.add_argument(
            "benchmark_dir", type=Path, help="Directory containing benchmark files"
        )
        list_parser.add_argument("--filter", help="Filter benchmarks by name pattern")
        list_parser.set_defaults(func=self._list_command)

        # Clean command
        clean_parser = subparsers.add_parser("clean", help="Clean old snapshots")
        clean_parser.add_argument(
            "--snapshot-dir", type=Path, help="Directory containing snapshots"
        )
        clean_parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be deleted without actually deleting",
        )
        clean_parser.set_defaults(func=self._clean_command)

        # Config command
        config_parser = subparsers.add_parser("config", help="Configuration management")
        config_parser.add_argument(
            "--init", action="store_true", help="Initialize default configuration file"
        )
        config_parser.add_argument("--show", action="store_true", help="Show current configuration")
        config_parser.set_defaults(func=self._config_command)

        return parser

    def _capture_command(self, args) -> int:
        """Handle the capture command."""
        # Update config from command line
        if args.snapshot_dir:
            self.config.snapshot_dir = str(args.snapshot_dir)
        if args.verbose:
            self.config.verbose = True
        if args.quiet:
            self.config.quiet = True

        benchmark_dir = args.benchmark_dir
        snapshot_dir = self.config.get_snapshot_dir()

        print(f"Capturing snapshots from {benchmark_dir}")
        print(f"Storing snapshots in {snapshot_dir}")

        # Initialize components
        runner = BenchmarkRunner(benchmark_dir)
        storage = SnapshotManager(snapshot_dir)

        # Discover benchmarks
        discovery = BenchmarkDiscovery(benchmark_dir)
        benchmarks = discovery.discover_all()

        if args.filter:
            benchmarks = [b for b in benchmarks if args.filter in b.name]

        captured_count = 0

        for benchmark in benchmarks:
            if self.config.should_exclude_benchmark(benchmark.name):
                if not self.config.quiet:
                    print(f"Skipping excluded benchmark: {benchmark.name}")
                continue

            print(f"Capturing: {benchmark.module_path}.{benchmark.name}")

            if benchmark.params or getattr(benchmark, "needs_runtime_eval", False):
                # Capture with all parameter combinations
                param_combinations = runner.get_param_combinations(benchmark)

                for params in param_combinations:
                    result = runner.run_benchmark(benchmark, params)
                    if result and result.success:
                        storage.store_snapshot(
                            benchmark_name=benchmark.name,
                            module_path=benchmark.module_path,
                            parameters=params,
                            param_names=benchmark.param_names,
                            return_value=result.return_value,
                        )
                        captured_count += 1
                        if not self.config.quiet:
                            print(f"  Captured with params: {params}")
                    else:
                        # Store failed capture marker
                        failure_reason = (
                            str(result.error) if result and result.error else "Unknown error"
                        )
                        storage.store_failed_capture(
                            benchmark_name=benchmark.name,
                            module_path=benchmark.module_path,
                            parameters=params,
                            param_names=benchmark.param_names,
                            failure_reason=failure_reason,
                        )
                        print(f"  Failed to capture with params: {params} - {failure_reason}")
                        if self.config.verbose and result and result.error:
                            import traceback

                            print(f"    Full error traceback:")
                            traceback.print_exc()
            else:
                # Capture without parameters
                result = runner.run_benchmark(benchmark)
                if result and result.success:
                    storage.store_snapshot(
                        benchmark_name=benchmark.name,
                        module_path=benchmark.module_path,
                        parameters=(),
                        param_names=None,
                        return_value=result.return_value,
                    )
                    captured_count += 1
                else:
                    # Store failed capture marker
                    failure_reason = (
                        str(result.error) if result and result.error else "Unknown error"
                    )
                    storage.store_failed_capture(
                        benchmark_name=benchmark.name,
                        module_path=benchmark.module_path,
                        parameters=(),
                        param_names=None,
                        failure_reason=failure_reason,
                    )
                    print(f"  Failed to capture - {failure_reason}")
                    if self.config.verbose and result and result.error:
                        import traceback

                        print(f"    Full error traceback:")
                        traceback.print_exc()

        print(f"Captured {captured_count} snapshots")
        return 0

    def _verify_command(self, args) -> int:
        """Handle the verify command."""
        # Update config from command line
        if args.snapshot_dir:
            self.config.snapshot_dir = str(args.snapshot_dir)
        if args.verbose:
            self.config.verbose = True
        if args.quiet:
            self.config.quiet = True

        benchmark_dir = args.benchmark_dir
        snapshot_dir = self.config.get_snapshot_dir()

        print(f"Verifying benchmarks in {benchmark_dir}")
        print(f"Comparing against snapshots in {snapshot_dir}")

        # Initialize components
        runner = BenchmarkRunner(benchmark_dir)
        storage = SnapshotManager(snapshot_dir)

        # Update comparison config
        comp_config = ComparisonConfig()
        if args.tolerance:
            comp_config.rtol = args.tolerance[0]
            comp_config.atol = args.tolerance[1]
        else:
            comp_config.rtol = self.config.tolerance["rtol"]
            comp_config.atol = self.config.tolerance["atol"]
            comp_config.equal_nan = self.config.tolerance.get("equal_nan", False)

        comparator = Comparator(comp_config)

        # Discover benchmarks
        discovery = BenchmarkDiscovery(benchmark_dir)
        benchmarks = discovery.discover_all()

        if args.filter:
            benchmarks = [b for b in benchmarks if args.filter in b.name]

        total_tests = 0
        passed_tests = 0
        failed_tests = 0

        for benchmark in benchmarks:
            if self.config.should_exclude_benchmark(benchmark.name):
                if not self.config.quiet:
                    print(f"Skipping excluded benchmark: {benchmark.name}")
                continue

            print(f"Verifying: {benchmark.module_path}.{benchmark.name}")

            if benchmark.params or getattr(benchmark, "needs_runtime_eval", False):
                # Verify with all parameter combinations
                param_combinations = runner.get_param_combinations(benchmark)

                for params in param_combinations:
                    # Check if this was a failed capture
                    if storage.is_failed_capture(benchmark.name, benchmark.module_path, params):
                        print(f"  Skipping failed capture with params: {params}")
                        continue

                    # Run benchmark
                    result = runner.run_benchmark(benchmark, params)
                    if not result or not result.success:
                        print(f"  Failed to run with params: {params}")
                        failed_tests += 1
                        total_tests += 1
                        continue

                    # Load snapshot
                    snapshot_data = storage.load_snapshot(
                        benchmark_name=benchmark.name,
                        module_path=benchmark.module_path,
                        parameters=params,
                    )

                    if snapshot_data is None:
                        print(f"  No snapshot found for params: {params}")
                        failed_tests += 1
                        total_tests += 1
                        continue

                    expected_value, metadata = snapshot_data

                    # Compare
                    comparison = comparator.compare(result.return_value, expected_value)
                    total_tests += 1

                    if comparison.match:
                        passed_tests += 1
                        if not self.config.quiet:
                            print(f"  ✓ Passed with params: {params}")
                    else:
                        failed_tests += 1
                        print(f"  ✗ Failed with params: {params}")
                        print(f"    Error: {comparison.error_message}")
                        if self.config.verbose and comparison.details:
                            print(f"    Details: {comparison.details}")
            else:
                # Check if this was a failed capture
                if storage.is_failed_capture(benchmark.name, benchmark.module_path, ()):
                    print(f"  Skipping failed capture")
                    continue

                # Verify without parameters
                result = runner.run_benchmark(benchmark)
                if not result or not result.success:
                    print(f"  Failed to run")
                    failed_tests += 1
                    total_tests += 1
                    continue

                snapshot_data = storage.load_snapshot(
                    benchmark_name=benchmark.name, module_path=benchmark.module_path, parameters=()
                )

                if snapshot_data is None:
                    print(f"  No snapshot found")
                    failed_tests += 1
                    total_tests += 1
                    continue

                expected_value, metadata = snapshot_data

                comparison = comparator.compare(result.return_value, expected_value)
                total_tests += 1

                if comparison.match:
                    passed_tests += 1
                    if not self.config.quiet:
                        print(f"  ✓ Passed")
                else:
                    failed_tests += 1
                    print(f"  ✗ Failed")
                    print(f"    Error: {comparison.error_message}")
                    if self.config.verbose and comparison.details:
                        print(f"    Details: {comparison.details}")

        print(f"\nVerification complete:")
        print(f"  Total tests: {total_tests}")
        print(f"  Passed: {passed_tests}")
        print(f"  Failed: {failed_tests}")

        return 0 if failed_tests == 0 else 1

    def _list_command(self, args) -> int:
        """Handle the list command."""
        benchmark_dir = args.benchmark_dir

        discovery = BenchmarkDiscovery(benchmark_dir)
        benchmarks = discovery.discover_all()

        if args.filter:
            benchmarks = [b for b in benchmarks if args.filter in b.name]

        print(f"Found {len(benchmarks)} benchmarks in {benchmark_dir}:")

        for benchmark in benchmarks:
            print(f"  {benchmark.module_path}.{benchmark.name}")
            if benchmark.benchmark_type == "method":
                print(f"    Type: method in class {benchmark.class_name}")
                if benchmark.has_setup:
                    print(f"    Setup: {benchmark.setup_method}")
            else:
                print(f"    Type: function")

            if benchmark.params:
                param_combinations = discovery.generate_parameter_combinations(benchmark)
                print(f"    Parameters: {len(param_combinations)} combinations")
                if self.config.verbose:
                    for i, params in enumerate(param_combinations[:5]):  # Show first 5
                        print(f"      {i + 1}: {params}")
                    if len(param_combinations) > 5:
                        print(f"      ... and {len(param_combinations) - 5} more")

        return 0

    def _clean_command(self, args) -> int:
        """Handle the clean command."""
        snapshot_dir = args.snapshot_dir or self.config.get_snapshot_dir()

        if not snapshot_dir.exists():
            print(f"Snapshot directory {snapshot_dir} does not exist")
            return 0

        storage = SnapshotManager(snapshot_dir)
        stats = storage.get_snapshot_stats()

        print(f"Snapshot directory: {snapshot_dir}")
        print(f"Total snapshots: {stats['total_snapshots']}")
        print(f"Total size: {stats['total_size_bytes'] / 1024 / 1024:.2f} MB")

        if args.dry_run:
            print("Dry run - no files would be deleted")
            return 0

        # For now, just show stats. Could add more sophisticated cleanup logic
        print("Use --dry-run to see what would be cleaned")

        return 0

    def _config_command(self, args) -> int:
        """Handle the config command."""
        if args.init:
            self.config_manager.create_default_config()
            return 0

        if args.show:
            print("Current configuration:")
            config_dict = self.config.to_dict()
            for key, value in config_dict.items():
                print(f"  {key}: {value}")
            return 0

        print("Use --init to create default config or --show to display current config")
        return 0


def main():
    """Main entry point for the CLI."""
    cli = SnapshotCLI()
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())
