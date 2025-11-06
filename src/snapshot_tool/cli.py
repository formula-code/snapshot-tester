"""
Command-line interface for snapshot testing.

This module provides CLI commands for capturing and verifying snapshots
of ASV benchmark outputs.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from .comparator import Comparator, ComparisonConfig
from .config import ConfigManager
from .discovery import BenchmarkDiscovery
from .runner import BenchmarkRunner
from .storage import SnapshotManager

logger = logging.getLogger(__name__)


class SnapshotCLI:
    """Command-line interface for snapshot testing."""

    def __init__(self):
        self.config_manager = ConfigManager()
        self.config = self.config_manager.get_config()

    def run(self, args: Optional[list[str]] = None) -> int:
        """Run the CLI with given arguments."""
        parser = self._create_parser()
        parsed_args = parser.parse_args(args)

        try:
            return parsed_args.func(parsed_args)
        except KeyboardInterrupt:
            logger.info("\nInterrupted by user")
            return 1
        except Exception as e:
            logger.error(f"Error: {e}")
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
        verify_parser.add_argument(
            "--summary",
            type=Path,
            default=Path("summary.json"),
            help="Path to write summary JSON file (default: summary.json)",
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

        logger.info(f"Capturing snapshots from {benchmark_dir}")
        logger.info(f"Storing snapshots in {snapshot_dir}")

        # Initialize components
        runner = BenchmarkRunner(benchmark_dir)
        storage = SnapshotManager(snapshot_dir)

        # Discover benchmarks
        discovery = BenchmarkDiscovery(benchmark_dir)
        benchmarks = discovery.discover_all()

        if args.filter:
            benchmarks = [b for b in benchmarks if re.search(args.filter, f"{b.module_path}.{b.name}")]

        captured_count = 0

        for benchmark in benchmarks:
            if self.config.should_exclude_benchmark(benchmark.name):
                if not self.config.quiet:
                    logger.info(f"Skipping excluded benchmark: {benchmark.name}")
                continue

            logger.info(f"Capturing: {benchmark.module_path}.{benchmark.name}")

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
                            class_name=benchmark.class_name,
                        )
                        captured_count += 1
                        if not self.config.quiet:
                            logger.info(f"  Captured with params: {params}")
                    else:
                        # Store failed capture marker
                        if result and result.error:
                            error_type = type(result.error).__name__
                            error_msg = str(result.error)
                            failure_reason = f"{error_type}: {error_msg}" if error_msg else error_type
                        else:
                            failure_reason = "Unknown error (no exception details)"

                        storage.store_failed_capture(
                            benchmark_name=benchmark.name,
                            module_path=benchmark.module_path,
                            parameters=params,
                            param_names=benchmark.param_names,
                            failure_reason=failure_reason,
                            class_name=benchmark.class_name,
                        )
                        logger.warning(f"  Failed to capture with params: {params} - {failure_reason}")
                        if self.config.verbose and result and result.error:
                            import traceback

                            logger.debug("    Full error traceback:")
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
                        class_name=benchmark.class_name,
                    )
                    captured_count += 1
                else:
                    # Store failed capture marker
                    if result and result.error:
                        error_type = type(result.error).__name__
                        error_msg = str(result.error)
                        failure_reason = f"{error_type}: {error_msg}" if error_msg else error_type
                    else:
                        failure_reason = "Unknown error (no exception details)"

                    storage.store_failed_capture(
                        benchmark_name=benchmark.name,
                        module_path=benchmark.module_path,
                        parameters=(),
                        param_names=None,
                        failure_reason=failure_reason,
                        class_name=benchmark.class_name,
                    )
                    logger.warning(f"  Failed to capture - {failure_reason}")
                    if self.config.verbose and result and result.error:
                        import traceback

                        logger.debug("    Full error traceback:")
                        traceback.print_exc()

        logger.info(f"Captured {captured_count} snapshots")
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

        logger.info(f"Verifying benchmarks in {benchmark_dir}")
        logger.info(f"Comparing against snapshots in {snapshot_dir}")

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
            benchmarks = [b for b in benchmarks if re.search(args.filter, f"{b.module_path}.{b.name}")]

        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        skipped_tests = 0

        for benchmark in benchmarks:
            if self.config.should_exclude_benchmark(benchmark.name):
                if not self.config.quiet:
                    logger.info(f"Skipping excluded benchmark: {benchmark.name}")
                continue

            logger.info(f"Verifying: {benchmark.module_path}.{benchmark.name}")

            if benchmark.params or getattr(benchmark, "needs_runtime_eval", False):
                # Verify with all parameter combinations
                param_combinations = runner.get_param_combinations(benchmark)

                for params in param_combinations:
                    # Load snapshot first to check if it exists
                    snapshot_data = storage.load_snapshot(
                        benchmark_name=benchmark.name,
                        module_path=benchmark.module_path,
                        parameters=params,
                        class_name=benchmark.class_name,
                    )

                    # If no snapshot, skip (wasn't captured successfully)
                    if snapshot_data is None:
                        logger.info(f"  Skipping (no snapshot) with params: {params}")
                        skipped_tests += 1
                        total_tests += 1
                        continue

                    _, metadata = snapshot_data

                    # If this was a failed capture, skip
                    if metadata.capture_failed:
                        logger.info(f"  Skipping failed capture with params: {params}")
                        skipped_tests += 1
                        total_tests += 1
                        continue

                    # Run benchmark
                    result = runner.run_benchmark(benchmark, params)
                    if not result or not result.success:
                        # Benchmark failed during verify but succeeded during capture
                        # This is a real failure (non-deterministic benchmark or environment change)
                        logger.error(f"  ✗ Failed to run with params: {params} (succeeded during capture)")
                        failed_tests += 1
                        total_tests += 1
                        continue

                    expected_value, metadata = snapshot_data

                    # Compare
                    comparison = comparator.compare(result.return_value, expected_value)
                    total_tests += 1

                    if comparison.skipped:
                        skipped_tests += 1
                        if not self.config.quiet:
                            logger.info(f"  ⊘ Skipped with params: {params}")
                            if self.config.verbose and comparison.details:
                                logger.debug(f"    Reason: {comparison.details}")
                    elif comparison.match:
                        passed_tests += 1
                        if not self.config.quiet:
                            logger.info(f"  ✓ Passed with params: {params}")
                    else:
                        failed_tests += 1
                        logger.error(f"  ✗ Failed with params: {params}")
                        logger.error(f"    Error: {comparison.error_message}")
                        if self.config.verbose and comparison.details:
                            logger.debug(f"    Details: {comparison.details}")
            else:
                # Check if snapshot exists
                snapshot_data = storage.load_snapshot(
                    benchmark_name=benchmark.name,
                    module_path=benchmark.module_path,
                    parameters=(),
                    class_name=benchmark.class_name,
                )

                # If no snapshot, skip (wasn't captured successfully)
                if snapshot_data is None:
                    logger.info("  Skipping (no snapshot)")
                    skipped_tests += 1
                    total_tests += 1
                    continue

                _, metadata = snapshot_data

                # If this was a failed capture, skip
                if metadata.capture_failed:
                    logger.info("  Skipping failed capture")
                    skipped_tests += 1
                    total_tests += 1
                    continue

                # Verify without parameters
                result = runner.run_benchmark(benchmark)
                if not result or not result.success:
                    # Benchmark failed during verify but succeeded during capture
                    # This is a real failure (non-deterministic benchmark or environment change)
                    logger.error("  ✗ Failed to run (succeeded during capture)")
                    failed_tests += 1
                    total_tests += 1
                    continue

                expected_value, metadata = snapshot_data

                comparison = comparator.compare(result.return_value, expected_value)
                total_tests += 1

                if comparison.skipped:
                    skipped_tests += 1
                    if not self.config.quiet:
                        logger.info("  ⊘ Skipped")
                        if self.config.verbose and comparison.details:
                            logger.debug(f"    Reason: {comparison.details}")
                elif comparison.match:
                    passed_tests += 1
                    if not self.config.quiet:
                        logger.info("  ✓ Passed")
                else:
                    failed_tests += 1
                    logger.error("  ✗ Failed")
                    logger.error(f"    Error: {comparison.error_message}")
                    if self.config.verbose and comparison.details:
                        logger.debug(f"    Details: {comparison.details}")

        logger.info("\nVerification complete:")
        logger.info(f"  Total tests: {total_tests}")
        logger.info(f"  Passed: {passed_tests}")
        logger.info(f"  Failed: {failed_tests}")
        logger.info(f"  Skipped: {skipped_tests}")

        # Write summary.json
        summary = {
            "total": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "skipped": skipped_tests,
            "timestamp": datetime.now().isoformat(),
            "snapshot_dir": str(snapshot_dir),
            "benchmark_dir": str(benchmark_dir),
        }

        summary_path = args.summary if hasattr(args, 'summary') else Path("summary.json")
        try:
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            logger.info(f"\nSummary written to {summary_path}")
        except Exception as e:
            logger.warning(f"Failed to write summary.json: {e}")

        return 0 if failed_tests == 0 else 1

    def _list_command(self, args) -> int:
        """Handle the list command."""
        benchmark_dir = args.benchmark_dir

        discovery = BenchmarkDiscovery(benchmark_dir)
        benchmarks = discovery.discover_all()

        if args.filter:
            benchmarks = [b for b in benchmarks if re.search(args.filter, f"{b.module_path}.{b.name}")]

        logger.info(f"Found {len(benchmarks)} benchmarks in {benchmark_dir}:")

        for benchmark in benchmarks:
            logger.info(f"  {benchmark.module_path}.{benchmark.name}")
            if benchmark.benchmark_type == "method":
                logger.info(f"    Type: method in class {benchmark.class_name}")
                if benchmark.has_setup:
                    logger.info(f"    Setup: {benchmark.setup_method}")
            else:
                logger.info("    Type: function")

            if benchmark.params:
                param_combinations = discovery.generate_parameter_combinations(benchmark)
                logger.info(f"    Parameters: {len(param_combinations)} combinations")
                if self.config.verbose:
                    for i, params in enumerate(param_combinations[:5]):  # Show first 5
                        logger.debug(f"      {i + 1}: {params}")
                    if len(param_combinations) > 5:
                        logger.debug(f"      ... and {len(param_combinations) - 5} more")

        return 0

    def _clean_command(self, args) -> int:
        """Handle the clean command."""
        snapshot_dir = args.snapshot_dir or self.config.get_snapshot_dir()

        if not snapshot_dir.exists():
            logger.info(f"Snapshot directory {snapshot_dir} does not exist")
            return 0

        storage = SnapshotManager(snapshot_dir)
        stats = storage.get_snapshot_stats()

        logger.info(f"Snapshot directory: {snapshot_dir}")
        logger.info(f"Total snapshots: {stats['total_snapshots']}")
        logger.info(f"Total size: {stats['total_size_bytes'] / 1024 / 1024:.2f} MB")

        if args.dry_run:
            logger.info("Dry run - no files would be deleted")
            return 0

        # For now, just show stats. Could add more sophisticated cleanup logic
        logger.info("Use --dry-run to see what would be cleaned")

        return 0

    def _config_command(self, args) -> int:
        """Handle the config command."""
        if args.init:
            self.config_manager.create_default_config()
            return 0

        if args.show:
            logger.info("Current configuration:")
            config_dict = self.config.to_dict()
            for key, value in config_dict.items():
                logger.info(f"  {key}: {value}")
            return 0

        logger.info("Use --init to create default config or --show to display current config")
        return 0


def main():
    """Main entry point for the CLI."""
    cli = SnapshotCLI()
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())
