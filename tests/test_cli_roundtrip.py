"""
CLI roundtrip tests for snapshot capture and verification.

Tests the core guarantee: list -> capture -> verify always succeeds with only
passes or skips on real benchmark repositories (astropy, pandas, shapely).

This mimics the behavior of customtest.sh.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import pytest


@pytest.fixture
def test_repos_dir():
    """Get the test_repos directory."""
    return Path(__file__).parent / "test_repos"


@pytest.fixture
def snapshot_dir():
    """Create a temporary snapshot directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


def list_snapshot_files(snapshot_dir: Path) -> list[Path]:
    """List snapshot artefacts produced by a capture.

    Under the SQLite backend, the payloads all live in ``snapshots.db``; the
    per-test JSON metadata sidecars are 1:1 with snapshot rows, so we use them
    as the count.
    """
    return [p for p in snapshot_dir.rglob("*.json") if p.name != "baseline.json"]


def _get_cli_filter() -> Optional[str]:
    filter_pattern = os.getenv("SNAPSHOT_TOOL_FILTER")
    return filter_pattern if filter_pattern else None


def _get_cli_timeout() -> Optional[float]:
    timeout_value = os.getenv("SNAPSHOT_TOOL_TIMEOUT")
    if not timeout_value:
        return None
    try:
        return float(timeout_value)
    except ValueError:
        return None


def run_snapshot_roundtrip(benchmark_dir: Path, snapshot_dir: Path, timeout_minutes: int = 10):
    """
    Run a complete snapshot roundtrip: list -> capture -> verify.

    Args:
        benchmark_dir: Directory containing benchmarks
        snapshot_dir: Directory to store snapshots
        timeout_minutes: Timeout in minutes for capture and verify steps

    Returns:
        tuple: (list_result, capture_result, verify_result)
    """
    timeout_seconds = timeout_minutes * 60

    filter_pattern = _get_cli_filter()
    benchmark_timeout = _get_cli_timeout()

    list_args = ["snapshot-tool", "list", str(benchmark_dir)]
    if filter_pattern:
        list_args.extend(["--filter", filter_pattern])

    # Step 1: List benchmarks
    list_result = subprocess.run(list_args, capture_output=True, text=True, timeout=60)

    capture_args = [
        "snapshot-tool",
        "capture",
        str(benchmark_dir),
        "--snapshot-dir",
        str(snapshot_dir),
    ]
    if filter_pattern:
        capture_args.extend(["--filter", filter_pattern])
    if benchmark_timeout is not None:
        capture_args.extend(["--timeout", str(benchmark_timeout)])

    # Step 2: Capture snapshots
    capture_result = subprocess.run(
        capture_args, capture_output=True, text=True, timeout=timeout_seconds
    )

    verify_args = [
        "snapshot-tool",
        "verify",
        str(benchmark_dir),
        "--snapshot-dir",
        str(snapshot_dir),
    ]
    if filter_pattern:
        verify_args.extend(["--filter", filter_pattern])
    if benchmark_timeout is not None:
        verify_args.extend(["--timeout", str(benchmark_timeout)])

    # Step 3: Verify snapshots
    verify_result = subprocess.run(
        verify_args, capture_output=True, text=True, timeout=timeout_seconds
    )

    return list_result, capture_result, verify_result


def assert_roundtrip_succeeds(result, step_name: str, repo_name: str):
    """
    Assert that a roundtrip step completes successfully.

    The roundtrip guarantee:
    - List: Always succeeds (returncode 0)
    - Capture: Succeeds even if some benchmarks fail (returncode 0 or 1)
    - Verify: Must have 100% pass or skip rate (returncode 0, Failed: 0)

    Individual benchmarks may fail during capture (due to bugs, missing deps, etc.),
    but these are marked as "failed captures" and skipped during verify.
    The verify step must never have failures - only passes and skips.

    Args:
        result: subprocess result
        step_name: Name of the step (List, Capture, Verify)
        repo_name: Name of the repository being tested
    """
    # For list and capture, CLI should complete without crashing
    if step_name in ["List", "Capture"]:
        # Allow returncode 0 or 1 - returncode 1 means some benchmarks failed to run
        # but the process completed
        assert result.returncode in [0, 1], (
            f"{step_name} crashed for {repo_name}:\n"
            f"Return code: {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        return

    # For verify step - MUST be 100% pass or skip
    if step_name == "Verify":
        # Verify must succeed with no failures
        assert result.returncode == 0, (
            f"{step_name} failed for {repo_name}:\n"
            f"Return code: {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

        # Check that verification completed
        output = result.stdout + result.stderr
        assert "Verification complete" in output or "Summary written" in output, (
            f"{step_name} for {repo_name} did not complete:\n{output[:1000]}"
        )

        # Extract pass/fail/skip counts
        import re

        passed_match = re.search(r"Passed:\s*(\d+)", output)
        failed_match = re.search(r"Failed:\s*(\d+)", output)
        skipped_match = re.search(r"Skipped:\s*(\d+)", output)

        if passed_match and failed_match:
            passed = int(passed_match.group(1))
            failed = int(failed_match.group(1))
            skipped = int(skipped_match.group(1)) if skipped_match else 0
            total = passed + failed + skipped

            # MUST have 0 failures
            assert failed == 0, (
                f"{step_name} for {repo_name} had failures:\n"
                f"{passed} passed, {failed} failed, {skipped} skipped out of {total}\n"
                f"Expected: Failed = 0 (all benchmarks should pass or be skipped)\n"
                f"Output:\n{output}"
            )

            # Ensure at least some benchmarks ran
            assert total > 0, f"{step_name} for {repo_name} had no benchmarks:\n{output}"

            # Ensure at least one benchmark passed (not all skipped)
            assert passed > 0, (
                f"{step_name} for {repo_name} had no passing benchmarks (all skipped):\n"
                f"{passed} passed, {failed} failed, {skipped} skipped"
            )


class TestAstropyRoundtrip:
    """Test snapshot roundtrip for astropy benchmarks."""

    def test_astropy_full_roundtrip(self, test_repos_dir, snapshot_dir):
        """Test complete roundtrip: list -> capture -> verify for astropy."""
        astropy_dir = test_repos_dir / "astropy_benchmarks"
        if not astropy_dir.exists():
            pytest.skip("Astropy benchmarks not found")

        # Astropy has many benchmarks - use 30 minute timeout
        list_result, capture_result, verify_result = run_snapshot_roundtrip(
            astropy_dir, snapshot_dir, timeout_minutes=30
        )

        # List should always succeed
        assert list_result.returncode == 0, (
            f"List failed:\n{list_result.stdout}\n{list_result.stderr}"
        )

        # Capture should succeed (or skip/timeout some benchmarks)
        assert_roundtrip_succeeds(capture_result, "Capture", "astropy_benchmarks")

        # Verify should succeed with 100% passes or skips (no failures allowed)
        assert_roundtrip_succeeds(verify_result, "Verify", "astropy_benchmarks")

        # Verify that snapshots were created
        snapshots = list_snapshot_files(snapshot_dir)
        if len(snapshots) == 0:
            # All benchmarks were skipped - that's OK, but verify should reflect this
            assert (
                "skipped" in verify_result.stdout.lower()
                or "no snapshots" in verify_result.stdout.lower()
            )


class TestPandasRoundtrip:
    """Test snapshot roundtrip for pandas benchmarks."""

    def test_pandas_full_roundtrip(self, test_repos_dir, snapshot_dir):
        """Test complete roundtrip: list -> capture -> verify for pandas."""
        pandas_dir = test_repos_dir / "pandas_benchmarks"
        if not pandas_dir.exists():
            pytest.skip("Pandas benchmarks not found")

        # Pandas has many benchmarks - use 30 minute timeout
        list_result, capture_result, verify_result = run_snapshot_roundtrip(
            pandas_dir, snapshot_dir, timeout_minutes=30
        )

        # List should always succeed
        assert list_result.returncode == 0, (
            f"List failed:\n{list_result.stdout}\n{list_result.stderr}"
        )

        # Capture should succeed (or skip/timeout some benchmarks)
        assert_roundtrip_succeeds(capture_result, "Capture", "pandas_benchmarks")

        # Verify should succeed with 100% passes or skips (no failures allowed)
        assert_roundtrip_succeeds(verify_result, "Verify", "pandas_benchmarks")

        # Verify that snapshots were created
        snapshots = list_snapshot_files(snapshot_dir)
        if len(snapshots) == 0:
            # All benchmarks were skipped - that's OK
            assert (
                "skipped" in verify_result.stdout.lower()
                or "no snapshots" in verify_result.stdout.lower()
            )


class TestShapelyRoundtrip:
    """Test snapshot roundtrip for shapely benchmarks."""

    def test_shapely_full_roundtrip(self, test_repos_dir, snapshot_dir):
        """Test complete roundtrip: list -> capture -> verify for shapely."""
        shapely_dir = test_repos_dir / "shapely_benchmarks"
        if not shapely_dir.exists():
            pytest.skip("Shapely benchmarks not found")

        list_result, capture_result, verify_result = run_snapshot_roundtrip(
            shapely_dir, snapshot_dir
        )

        # List should always succeed
        assert list_result.returncode == 0, (
            f"List failed:\n{list_result.stdout}\n{list_result.stderr}"
        )

        # Capture should succeed (or skip/timeout some benchmarks)
        assert_roundtrip_succeeds(capture_result, "Capture", "shapely_benchmarks")

        # Verify should succeed with 100% passes or skips (no failures allowed)
        assert_roundtrip_succeeds(verify_result, "Verify", "shapely_benchmarks")

        # Shapely should create some snapshots (we know shapely works)
        snapshots = list_snapshot_files(snapshot_dir)
        assert len(snapshots) > 0, "Shapely should create at least one snapshot"

    def test_shapely_multiple_verify_passes(self, test_repos_dir, snapshot_dir):
        """Test that verify passes multiple times (determinism check)."""
        shapely_dir = test_repos_dir / "shapely_benchmarks"
        if not shapely_dir.exists():
            pytest.skip("Shapely benchmarks not found")

        # Capture once
        filter_pattern = _get_cli_filter()
        benchmark_timeout = _get_cli_timeout()

        capture_args = [
            "snapshot-tool",
            "capture",
            str(shapely_dir),
            "--snapshot-dir",
            str(snapshot_dir),
        ]
        if filter_pattern:
            capture_args.extend(["--filter", filter_pattern])
        if benchmark_timeout is not None:
            capture_args.extend(["--timeout", str(benchmark_timeout)])

        capture_result = subprocess.run(capture_args, capture_output=True, text=True, timeout=300)
        assert_roundtrip_succeeds(capture_result, "Capture", "shapely_benchmarks")

        # Verify three times - all should pass with no failures
        for round_num in range(3):
            verify_args = [
                "snapshot-tool",
                "verify",
                str(shapely_dir),
                "--snapshot-dir",
                str(snapshot_dir),
            ]
            if filter_pattern:
                verify_args.extend(["--filter", filter_pattern])
            if benchmark_timeout is not None:
                verify_args.extend(["--timeout", str(benchmark_timeout)])

            verify_result = subprocess.run(verify_args, capture_output=True, text=True, timeout=300)

            assert_roundtrip_succeeds(verify_result, "Verify", "shapely_benchmarks")


class TestAllReposRoundtrip:
    """Test snapshot roundtrip for all repositories together."""

    @pytest.mark.slow
    def test_all_repos_roundtrip(self, test_repos_dir, snapshot_dir):
        """
        Test roundtrip for all three repos sequentially.
        This is the equivalent of running customtest.sh.
        """
        repos = ["astropy_benchmarks", "pandas_benchmarks", "shapely_benchmarks"]

        results = {}

        for repo_name in repos:
            repo_dir = test_repos_dir / repo_name
            if not repo_dir.exists():
                continue

            # Create isolated snapshot directory for this repo
            repo_snapshot_dir = snapshot_dir / repo_name
            repo_snapshot_dir.mkdir(parents=True, exist_ok=True)

            list_result, capture_result, verify_result = run_snapshot_roundtrip(
                repo_dir, repo_snapshot_dir
            )

            results[repo_name] = {
                "list": list_result.returncode,
                "capture": capture_result.returncode,
                "verify": verify_result.returncode,
                "verify_output": verify_result.stdout + verify_result.stderr,
            }

        # All operations should succeed
        failed_repos = []
        for repo_name, result in results.items():
            if result["list"] != 0:
                failed_repos.append(f"{repo_name}: list failed")
            if result["capture"] not in [0, 1]:
                failed_repos.append(f"{repo_name}: capture crashed")
            if result["verify"] != 0:
                failed_repos.append(f"{repo_name}: verify failed")

            # Check for failures in verify output
            output_lower = result["verify_output"].lower()
            if "failed" in output_lower and "0 failed" not in output_lower:
                # Look for actual failure counts
                import re

                failure_match = re.search(r"(\d+)\s+failed", output_lower)
                if failure_match and int(failure_match.group(1)) > 0:
                    failed_repos.append(f"{repo_name}: verify had failures")

        assert len(failed_repos) == 0, "Some repositories failed roundtrip test:\n" + "\n".join(
            failed_repos
        )
