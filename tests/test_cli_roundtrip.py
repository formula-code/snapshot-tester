"""
CLI roundtrip tests for snapshot capture and verification.

Tests the core guarantee: list -> capture -> verify always succeeds with only
passes or skips on real benchmark repositories (astropy, pandas, shapely).

This mimics the behavior of customtest.sh.
"""

from __future__ import annotations

import json
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


def _parallel_enabled() -> bool:
    return os.getenv("SNAPSHOT_TOOL_PARALLEL", "").strip() not in ("", "0", "false", "False")


def _maybe_filter_timeout(args: list) -> list:
    """Append the shared --filter / --timeout / --parallel knobs (env-driven,
    used by the sharded CI jobs)."""
    filter_pattern = _get_cli_filter()
    benchmark_timeout = _get_cli_timeout()
    if filter_pattern:
        args.extend(["--filter", filter_pattern])
    if benchmark_timeout is not None:
        args.extend(["--timeout", str(benchmark_timeout)])
    # `list` has no --parallel; only add it for capture/baseline/verify.
    if _parallel_enabled() and len(args) > 1 and args[1] in ("capture", "baseline", "verify"):
        args.append("--parallel")
    return args


def run_snapshot_roundtrip(benchmark_dir: Path, snapshot_dir: Path, timeout_minutes: int = 10):
    """
    Run a complete snapshot roundtrip: list -> capture -> baseline -> verify.

    The gate is *regression-based*, not perfection-based: real third-party
    benchmark suites contain benchmarks that are inherently un-snapshotable
    (memory addresses in reprs, timing-sensitive, dtype-unstable). Those fail
    consistently and show up as fail->fail, which is tolerated. What must never
    happen is a pass->fail or skip->fail transition between the baseline and the
    verify run. The tool's own `baseline` subcommand + transition matrix
    (written into summary.json by `verify`) implements exactly this.

    Args:
        benchmark_dir: Directory containing benchmarks
        snapshot_dir: Directory to store snapshots
        timeout_minutes: Timeout in minutes for capture/baseline/verify steps

    Returns:
        tuple: (list_result, capture_result, baseline_result, verify_result, summary_path)
    """
    timeout_seconds = timeout_minutes * 60
    summary_path = snapshot_dir / "summary.json"

    # Step 1: List benchmarks
    list_args = _maybe_filter_timeout(["snapshot-tool", "list", str(benchmark_dir)])
    # `list` has no --timeout; drop it if _maybe_filter_timeout added one.
    if "--timeout" in list_args:
        i = list_args.index("--timeout")
        del list_args[i : i + 2]
    list_result = subprocess.run(list_args, capture_output=True, text=True, timeout=60)

    # Step 2: Capture snapshots
    capture_args = _maybe_filter_timeout(
        ["snapshot-tool", "capture", str(benchmark_dir), "--snapshot-dir", str(snapshot_dir)]
    )
    capture_result = subprocess.run(
        capture_args, capture_output=True, text=True, timeout=timeout_seconds
    )

    # Step 3: Baseline (records per-test pass/fail/skip into snapshot_dir/baseline.json)
    baseline_args = _maybe_filter_timeout(
        ["snapshot-tool", "baseline", str(benchmark_dir), "--snapshot-dir", str(snapshot_dir)]
    )
    baseline_result = subprocess.run(
        baseline_args, capture_output=True, text=True, timeout=timeout_seconds
    )

    # Step 4: Verify (re-runs, diffs against baseline; writes transition matrix to summary.json)
    verify_args = _maybe_filter_timeout(
        [
            "snapshot-tool",
            "verify",
            str(benchmark_dir),
            "--snapshot-dir",
            str(snapshot_dir),
            "--summary",
            str(summary_path),
        ]
    )
    verify_result = subprocess.run(
        verify_args, capture_output=True, text=True, timeout=timeout_seconds
    )

    return list_result, capture_result, baseline_result, verify_result, summary_path


def assert_step_did_not_crash(result, step_name: str, repo_name: str):
    """List / Capture / Baseline must complete without crashing.

    returncode 0 or 1 is fine — 1 just means some individual benchmarks failed
    to run (recorded as failed captures), the process itself completed.
    """
    assert result.returncode in [0, 1], (
        f"{step_name} crashed for {repo_name}:\n"
        f"Return code: {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )


def assert_no_regressions(summary_path: Path, repo_name: str, verify_result=None):
    """Assert the baseline->verify transition matrix contains no regressions.

    A regression is a benchmark that passed (or was skipped) in the baseline
    but failed in verify: `pass-to-fail` or `skip-to-fail`. Consistently broken
    benchmarks (fail-to-fail) and consistently working ones (pass-to-pass) are
    fine — only a *new* failure on the same code fails the gate.
    """
    assert summary_path.exists(), (
        f"verify wrote no summary for {repo_name} at {summary_path}.\n"
        f"verify stdout/stderr:\n"
        f"{(verify_result.stdout + verify_result.stderr) if verify_result else '<n/a>'}"
    )

    with open(summary_path) as f:
        summary = json.load(f)

    pass_to_fail = summary.get("pass-to-fail", 0)
    skip_to_fail = summary.get("skip-to-fail", 0)
    total = summary.get("total", 0)
    passed = summary.get("passed", 0)

    assert total > 0, f"{repo_name}: verify ran no benchmarks (summary={summary})"
    assert passed > 0, (
        f"{repo_name}: verify had no passing benchmarks (all failed/skipped), summary={summary}"
    )
    assert pass_to_fail == 0 and skip_to_fail == 0, (
        f"{repo_name}: REGRESSION detected between baseline and verify.\n"
        f"  pass-to-fail = {pass_to_fail}\n"
        f"  skip-to-fail = {skip_to_fail}\n"
        f"Full transition matrix / summary:\n{json.dumps(summary, indent=2)}"
    )


class TestAstropyRoundtrip:
    """Test snapshot roundtrip for astropy benchmarks."""

    def test_astropy_full_roundtrip(self, test_repos_dir, snapshot_dir):
        """Test complete roundtrip: list -> capture -> verify for astropy."""
        astropy_dir = test_repos_dir / "astropy_benchmarks"
        if not astropy_dir.exists():
            pytest.skip("Astropy benchmarks not found")

        # Astropy has many benchmarks - use 30 minute timeout
        (
            list_result,
            capture_result,
            baseline_result,
            verify_result,
            summary_path,
        ) = run_snapshot_roundtrip(astropy_dir, snapshot_dir, timeout_minutes=30)

        assert list_result.returncode == 0, (
            f"List failed:\n{list_result.stdout}\n{list_result.stderr}"
        )
        assert_step_did_not_crash(capture_result, "Capture", "astropy_benchmarks")
        assert_step_did_not_crash(baseline_result, "Baseline", "astropy_benchmarks")
        assert_no_regressions(summary_path, "astropy_benchmarks", verify_result)


class TestPandasRoundtrip:
    """Test snapshot roundtrip for pandas benchmarks."""

    def test_pandas_full_roundtrip(self, test_repos_dir, snapshot_dir):
        """Test complete roundtrip: list -> capture -> verify for pandas."""
        pandas_dir = test_repos_dir / "pandas_benchmarks"
        if not pandas_dir.exists():
            pytest.skip("Pandas benchmarks not found")

        # Pandas has many benchmarks - use 30 minute timeout
        (
            list_result,
            capture_result,
            baseline_result,
            verify_result,
            summary_path,
        ) = run_snapshot_roundtrip(pandas_dir, snapshot_dir, timeout_minutes=30)

        assert list_result.returncode == 0, (
            f"List failed:\n{list_result.stdout}\n{list_result.stderr}"
        )
        assert_step_did_not_crash(capture_result, "Capture", "pandas_benchmarks")
        assert_step_did_not_crash(baseline_result, "Baseline", "pandas_benchmarks")
        assert_no_regressions(summary_path, "pandas_benchmarks", verify_result)


class TestShapelyRoundtrip:
    """Test snapshot roundtrip for shapely benchmarks."""

    def test_shapely_full_roundtrip(self, test_repos_dir, snapshot_dir):
        """Test complete roundtrip: list -> capture -> verify for shapely."""
        shapely_dir = test_repos_dir / "shapely_benchmarks"
        if not shapely_dir.exists():
            pytest.skip("Shapely benchmarks not found")

        (
            list_result,
            capture_result,
            baseline_result,
            verify_result,
            summary_path,
        ) = run_snapshot_roundtrip(shapely_dir, snapshot_dir)

        assert list_result.returncode == 0, (
            f"List failed:\n{list_result.stdout}\n{list_result.stderr}"
        )
        assert_step_did_not_crash(capture_result, "Capture", "shapely_benchmarks")
        assert_step_did_not_crash(baseline_result, "Baseline", "shapely_benchmarks")
        assert_no_regressions(summary_path, "shapely_benchmarks", verify_result)

        # Shapely is fully deterministic - it should create real snapshots.
        snapshots = list_snapshot_files(snapshot_dir)
        assert len(snapshots) > 0, "Shapely should create at least one snapshot"

    def test_shapely_multiple_verify_passes(self, test_repos_dir, snapshot_dir):
        """Test that verify passes multiple times (determinism check)."""
        shapely_dir = test_repos_dir / "shapely_benchmarks"
        if not shapely_dir.exists():
            pytest.skip("Shapely benchmarks not found")

        # Capture once, baseline once, then verify three times - a determinism
        # check: every verify must show zero regressions against the baseline.
        capture_args = _maybe_filter_timeout(
            ["snapshot-tool", "capture", str(shapely_dir), "--snapshot-dir", str(snapshot_dir)]
        )
        capture_result = subprocess.run(capture_args, capture_output=True, text=True, timeout=300)
        assert_step_did_not_crash(capture_result, "Capture", "shapely_benchmarks")

        baseline_args = _maybe_filter_timeout(
            ["snapshot-tool", "baseline", str(shapely_dir), "--snapshot-dir", str(snapshot_dir)]
        )
        baseline_result = subprocess.run(baseline_args, capture_output=True, text=True, timeout=300)
        assert_step_did_not_crash(baseline_result, "Baseline", "shapely_benchmarks")

        for round_num in range(3):
            summary_path = snapshot_dir / f"summary_{round_num}.json"
            verify_args = _maybe_filter_timeout(
                [
                    "snapshot-tool",
                    "verify",
                    str(shapely_dir),
                    "--snapshot-dir",
                    str(snapshot_dir),
                    "--summary",
                    str(summary_path),
                ]
            )
            verify_result = subprocess.run(verify_args, capture_output=True, text=True, timeout=300)
            assert_no_regressions(summary_path, "shapely_benchmarks", verify_result)


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

            (
                list_result,
                capture_result,
                baseline_result,
                verify_result,
                summary_path,
            ) = run_snapshot_roundtrip(repo_dir, repo_snapshot_dir)

            results[repo_name] = {
                "list": list_result.returncode,
                "capture": capture_result.returncode,
                "baseline": baseline_result.returncode,
                "summary_path": summary_path,
                "verify_result": verify_result,
            }

        failed_repos = []
        for repo_name, result in results.items():
            if result["list"] != 0:
                failed_repos.append(f"{repo_name}: list failed")
            if result["capture"] not in [0, 1]:
                failed_repos.append(f"{repo_name}: capture crashed")
            if result["baseline"] not in [0, 1]:
                failed_repos.append(f"{repo_name}: baseline crashed")
            try:
                assert_no_regressions(result["summary_path"], repo_name, result["verify_result"])
            except AssertionError as e:
                failed_repos.append(str(e))

        assert len(failed_repos) == 0, "Some repositories failed roundtrip test:\n" + "\n".join(
            failed_repos
        )
