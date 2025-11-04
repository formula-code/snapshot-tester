"""
Pytest configuration and shared fixtures for snapshot_tool tests.
"""

import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def temp_snapshot_dir(temp_dir):
    """Create a temporary snapshot directory."""
    snapshot_path = temp_dir / ".snapshots"
    snapshot_path.mkdir(parents=True, exist_ok=True)
    return snapshot_path


@pytest.fixture
def temp_benchmark_dir(temp_dir):
    """Create a temporary benchmark directory."""
    benchmark_path = temp_dir / "benchmarks"
    benchmark_path.mkdir(parents=True, exist_ok=True)
    return benchmark_path
