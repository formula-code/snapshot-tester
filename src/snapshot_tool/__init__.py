"""Capture and compare ASV benchmark return values to check correctness after optimizations."""

from __future__ import annotations

import logging
import sys

__version__ = "0.2.1"


def configure_logging(level=logging.INFO):
    """Add a stdout handler to the snapshot_tool logger if it has none."""
    logger = logging.getLogger("snapshot_tool")

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter("%(levelname)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)

    return logger


configure_logging()

from .cli import SnapshotCLI, main
from .comparator import Comparator, ComparisonConfig, ComparisonResult
from .config import ConfigManager, SnapshotConfig
from .discovery import BenchmarkDiscovery, BenchmarkInfo
from .rng_patcher import RNGPatcher, patch_all_rngs, reset_all_rngs, unpatch_all_rngs
from .runner import BenchmarkRunner
from .storage import SnapshotManager, SnapshotMetadata
from .tracer import ExecutionTracer, TraceResult

__all__ = [
    # Version
    "__version__",
    # Discovery
    "BenchmarkDiscovery",
    "BenchmarkInfo",
    # Runner
    "BenchmarkRunner",
    # Storage
    "SnapshotManager",
    "SnapshotMetadata",
    # Comparator
    "Comparator",
    "ComparisonConfig",
    "ComparisonResult",
    # Tracer
    "ExecutionTracer",
    "TraceResult",
    # Config
    "ConfigManager",
    "SnapshotConfig",
    # RNG Patcher
    "RNGPatcher",
    "patch_all_rngs",
    "unpatch_all_rngs",
    "reset_all_rngs",
    # CLI
    "SnapshotCLI",
    "main",
]
