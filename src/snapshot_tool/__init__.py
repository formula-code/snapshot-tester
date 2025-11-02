"""
Snapshot testing tool for ASV benchmarks.

This package provides tools to capture and compare function return values
from benchmarks to verify correctness after optimizations.
"""

import logging
import sys

__version__ = "0.1.0"

# Configure logging for the package
def configure_logging(level=logging.INFO):
    """Configure logging for the snapshot_tool package."""
    # Configure the package-level logger
    logger = logging.getLogger('snapshot_tool')

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)

    return logger

# Configure logging by default
configure_logging()

# Import main classes for public API
from .cli import SnapshotCLI, main
from .comparator import Comparator, ComparisonConfig, ComparisonResult
from .config import ConfigManager, SnapshotConfig
from .discovery import BenchmarkDiscovery, BenchmarkInfo
from .rng_patcher import RNGPatcher, patch_all_rngs, unpatch_all_rngs, reset_all_rngs
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
