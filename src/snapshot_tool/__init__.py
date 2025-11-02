"""
Snapshot testing tool for ASV benchmarks.

This package provides tools to capture and compare function return values
from benchmarks to verify correctness after optimizations.
"""

__version__ = "0.1.0"

# Import main classes for public API
from .cli import SnapshotCLI, main
from .comparator import Comparator, ComparisonConfig, ComparisonResult
from .config import ConfigManager, SnapshotConfig
from .discovery import BenchmarkDiscovery, BenchmarkInfo
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
    # CLI
    "SnapshotCLI",
    "main",
]
