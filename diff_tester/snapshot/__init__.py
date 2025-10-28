"""
Snapshot testing tool for ASV benchmarks.

This module provides functionality to capture and compare function outputs
from ASV benchmarks to verify correctness after optimizations.
"""

__version__ = "0.1.0"

from .discovery import BenchmarkDiscovery, BenchmarkInfo
from .tracer import ExecutionTracer, TraceResult
from .storage import SnapshotManager, SnapshotMetadata
from .comparator import Comparator, ComparisonResult, ComparisonConfig
from .runner import BenchmarkRunner
from .config import SnapshotConfig, ConfigManager
from .cli import SnapshotCLI, main

__all__ = [
    "BenchmarkDiscovery",
    "BenchmarkInfo", 
    "ExecutionTracer",
    "TraceResult",
    "SnapshotManager",
    "SnapshotMetadata",
    "Comparator",
    "ComparisonResult",
    "ComparisonConfig",
    "BenchmarkRunner",
    "SnapshotConfig",
    "ConfigManager",
    "SnapshotCLI",
    "main",
]
