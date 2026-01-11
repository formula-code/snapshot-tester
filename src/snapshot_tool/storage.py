"""
Snapshot storage and management system.

This module handles storing and retrieving snapshots using pickle files
with an organized directory structure.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_COMPRESS_THRESHOLD_BYTES = 5 * 1024 * 1024


@dataclass
class SnapshotMetadata:
    """Metadata for a snapshot."""

    benchmark_name: str
    module_path: str
    parameters: tuple[Any, ...]
    param_names: Optional[list[str]]
    timestamp: datetime
    class_name: Optional[str] = None  # Added to disambiguate benchmarks with same name
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    python_version: Optional[str] = None
    platform: Optional[str] = None
    capture_failed: bool = False
    failure_reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert datetime to string
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SnapshotMetadata":
        """Create from dictionary."""
        # Convert timestamp string back to datetime
        if isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class SnapshotManager:
    """Manages snapshot storage and retrieval."""

    def __init__(
        self, snapshot_dir: Path, *, compress_threshold_bytes: int = DEFAULT_COMPRESS_THRESHOLD_BYTES
    ):
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.compress_threshold_bytes = compress_threshold_bytes

    # -----------------
    # Baseline utilities
    # -----------------
    def baseline_path(self) -> Path:
        """Path to the baseline status file inside the snapshot directory."""
        return self.snapshot_dir / "baseline.json"

    def get_test_id(
        self,
        *,
        module_path: str,
        benchmark_name: str,
        parameters: tuple[Any, ...],
        class_name: Optional[str] = None,
    ) -> str:
        """Return a stable identifier for a benchmark + parameters.

        The identifier mirrors the on-disk snapshot layout to ensure stability
        across baseline and verify runs, including parameter hashing.
        """
        param_hash = self._generate_param_hash(parameters)
        benchmark_dir = f"{class_name}.{benchmark_name}" if class_name else benchmark_name
        # Avoid accidental path traversal by normalizing components explicitly
        return f"{module_path}/{benchmark_dir}/{param_hash}"

    def write_baseline(self, entries: dict[str, str], meta: Optional[dict[str, Any]] = None) -> Path:
        """Persist baseline pass/fail statuses under the snapshot directory.

        Args:
            entries: Mapping of `test_id` -> one of "pass" | "fail" | "skip".
            meta: Optional metadata to include (e.g., counts, dirs).
        Returns:
            Path to the written file.
        """
        payload: dict[str, Any] = {
            "schema": "snapshot_tool/baseline@2",
            "timestamp": datetime.now().isoformat(),
            "entries": entries,
        }
        if meta:
            payload["meta"] = meta

        path = self.baseline_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        return path

    def read_baseline(self) -> Optional[dict[str, Any]]:
        """Load baseline statuses if present, else None."""
        path = self.baseline_path()
        if not path.exists():
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read baseline file {path}: {e}")
            return None

    # -----------------
    # Baseline utilities
    # -----------------
    def baseline_path(self) -> Path:
        """Path to the baseline status file inside the snapshot directory."""
        return self.snapshot_dir / "baseline.json"

    def get_test_id(
        self,
        *,
        module_path: str,
        benchmark_name: str,
        parameters: tuple[Any, ...],
        class_name: Optional[str] = None,
    ) -> str:
        """Return a stable identifier for a benchmark + parameters.

        The identifier mirrors the on-disk snapshot layout to ensure stability
        across baseline and verify runs, including parameter hashing.
        """
        param_hash = self._generate_param_hash(parameters)
        benchmark_dir = f"{class_name}.{benchmark_name}" if class_name else benchmark_name
        # Avoid accidental path traversal by normalizing components explicitly
        return f"{module_path}/{benchmark_dir}/{param_hash}"

    def write_baseline(self, entries: dict[str, str], meta: Optional[dict[str, Any]] = None) -> Path:
        """Persist baseline pass/fail statuses under the snapshot directory.

        Args:
            entries: Mapping of `test_id` -> one of "pass" | "fail" | "skip".
            meta: Optional metadata to include (e.g., counts, dirs).
        Returns:
            Path to the written file.
        """
        payload: dict[str, Any] = {
            "schema": "snapshot_tool/baseline@2",
            "timestamp": datetime.now().isoformat(),
            "entries": entries,
        }
        if meta:
            payload["meta"] = meta

        path = self.baseline_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        return path

    def read_baseline(self) -> Optional[dict[str, Any]]:
        """Load baseline statuses if present, else None."""
        path = self.baseline_path()
        if not path.exists():
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read baseline file {path}: {e}")
            return None

    def store_snapshot(
        self,
        benchmark_name: str,
        module_path: str,
        parameters: tuple[Any, ...],
        param_names: Optional[list[str]],
        return_value: Any,
        class_name: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Path:
        """Store a snapshot with its metadata."""

        # Generate parameter hash for unique identification
        param_hash = self._generate_param_hash(parameters)

        # Create directory structure: .snapshots/<module>/<class>.<benchmark>/ or .snapshots/<module>/<benchmark>/
        if class_name:
            benchmark_dir = f"{class_name}.{benchmark_name}"
        else:
            benchmark_dir = benchmark_name
        base_path = self.snapshot_dir / module_path / benchmark_dir / param_hash
        base_path.parent.mkdir(parents=True, exist_ok=True)

        # Create metadata
        snapshot_metadata = SnapshotMetadata(
            benchmark_name=benchmark_name,
            module_path=module_path,
            parameters=parameters,
            param_names=param_names,
            class_name=class_name,
            timestamp=datetime.now(),
            git_commit=self._get_git_commit(),
            git_branch=self._get_git_branch(),
            python_version=self._get_python_version(),
            platform=self._get_platform(),
            **(metadata or {}),
        )

        # Store the snapshot data
        serialized_value = self._serialize_value(return_value)
        snapshot_data = {"return_value": serialized_value, "metadata": snapshot_metadata}

        # Attempt to store snapshot; if pickling still fails due to nested
        # unpicklables, fall back to a placeholder structure.
        try:
            snapshot_path = self._write_snapshot_data(snapshot_data, base_path)
        except Exception as e:
            fallback_data = {
                "return_value": {
                    "__unpicklable__": True,
                    "__error__": f"Pickle failed: {e}",
                },
                "metadata": snapshot_metadata,
            }
            snapshot_path = self._write_snapshot_data(fallback_data, base_path)

        # Store metadata separately as JSON for easy inspection
        metadata_path = base_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(snapshot_metadata.to_dict(), f, indent=2, default=str)

        return snapshot_path

    def store_failed_capture(
        self,
        benchmark_name: str,
        module_path: str,
        parameters: tuple[Any, ...],
        param_names: Optional[list[str]],
        failure_reason: str,
        class_name: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Path:
        """Store a failed capture marker."""

        # Generate parameter hash for unique identification
        param_hash = self._generate_param_hash(parameters)

        # Create directory structure: .snapshots/<module>/<class>.<benchmark>/ or .snapshots/<module>/<benchmark>/
        if class_name:
            benchmark_dir = f"{class_name}.{benchmark_name}"
        else:
            benchmark_dir = benchmark_name
        base_path = self.snapshot_dir / module_path / benchmark_dir / param_hash
        base_path.parent.mkdir(parents=True, exist_ok=True)

        # Create metadata for failed capture
        snapshot_metadata = SnapshotMetadata(
            benchmark_name=benchmark_name,
            module_path=module_path,
            parameters=parameters,
            param_names=param_names,
            class_name=class_name,
            timestamp=datetime.now(),
            git_commit=self._get_git_commit(),
            git_branch=self._get_git_branch(),
            python_version=self._get_python_version(),
            platform=self._get_platform(),
            capture_failed=True,
            failure_reason=failure_reason,
            **(metadata or {}),
        )

        # Store the failed capture marker
        snapshot_data = {
            "return_value": None,  # No return value for failed captures
            "metadata": snapshot_metadata,
        }

        snapshot_path = self._write_snapshot_data(snapshot_data, base_path)

        # Store metadata separately as JSON for easy inspection
        metadata_path = base_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(snapshot_metadata.to_dict(), f, indent=2, default=str)

        return snapshot_path

    def _serialize_dict_safely(self, d: dict) -> dict:
        """Safely serialize a dictionary, handling generators and other non-serializable values."""
        result = {}
        for key, value in d.items():
            try:
                # Try to serialize the key
                serialized_key = self._serialize_value(key)
                # Try to serialize the value
                serialized_value = self._serialize_value(value)
                result[serialized_key] = serialized_value
            except Exception as e:
                # If we can't serialize this key-value pair, store an error message
                result[f"__error_{key}__"] = f"Cannot serialize: {e}"
        return result

    def _serialize_value(self, value: Any) -> Any:
        """Safely serialize a value, handling class instances and generators."""
        try:
            # Try to pickle AND unpickle the value to ensure it's truly serializable
            # This catches objects that pickle fine but fail to unpickle (e.g., version mismatches)
            pickled = pickle.dumps(value)
            pickle.loads(pickled)  # Test round-trip
            return value
        except Exception as e:
            # Catch all exceptions during pickling/unpickling
            # Common issues: PicklingError, TypeError, AttributeError, ImportError, etc.
            # Check if it's a generator
            if hasattr(value, "__iter__") and hasattr(value, "__next__"):
                # It's a generator - cannot be pickled
                return {
                    "__generator__": True,
                    "__generator_type__": type(value).__name__,
                    "__error__": f"Cannot pickle generator: {e}",
                }

            # Check if it's a callable (e.g., local function/closure)
            if callable(value):
                return {
                    "__callable__": True,
                    "__callable_type__": type(value).__name__,
                    "name": getattr(value, "__name__", ""),
                    "qualname": getattr(value, "__qualname__", ""),
                    "module": getattr(value, "__module__", ""),
                }

            # If pickling fails, try to create a serializable representation
            # First, try to convert iterables to plain lists (for HomogeneousList, etc.)
            if hasattr(value, "__iter__") and not isinstance(value, (str, bytes, dict)):
                try:
                    # Try to convert to a plain list and serialize elements
                    plain_list = [self._serialize_value(item) for item in value]
                    # Test if the list can be pickled
                    pickle.dumps(plain_list)
                    return plain_list
                except Exception:
                    # If list conversion fails, continue to other methods
                    pass

            if hasattr(value, "__dict__"):
                # For class instances, create a dict representation
                # Safely serialize the __dict__ to avoid generator issues
                try:
                    serialized_dict = self._serialize_dict_safely(value.__dict__)
                except Exception as dict_error:
                    serialized_dict = {"__dict_error__": f"Cannot serialize __dict__: {dict_error}"}

                return {
                    "__class_instance__": True,
                    "__class_name__": value.__class__.__name__,
                    "__module__": getattr(value.__class__, "__module__", ""),
                    "__dict__": serialized_dict,
                    "__error__": str(e),
                }
            else:
                # Fall back to string representation
                return {
                    "__unpicklable__": True,
                    "__type__": type(value).__name__,
                    "__str__": str(value),
                    "__error__": str(e),
                }

    def _deserialize_value(self, value: Any) -> Any:
        """Deserialize a value, handling class instances and generators."""
        if isinstance(value, dict) and value.get("__generator__"):
            # This is a serialized generator - cannot be deserialized
            return value  # Return the dict representation

        if isinstance(value, dict) and value.get("__class_instance__"):
            # This is a serialized class instance
            # For now, we'll return the dict representation
            # In a real implementation, you might want to reconstruct the class
            return value
        return value

    def load_snapshot(
        self, benchmark_name: str, module_path: str, parameters: tuple[Any, ...], class_name: Optional[str] = None
    ) -> Optional[tuple[Any, SnapshotMetadata]]:
        """Load a snapshot and its metadata."""

        param_hash = self._generate_param_hash(parameters)

        # Use class name in path if provided
        if class_name:
            benchmark_dir = f"{class_name}.{benchmark_name}"
        else:
            benchmark_dir = benchmark_name
        base_path = self.snapshot_dir / module_path / benchmark_dir / param_hash
        snapshot_path = self._resolve_snapshot_path(base_path)
        if snapshot_path is None:
            return None

        try:
            with self._open_snapshot_file(snapshot_path, "rb") as f:
                snapshot_data = pickle.load(f)

            serialized_value = snapshot_data["return_value"]
            metadata = snapshot_data["metadata"]

            # Deserialize the return value
            return_value = self._deserialize_value(serialized_value)

            return return_value, metadata

        except Exception as e:
            logger.warning(f"Failed to load snapshot {snapshot_path}: {e}")
            return None

    def is_failed_capture(
        self, benchmark_name: str, module_path: str, parameters: tuple[Any, ...]
    ) -> bool:
        """Check if a snapshot represents a failed capture."""
        snapshot_data = self.load_snapshot(benchmark_name, module_path, parameters)
        if snapshot_data is None:
            return False

        _, metadata = snapshot_data
        return metadata.capture_failed

    def list_snapshots(
        self, module_path: Optional[str] = None, benchmark_name: Optional[str] = None
    ) -> list[tuple[Path, SnapshotMetadata]]:
        """List all available snapshots."""
        snapshots = []

        search_dir = self.snapshot_dir
        if module_path:
            search_dir = search_dir / module_path
        if benchmark_name:
            search_dir = search_dir / benchmark_name

        if not search_dir.exists():
            return snapshots

        # Find all .pkl and .pkl.gz files
        snapshot_files = list(search_dir.rglob("*.pkl"))
        snapshot_files.extend(search_dir.rglob("*.pkl.gz"))
        for pkl_file in snapshot_files:
            try:
                with self._open_snapshot_file(pkl_file, "rb") as f:
                    snapshot_data = pickle.load(f)
                metadata = snapshot_data["metadata"]
                snapshots.append((pkl_file, metadata))
            except Exception as e:
                logger.warning(f"Failed to load snapshot {pkl_file}: {e}")
                continue

        return snapshots

    def delete_snapshot(
        self, benchmark_name: str, module_path: str, parameters: tuple[Any, ...]
    ) -> bool:
        """Delete a specific snapshot."""

        param_hash = self._generate_param_hash(parameters)
        base_path = self.snapshot_dir / module_path / benchmark_name / param_hash
        snapshot_path = self._resolve_snapshot_path(base_path)
        metadata_path = base_path.with_suffix(".json")

        deleted = False

        if snapshot_path and snapshot_path.exists():
            snapshot_path.unlink()
            deleted = True

        if metadata_path.exists():
            metadata_path.unlink()

        return deleted

    def _write_snapshot_data(self, snapshot_data: dict[str, Any], base_path: Path) -> Path:
        """Serialize and store snapshot data, compressing if over size threshold."""
        pickled = pickle.dumps(snapshot_data)
        if len(pickled) >= self.compress_threshold_bytes:
            snapshot_path = base_path.with_suffix(".pkl.gz")
            with gzip.open(snapshot_path, "wb") as f:
                f.write(pickled)
        else:
            snapshot_path = base_path.with_suffix(".pkl")
            with open(snapshot_path, "wb") as f:
                f.write(pickled)
        return snapshot_path

    def _resolve_snapshot_path(self, base_path: Path) -> Optional[Path]:
        """Return existing snapshot path (compressed or plain) if present."""
        gz_path = base_path.with_suffix(".pkl.gz")
        if gz_path.exists():
            return gz_path
        pkl_path = base_path.with_suffix(".pkl")
        if pkl_path.exists():
            return pkl_path
        return None

    def _open_snapshot_file(self, path: Path, mode: str):
        """Open snapshot file, handling gzip when needed."""
        if path.name.endswith(".pkl.gz"):
            return gzip.open(path, mode)
        return open(path, mode)

    def cleanup_empty_directories(self) -> None:
        """Remove empty directories in the snapshot tree."""
        for root, dirs, files in os.walk(self.snapshot_dir, topdown=False):
            for dir_name in dirs:
                dir_path = Path(root) / dir_name
                try:
                    if not any(dir_path.iterdir()):
                        dir_path.rmdir()
                except OSError:
                    pass  # Directory not empty or permission error

    def get_snapshot_stats(self) -> dict[str, Any]:
        """Get statistics about stored snapshots."""
        snapshots = self.list_snapshots()

        stats = {
            "total_snapshots": len(snapshots),
            "modules": set(),
            "benchmarks": set(),
            "oldest_snapshot": None,
            "newest_snapshot": None,
            "total_size_bytes": 0,
        }

        for snapshot_path, metadata in snapshots:
            stats["modules"].add(metadata.module_path)
            stats["benchmarks"].add(f"{metadata.module_path}.{metadata.benchmark_name}")

            if stats["oldest_snapshot"] is None or metadata.timestamp < stats["oldest_snapshot"]:
                stats["oldest_snapshot"] = metadata.timestamp

            if stats["newest_snapshot"] is None or metadata.timestamp > stats["newest_snapshot"]:
                stats["newest_snapshot"] = metadata.timestamp

            try:
                stats["total_size_bytes"] += snapshot_path.stat().st_size
            except OSError:
                pass

        stats["modules"] = list(stats["modules"])
        stats["benchmarks"] = list(stats["benchmarks"])

        return stats

    def _generate_param_hash(self, parameters: tuple[Any, ...]) -> str:
        """Generate a hash for parameter combination."""
        # Convert parameters to a string representation for hashing
        param_str = str(parameters)
        return hashlib.md5(param_str.encode()).hexdigest()[:16]

    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            import subprocess

            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()[:12]  # Short hash
        except Exception:
            pass
        return None

    def _get_git_branch(self) -> Optional[str]:
        """Get current git branch."""
        try:
            import subprocess

            result = subprocess.run(
                ["git", "branch", "--show-current"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    def _get_python_version(self) -> str:
        """Get Python version."""
        import sys

        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    def _get_platform(self) -> str:
        """Get platform information."""
        import platform

        return f"{platform.system()}-{platform.machine()}"
