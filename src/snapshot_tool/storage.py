"""
Snapshot storage and management system.

Snapshots live in a single SQLite database at ``<snapshot_dir>/snapshots.db``.
The captured return values are pickled, gzipped, and stored in a content-addressed
``blobs`` table (sha256 → gzipped pickle, refcounted) so that benchmarks producing
identical outputs share a single payload on disk. Per-test metadata lives in a
``snapshots`` table that references the blob by hash.

A JSON metadata sidecar is still written next to where the per-test entry would
have lived under the old layout (``<snapshot_dir>/<module>/<class.benchmark>/<param_hash>.json``)
because downstream tooling consumes it.

The baseline file (``<snapshot_dir>/baseline.json``) is unchanged — it is small,
human-readable, and easy to diff in CI.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import pickle
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS blobs (
    hash             TEXT PRIMARY KEY,
    data             BLOB NOT NULL,
    refcount         INTEGER NOT NULL DEFAULT 0,
    raw_size         INTEGER NOT NULL,
    compressed_size  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS snapshots (
    test_id          TEXT PRIMARY KEY,
    module_path      TEXT NOT NULL,
    benchmark_name   TEXT NOT NULL,
    class_name       TEXT,
    param_hash       TEXT NOT NULL,
    parameters       BLOB NOT NULL,
    param_names      BLOB,
    blob_hash        TEXT,
    capture_failed   INTEGER NOT NULL DEFAULT 0,
    failure_reason   TEXT,
    timestamp        TEXT NOT NULL,
    git_commit       TEXT,
    git_branch       TEXT,
    python_version   TEXT,
    platform         TEXT,
    FOREIGN KEY (blob_hash) REFERENCES blobs(hash)
);

CREATE INDEX IF NOT EXISTS idx_snapshots_module
    ON snapshots(module_path);
CREATE INDEX IF NOT EXISTS idx_snapshots_module_bench
    ON snapshots(module_path, benchmark_name);
"""


@dataclass
class SnapshotMetadata:
    """Metadata for a snapshot."""

    benchmark_name: str
    module_path: str
    parameters: tuple
    param_names: Optional[list]
    timestamp: datetime
    class_name: Optional[str] = None
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    python_version: Optional[str] = None
    platform: Optional[str] = None
    capture_failed: bool = False
    failure_reason: Optional[str] = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> SnapshotMetadata:
        if isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class SnapshotManager:
    """Manages snapshot storage in a SQLite database with deduplicated, gzipped payloads."""

    DB_NAME = "snapshots.db"

    def __init__(
        self,
        snapshot_dir,
        *,
        compress_threshold_bytes: int = 0,  # accepted for API back-compat; unused
    ):
        # ``compress_threshold_bytes`` is intentionally accepted but ignored —
        # the SQLite backend always gzips every payload. Kept in the signature
        # so callers passing it as a keyword don't crash.
        del compress_threshold_bytes

        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.snapshot_dir / self.DB_NAME
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Baseline utilities (unchanged — baseline.json stays a flat file)
    # ------------------------------------------------------------------

    def baseline_path(self) -> Path:
        return self.snapshot_dir / "baseline.json"

    def write_baseline(self, entries: dict, meta: Optional[dict] = None) -> Path:
        payload: dict = {
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

    def read_baseline(self) -> Optional[dict]:
        path = self.baseline_path()
        if not path.exists():
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read baseline file {path}: {e}")
            return None

    def get_test_id(
        self,
        *,
        module_path: str,
        benchmark_name: str,
        parameters: tuple,
        class_name: Optional[str] = None,
    ) -> str:
        """Return a stable identifier for a (benchmark, parameters) pair."""
        param_hash = self._generate_param_hash(parameters)
        bench_dir = f"{class_name}.{benchmark_name}" if class_name else benchmark_name
        return f"{module_path}/{bench_dir}/{param_hash}"

    # ------------------------------------------------------------------
    # Public snapshot API
    # ------------------------------------------------------------------

    def store_snapshot(
        self,
        benchmark_name: str,
        module_path: str,
        parameters: tuple,
        param_names: Optional[list],
        return_value: Any,
        class_name: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Path:
        """Store a snapshot and its metadata; returns the JSON sidecar path."""

        # 1. Serialize the return value (pickle + best-effort placeholder fallback)
        serialized = self._serialize_value(return_value)

        snapshot_metadata = SnapshotMetadata(
            benchmark_name=benchmark_name,
            module_path=module_path,
            parameters=tuple(parameters),
            param_names=param_names,
            class_name=class_name,
            timestamp=datetime.now(),
            git_commit=self._get_git_commit(),
            git_branch=self._get_git_branch(),
            python_version=self._get_python_version(),
            platform=self._get_platform(),
            **(metadata or {}),
        )

        try:
            self._write_row(
                snapshot_metadata,
                return_value=serialized,
                capture_failed=False,
                failure_reason=None,
            )
        except Exception as e:
            # Pickle of the value itself failed (the serializer's placeholders
            # should have prevented this; this is a belt-and-braces fallback).
            placeholder = {
                "__unpicklable__": True,
                "__error__": f"Pickle failed: {e}",
            }
            self._write_row(
                snapshot_metadata,
                return_value=placeholder,
                capture_failed=False,
                failure_reason=None,
            )

        return self._write_json_sidecar(snapshot_metadata)

    def store_failed_capture(
        self,
        benchmark_name: str,
        module_path: str,
        parameters: tuple,
        param_names: Optional[list],
        failure_reason,
        class_name: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Path:
        """Record a failed capture; returns the JSON sidecar path."""

        # Coerce non-string failure_reason (e.g., a raw Exception passed in by
        # an embedding harness) to a string — the column is TEXT.
        if failure_reason is not None and not isinstance(failure_reason, str):
            failure_reason = f"{type(failure_reason).__name__}: {failure_reason}"

        snapshot_metadata = SnapshotMetadata(
            benchmark_name=benchmark_name,
            module_path=module_path,
            parameters=tuple(parameters),
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

        self._write_row(
            snapshot_metadata,
            return_value=None,  # no payload for failed captures
            capture_failed=True,
            failure_reason=failure_reason,
        )

        return self._write_json_sidecar(snapshot_metadata)

    def load_snapshot(
        self,
        benchmark_name: str,
        module_path: str,
        parameters: tuple,
        class_name: Optional[str] = None,
    ):
        """Load (return_value, SnapshotMetadata) for a snapshot, or None if missing."""

        test_id = self.get_test_id(
            module_path=module_path,
            benchmark_name=benchmark_name,
            parameters=tuple(parameters),
            class_name=class_name,
        )
        row = self._conn.execute(
            """
            SELECT s.module_path, s.benchmark_name, s.class_name,
                   s.parameters, s.param_names,
                   s.capture_failed, s.failure_reason, s.timestamp,
                   s.git_commit, s.git_branch, s.python_version, s.platform,
                   b.data
              FROM snapshots s
         LEFT JOIN blobs b ON b.hash = s.blob_hash
             WHERE s.test_id = ?
            """,
            (test_id,),
        ).fetchone()

        if row is None:
            return None

        try:
            (
                module_path_db,
                benchmark_name_db,
                class_name_db,
                parameters_blob,
                param_names_blob,
                capture_failed,
                failure_reason,
                timestamp,
                git_commit,
                git_branch,
                python_version,
                platform,
                data,
            ) = row

            metadata = SnapshotMetadata(
                benchmark_name=benchmark_name_db,
                module_path=module_path_db,
                parameters=tuple(pickle.loads(parameters_blob)),
                param_names=pickle.loads(param_names_blob)
                if param_names_blob is not None
                else None,
                class_name=class_name_db,
                timestamp=datetime.fromisoformat(timestamp),
                git_commit=git_commit,
                git_branch=git_branch,
                python_version=python_version,
                platform=platform,
                capture_failed=bool(capture_failed),
                failure_reason=failure_reason,
            )

            if data is None:
                # Failed-capture row, or value was never stored
                return self._deserialize_value(None), metadata

            raw = gzip.decompress(data)
            value = self._deserialize_value(pickle.loads(raw))
            return value, metadata
        except Exception as e:
            logger.warning(f"Failed to load snapshot {test_id}: {e}")
            return None

    def is_failed_capture(
        self,
        benchmark_name: str,
        module_path: str,
        parameters: tuple,
    ) -> bool:
        loaded = self.load_snapshot(benchmark_name, module_path, parameters)
        if loaded is None:
            return False
        _, metadata = loaded
        return metadata.capture_failed

    def list_snapshots(
        self,
        module_path: Optional[str] = None,
        benchmark_name: Optional[str] = None,
    ):
        """List all snapshots as a list of (json_sidecar_path, SnapshotMetadata) tuples."""

        query = """
            SELECT module_path, benchmark_name, class_name, param_hash,
                   parameters, param_names,
                   capture_failed, failure_reason, timestamp,
                   git_commit, git_branch, python_version, platform
              FROM snapshots
        """
        clauses = []
        args: list = []
        if module_path:
            clauses.append("module_path = ?")
            args.append(module_path)
        if benchmark_name:
            clauses.append("benchmark_name = ?")
            args.append(benchmark_name)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)

        results = []
        for row in self._conn.execute(query, args).fetchall():
            (
                module_path_db,
                benchmark_name_db,
                class_name_db,
                param_hash,
                parameters_blob,
                param_names_blob,
                capture_failed,
                failure_reason,
                timestamp,
                git_commit,
                git_branch,
                python_version,
                platform,
            ) = row
            try:
                metadata = SnapshotMetadata(
                    benchmark_name=benchmark_name_db,
                    module_path=module_path_db,
                    parameters=tuple(pickle.loads(parameters_blob)),
                    param_names=(
                        pickle.loads(param_names_blob) if param_names_blob is not None else None
                    ),
                    class_name=class_name_db,
                    timestamp=datetime.fromisoformat(timestamp),
                    git_commit=git_commit,
                    git_branch=git_branch,
                    python_version=python_version,
                    platform=platform,
                    capture_failed=bool(capture_failed),
                    failure_reason=failure_reason,
                )
            except Exception as e:
                logger.warning(f"Failed to load metadata row for {benchmark_name_db}: {e}")
                continue
            sidecar = self._sidecar_path(
                module_path_db, benchmark_name_db, class_name_db, param_hash
            )
            results.append((sidecar, metadata))
        return results

    def delete_snapshot(
        self,
        benchmark_name: str,
        module_path: str,
        parameters: tuple,
        class_name: Optional[str] = None,
    ) -> bool:
        """Delete a snapshot row (and decrement the underlying blob refcount)."""

        test_id = self.get_test_id(
            module_path=module_path,
            benchmark_name=benchmark_name,
            parameters=tuple(parameters),
            class_name=class_name,
        )
        with self._conn:
            row = self._conn.execute(
                "SELECT blob_hash, param_hash FROM snapshots WHERE test_id = ?",
                (test_id,),
            ).fetchone()
            if row is None:
                return False
            old_blob_hash, param_hash = row
            self._conn.execute("DELETE FROM snapshots WHERE test_id = ?", (test_id,))
            if old_blob_hash is not None:
                self._release_blob(old_blob_hash)

        # Remove JSON sidecar if present
        sidecar = self._sidecar_path(module_path, benchmark_name, class_name, param_hash)
        if sidecar.exists():
            try:
                sidecar.unlink()
            except OSError:
                pass
        return True

    def get_snapshot_stats(self) -> dict:
        """Aggregate stats over the snapshot DB."""

        total_snapshots = self._conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]
        size_row = self._conn.execute(
            "SELECT COALESCE(SUM(compressed_size), 0), COALESCE(SUM(raw_size), 0), COUNT(*) "
            "FROM blobs"
        ).fetchone()
        total_compressed, total_raw, unique_blobs = size_row

        modules = [r[0] for r in self._conn.execute("SELECT DISTINCT module_path FROM snapshots")]
        benchmarks = [
            f"{r[0]}.{r[1]}"
            for r in self._conn.execute(
                "SELECT DISTINCT module_path, benchmark_name FROM snapshots"
            )
        ]

        ts_row = self._conn.execute(
            "SELECT MIN(timestamp), MAX(timestamp) FROM snapshots"
        ).fetchone()
        oldest_ts, newest_ts = ts_row
        oldest = datetime.fromisoformat(oldest_ts) if oldest_ts else None
        newest = datetime.fromisoformat(newest_ts) if newest_ts else None

        return {
            "total_snapshots": total_snapshots,
            "unique_blobs": unique_blobs,
            "modules": modules,
            "benchmarks": benchmarks,
            "oldest_snapshot": oldest,
            "newest_snapshot": newest,
            "total_size_bytes": int(total_compressed),
            "uncompressed_size_bytes": int(total_raw),
        }

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        try:
            self._conn.close()
        except Exception:
            pass

    def __del__(self):
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_row(
        self,
        meta: SnapshotMetadata,
        return_value: Any,
        capture_failed: bool,
        failure_reason: Optional[str],
    ) -> None:
        test_id = self.get_test_id(
            module_path=meta.module_path,
            benchmark_name=meta.benchmark_name,
            parameters=meta.parameters,
            class_name=meta.class_name,
        )
        param_hash = self._generate_param_hash(meta.parameters)

        blob_hash: Optional[str] = None
        if not capture_failed:
            blob_hash = self._store_blob(return_value)

        params_blob = pickle.dumps(tuple(meta.parameters), protocol=pickle.HIGHEST_PROTOCOL)
        param_names_blob = (
            pickle.dumps(list(meta.param_names), protocol=pickle.HIGHEST_PROTOCOL)
            if meta.param_names is not None
            else None
        )

        with self._conn:
            # Decrement refcount of any previously-stored blob for this test_id
            prev = self._conn.execute(
                "SELECT blob_hash FROM snapshots WHERE test_id = ?",
                (test_id,),
            ).fetchone()
            old_blob_hash = prev[0] if prev else None

            self._conn.execute(
                """
                INSERT INTO snapshots (
                    test_id, module_path, benchmark_name, class_name, param_hash,
                    parameters, param_names,
                    blob_hash, capture_failed, failure_reason,
                    timestamp, git_commit, git_branch, python_version, platform
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(test_id) DO UPDATE SET
                    module_path     = excluded.module_path,
                    benchmark_name  = excluded.benchmark_name,
                    class_name      = excluded.class_name,
                    param_hash      = excluded.param_hash,
                    parameters      = excluded.parameters,
                    param_names     = excluded.param_names,
                    blob_hash       = excluded.blob_hash,
                    capture_failed  = excluded.capture_failed,
                    failure_reason  = excluded.failure_reason,
                    timestamp       = excluded.timestamp,
                    git_commit      = excluded.git_commit,
                    git_branch      = excluded.git_branch,
                    python_version  = excluded.python_version,
                    platform        = excluded.platform
                """,
                (
                    test_id,
                    meta.module_path,
                    meta.benchmark_name,
                    meta.class_name,
                    param_hash,
                    params_blob,
                    param_names_blob,
                    blob_hash,
                    1 if capture_failed else 0,
                    failure_reason,
                    meta.timestamp.isoformat(),
                    meta.git_commit,
                    meta.git_branch,
                    meta.python_version,
                    meta.platform,
                ),
            )

            if old_blob_hash is not None:
                self._release_blob(old_blob_hash)

    def _store_blob(self, value: Any) -> str:
        """Pickle + gzip the value, insert into ``blobs`` (or bump refcount). Return its hash."""
        raw = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        h = hashlib.sha256(raw).hexdigest()

        row = self._conn.execute("SELECT 1 FROM blobs WHERE hash = ?", (h,)).fetchone()
        if row is None:
            compressed = gzip.compress(raw)
            self._conn.execute(
                "INSERT INTO blobs (hash, data, refcount, raw_size, compressed_size) "
                "VALUES (?, ?, 1, ?, ?)",
                (h, compressed, len(raw), len(compressed)),
            )
        else:
            self._conn.execute(
                "UPDATE blobs SET refcount = refcount + 1 WHERE hash = ?",
                (h,),
            )
        return h

    def _release_blob(self, blob_hash: str) -> None:
        """Decrement a blob's refcount; delete the row if it falls to zero."""
        self._conn.execute(
            "UPDATE blobs SET refcount = refcount - 1 WHERE hash = ?",
            (blob_hash,),
        )
        self._conn.execute(
            "DELETE FROM blobs WHERE hash = ? AND refcount <= 0",
            (blob_hash,),
        )

    def _sidecar_path(
        self,
        module_path: str,
        benchmark_name: str,
        class_name: Optional[str],
        param_hash: str,
    ) -> Path:
        bench_dir = f"{class_name}.{benchmark_name}" if class_name else benchmark_name
        return self.snapshot_dir / module_path / bench_dir / f"{param_hash}.json"

    def _write_json_sidecar(self, meta: SnapshotMetadata) -> Path:
        param_hash = self._generate_param_hash(meta.parameters)
        path = self._sidecar_path(
            meta.module_path, meta.benchmark_name, meta.class_name, param_hash
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(meta.to_dict(), f, indent=2, default=str)
        return path

    # ------------------------------------------------------------------
    # Value serialization (unchanged placeholder semantics)
    # ------------------------------------------------------------------

    def _serialize_dict_safely(self, d: dict) -> dict:
        result = {}
        for key, value in d.items():
            try:
                result[self._serialize_value(key)] = self._serialize_value(value)
            except Exception as e:
                result[f"__error_{key}__"] = f"Cannot serialize: {e}"
        return result

    def _serialize_value(self, value: Any) -> Any:
        try:
            pickled = pickle.dumps(value)
            pickle.loads(pickled)  # round-trip test
            return value
        except Exception as e:
            if hasattr(value, "__iter__") and hasattr(value, "__next__"):
                return {
                    "__generator__": True,
                    "__generator_type__": type(value).__name__,
                    "__error__": f"Cannot pickle generator: {e}",
                }

            if callable(value):
                return {
                    "__callable__": True,
                    "__callable_type__": type(value).__name__,
                    "name": getattr(value, "__name__", ""),
                    "qualname": getattr(value, "__qualname__", ""),
                    "module": getattr(value, "__module__", ""),
                }

            if hasattr(value, "__iter__") and not isinstance(value, (str, bytes, dict)):
                try:
                    plain_list = [self._serialize_value(item) for item in value]
                    pickle.dumps(plain_list)
                    return plain_list
                except Exception:
                    pass

            if hasattr(value, "__dict__"):
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

            return {
                "__unpicklable__": True,
                "__type__": type(value).__name__,
                "__str__": str(value),
                "__error__": str(e),
            }

    def _deserialize_value(self, value: Any) -> Any:
        # Tagged dicts (__generator__, __class_instance__, ...) are returned as-is;
        # the Comparator interprets them.
        return value

    # ------------------------------------------------------------------
    # Metadata helpers (git, python, platform)
    # ------------------------------------------------------------------

    def _generate_param_hash(self, parameters: tuple) -> str:
        param_str = str(tuple(parameters))
        return hashlib.md5(param_str.encode()).hexdigest()[:16]

    def _get_git_commit(self) -> Optional[str]:
        try:
            import subprocess

            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()[:12]
        except Exception:
            pass
        return None

    def _get_git_branch(self) -> Optional[str]:
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
        import sys

        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    def _get_platform(self) -> str:
        import platform

        return f"{platform.system()}-{platform.machine()}"
