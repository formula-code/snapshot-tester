# Snapshot Storage

`SnapshotManager` reads and writes everything under `--snapshot-dir` (default `.snapshots/`). All captured return values are in one **SQLite database** at `<snapshot_dir>/snapshots.db`, content-addressed and gzipped. A **JSON metadata sidecar** per test is also written in a per-`(module, benchmark)` directory, so downstream tooling can read metadata without opening the database.

## Directory layout

```
.snapshots/
├── snapshots.db                                     # SQLite DB: blobs + snapshot rows
├── baseline.json                                    # produced by `baseline`; consumed by `verify`
├── <module_path>/
│   ├── <benchmark_name>/                            # function-level benchmark
│   │   └── <param-hash>.json                        # metadata sidecar (one per snapshot)
│   └── <class_name>.<benchmark_name>/               # class-method benchmark
│       └── <param-hash>.json
```

- `<module_path>` is the dotted import path from [discovery](discovery.md), with `.` preserved (not converted back to `/`). `benchmarks/geometry/union.py` → directory `geometry.union/`.
- `<benchmark_name>` for free functions; `<class_name>.<benchmark_name>` for methods. The class is included to disambiguate same-named benchmark methods on different classes in the same module.
- `<param-hash>` is the first 16 hex characters of `md5(str(tuple(parameters)))`. Empty `parameters=()` always hashes to `bcd8b0c2eb1fce71`.

Payloads are stored only in the database; there are no `.pkl` or `.pkl.gz` files.

## SQLite schema

Two tables, joined by `blob_hash`:

```sql
CREATE TABLE blobs (
    hash             TEXT PRIMARY KEY,    -- sha256 of raw pickle bytes
    data             BLOB NOT NULL,       -- gzipped pickle of return_value
    refcount         INTEGER NOT NULL DEFAULT 0,
    raw_size         INTEGER NOT NULL,
    compressed_size  INTEGER NOT NULL
);

CREATE TABLE snapshots (
    test_id          TEXT PRIMARY KEY,    -- "<module>/<bench-dir>/<param-hash>"
    module_path      TEXT NOT NULL,
    benchmark_name   TEXT NOT NULL,
    class_name       TEXT,
    param_hash       TEXT NOT NULL,
    parameters       BLOB NOT NULL,       -- pickled tuple
    param_names      BLOB,                -- pickled list[str] or NULL
    blob_hash        TEXT,                -- NULL for failed-capture rows
    capture_failed   INTEGER NOT NULL DEFAULT 0,
    failure_reason   TEXT,
    timestamp        TEXT NOT NULL,
    git_commit       TEXT,
    git_branch       TEXT,
    python_version   TEXT,
    platform         TEXT,
    FOREIGN KEY (blob_hash) REFERENCES blobs(hash)
);

CREATE INDEX idx_snapshots_module ON snapshots(module_path);
CREATE INDEX idx_snapshots_module_bench ON snapshots(module_path, benchmark_name);
```

PRAGMAs set at open: `journal_mode = WAL`, `synchronous = NORMAL`, `foreign_keys = ON`.

## Content addressing and deduplication

Each captured return value is pickled, and the raw pickle bytes are hashed with sha256. The `blobs` table is keyed by that hash, so two benchmarks with **identical** outputs share one blob; only the `refcount` goes up. In real benchmark suites, many parameter combinations return the same value, so deduplication saves space.

The compression ratio depends on what your benchmarks return. Repetitive text or dictionaries compress well; random float64 numpy arrays compress very little. Deduplication does not depend on how well the data compresses.

### Refcount lifecycle

Every `store_snapshot` commits one SQLite transaction:

1. Pickle the return value and compute its sha256.
2. Insert the blob if the hash is new; otherwise increment its `refcount`.
3. Read the current `blob_hash` (if any) for this `test_id`.
4. Insert or update the `snapshots` row with the new `blob_hash`.
5. Decrement the old blob's `refcount`. Delete the blob when the count reaches zero.

`delete_snapshot` releases the blob in the same way. Overwriting a snapshot with the same value writes no new blob bytes: the old and new hashes are equal and the net refcount change is zero.

## JSON metadata sidecars

Every snapshot — successful or failed-capture — gets a JSON sidecar at `.snapshots/<module>/<class.benchmark>/<param-hash>.json` containing the full `SnapshotMetadata` block:

```json
{
  "benchmark_name": "time_union",
  "module_path": "geometry.union",
  "parameters": [100, "polygon"],
  "param_names": ["n", "geom_type"],
  "timestamp": "2026-05-14T10:14:32.018000",
  "class_name": "TimeUnion",
  "git_commit": "abc123def456",
  "git_branch": "main",
  "python_version": "3.12.5",
  "platform": "Darwin-arm64",
  "capture_failed": false,
  "failure_reason": null
}
```

The sidecars are for downstream tooling (CI summaries, dashboards); `verify` reads metadata from SQLite, not from the JSON. Every `store_snapshot` and `store_failed_capture` writes the sidecar again, so it matches the database row.

## Serialization of unpicklable values

Not every Python value can round-trip through `pickle.dumps`/`pickle.loads`. `serialize_value` tests the round-trip. If it fails, an iterable that is not a string, bytes or dict becomes a list of serialized items when possible; otherwise the value becomes a tagged dict:

| Tag | Meaning | What `Comparator` does |
|-----|---------|------------------------|
| `__generator__` | A generator (has `__iter__` and `__next__`). | Skip comparison. |
| `__callable__` | A function/closure/method. The dict carries `name`, `qualname`, `module`. | Skip comparison. |
| `__unpicklable__` | Anything else whose pickle round-trip raised; stored as `__str__` plus `__type__`. | Skip comparison. |
| `__class_instance__` | An object with a `__dict__` that could not be pickled. Carries `__class_name__`, `__module__`, and the serialized `__dict__`. | Recursive attribute-by-attribute comparison against the actual instance's `__dict__`. |

These placeholder dicts can be pickled, so they go into the `blobs` table like any other captured value.

## Failed-capture markers

When a benchmark raises an error or times out, `capture` calls `store_failed_capture(...)` instead of `store_snapshot(...)`. The resulting `snapshots` row has `capture_failed = 1`, `blob_hash = NULL`, and a `failure_reason` string describing what went wrong. The JSON sidecar is still written so failed captures are visible on disk without opening the DB.

`verify` checks `capture_failed` after loading and reports a `[SKIP]`. If a flaky benchmark failed during capture, later verify runs skip it until you capture again.

## `test_id`

```python
storage.get_test_id(
    module_path="geometry.union",
    benchmark_name="time_union",
    parameters=(100, "polygon"),
    class_name="TimeUnion",
)
# → "geometry.union/TimeUnion.time_union/<md5-prefix>"
```

`test_id` is the primary key of the `snapshots` table and the path of the JSON sidecar relative to the snapshot directory, without `.json`. It is also the key in `baseline.json`. See [Baseline & Verify](baseline-and-verify.md) for how it's consumed.

## Programmatic use

```python
from snapshot_tool import SnapshotManager

storage = SnapshotManager(".snapshots/")

# Write
storage.store_snapshot(
    benchmark_name="time_compute",
    module_path="foo",
    parameters=(10,),
    param_names=["n"],
    return_value=[1.0, 2.0, 3.0],
)

# Read
loaded = storage.load_snapshot("time_compute", "foo", (10,))
if loaded is not None:
    value, metadata = loaded

# Inspect
stats = storage.get_snapshot_stats()
# {
#   "total_snapshots": N,
#   "unique_blobs": M,                       # <= N when dedup hit
#   "modules": [...], "benchmarks": [...],
#   "oldest_snapshot": dt, "newest_snapshot": dt,
#   "total_size_bytes": <compressed bytes on disk>,
#   "uncompressed_size_bytes": <raw pickle bytes>,
# }

# List
for sidecar_path, meta in storage.list_snapshots(module_path="foo"):
    print(sidecar_path, meta.capture_failed)

# Delete a specific row (decrements the underlying blob refcount automatically)
storage.delete_snapshot("time_compute", "foo", (10,))

# Close the underlying SQLite connection (also runs on __del__)
storage.close()
```

`store_snapshot` / `store_failed_capture` return the **JSON sidecar path**, a stable file for downstream tooling. The payload itself is in `snapshots.db`.

## Operational notes

- **Treat `.snapshots/` as disposable build output.** `customtest.sh` `rm -rf`s it before each run. It is not source.
- **Do not commit `snapshots.db` to version control.** Binary diffs are not readable, and the metadata fields (`timestamp`, `git_commit`, `platform`) change on every run. Capture again from a known good commit on each CI run.
- **The database is local.** Use it from one process at a time on a local disk. Shared filesystems and access from several hosts are not supported.
- **Pickles across Python versions.** `pickle.HIGHEST_PROTOCOL` differs between Python versions, so a snapshot captured on Python 3.12 may not load on Python 3.8. Capture and verify with the same interpreter.
- **Old `.pkl` / `.pkl.gz` snapshot directories are not read.** Capture again to create `snapshots.db`.

## Next steps

- [Comparison](comparison.md) — How serialized placeholders are interpreted at verify time.
- [Baseline & Verify](baseline-and-verify.md) — How `baseline.json` is structured and used.
