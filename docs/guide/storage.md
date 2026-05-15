# Snapshot Storage

`SnapshotManager` owns everything under `--snapshot-dir` (default `.snapshots/`). All captured return values live in a single **SQLite database** at `<snapshot_dir>/snapshots.db`, content-addressed and gzipped for compactness. The per-test **JSON metadata sidecars** are still written to disk in their per-`(module, benchmark)` directory so downstream tooling can read them without opening the database.

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
- `<param-hash>` is the first 16 hex characters of `md5(repr(parameters))`. Empty `parameters=()` always hashes to `99914b932bd37a50`.

There are no `.pkl` or `.pkl.gz` files — everything that used to be a pickle now lives in the database.

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

## Why content-addressing + dedup

Each captured return value is pickled, then sha256-hashed by its raw pickle bytes. The `blobs` table is keyed by that hash, so two benchmarks producing **identical** outputs share one blob — only the `refcount` is incremented. This matters in real benchmark suites because many parameterized benchmarks emit the same trivial value (`None`, `0`, an empty list) across param combinations, and many benchmarks share computed sub-results.

A representative roundtrip on the bundled shapely suite:

```
58 snapshots → 45 unique blobs   (≈22% deduplicated)
raw payload size    111 MB
gzipped in SQLite    84 MB       (≈24% smaller after gzip)
```

Compression ratio depends entirely on what your benchmarks return. Repetitive text or dictionaries shrink 3–10×; incompressible float64 numpy arrays barely move. Either way the dedup wins are independent of the data's compressibility.

### Refcount lifecycle

Every `store_snapshot` runs in a single SQLite transaction:

1. Pickle the return value, compute its sha256.
2. Read the current `blob_hash` (if any) for this `test_id`.
3. Insert the new blob if novel; otherwise bump its `refcount`.
4. Upsert the `snapshots` row with the new `blob_hash`.
5. Decrement the old blob's `refcount`. Delete the blob if it falls to zero.

`delete_snapshot` follows the same release path. Overwriting a snapshot with the same value is cheap — the old and new blob hashes match, refcount is net zero, no new bytes get written.

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

The sidecars are convenience artefacts for downstream tooling (CI summaries, dashboards) — `verify` reads metadata from SQLite, not the JSON. They're rewritten on every `store_snapshot` to stay in sync with the DB row.

## Serialization of unpicklable values

Not every Python value can round-trip through `pickle.dumps`/`pickle.loads`. `_serialize_value` tests the round-trip and falls back to a tagged dict when it fails:

| Tag | Meaning | What `Comparator` does |
|-----|---------|------------------------|
| `__generator__` | A generator (has `__iter__` and `__next__`). | Skip comparison. |
| `__callable__` | A function/closure/method. The dict carries `name`, `qualname`, `module`. | Skip comparison. |
| `__unpicklable__` | Anything else whose pickle round-trip raised; stored as `__str__` plus `__type__`. | Skip comparison. |
| `__class_instance__` | A class instance whose `__dict__` could be serialized but the instance itself couldn't. Carries `__class_name__`, `__module__`, and `__dict__`. | Recursive attribute-by-attribute comparison against the actual instance's `__dict__`. |

These placeholder dicts are pickleable, so they land in the `blobs` table just like any other captured value.

## Failed-capture markers

When a benchmark crashes, times out, or returns something we can't even pickle as a placeholder, `capture` calls `store_failed_capture(...)` instead of `store_snapshot(...)`. The resulting `snapshots` row has `capture_failed = 1`, `blob_hash = NULL`, and a `failure_reason` string describing what went wrong. The JSON sidecar is still written so failed captures are visible on disk without opening the DB.

`verify` checks `capture_failed` after loading and treats it as a `[SKIP]`. If you captured against a benchmark that's flaky, the first failure goes into the DB and subsequent verifies don't re-fail on it — they skip until you re-capture.

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

`test_id` is the primary key on the `snapshots` table, and the same string used as the directory path for the JSON sidecar (minus `.snapshots/` and `.json`). It's also the key in `baseline.json`. See [Baseline & Verify](baseline-and-verify.md) for how it's consumed.

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

`store_snapshot` / `store_failed_capture` return the **JSON sidecar path** — useful when downstream tooling wants a stable on-disk artefact to point at. The payload itself lives in `snapshots.db`.

## Operational notes

- **Treat `.snapshots/` as disposable build output.** `customtest.sh` `rm -rf`s it before each run. It is not source.
- **Don't commit `snapshots.db` to version control.** Binary diffs are useless, and the metadata fields (`timestamp`, `git_commit`, `platform`) drift on every run. The recommended pattern is to re-capture from a known-good commit on each CI run.
- **The DB is local-only.** No locking story for shared filesystems, no replication. SQLite + WAL handles same-process concurrency safely, but cross-process or cross-host coordination is out of scope.
- **Cross-Python-version pickles.** The pickle format can change between major Python releases. If you capture on Python 3.12 and verify on Python 3.8, expect breakage. CI matrices should capture and verify on the same interpreter.
- **No backward compatibility for `.pkl` / `.pkl.gz` directories.** The SQLite backend is a clean break — existing pickle snapshot directories are not read. Re-capture against the new schema.

## Next steps

- [Comparison](comparison.md) — How serialized placeholders are interpreted at verify time.
- [Baseline & Verify](baseline-and-verify.md) — How `baseline.json` is structured and used.
