# Baseline & Verify

`snapshot-tool` has two related comparison subcommands that produce different outputs:

| Subcommand | What it writes | When to use |
|------------|----------------|-------------|
| `verify`   | `summary.json` with per-outcome counts (`passed`, `failed`, `skipped`). Exits non-zero on failure. | Day-to-day correctness gate. |
| `baseline` | `<snapshot_dir>/baseline.json` with a per-test pass/fail/skip map. Exits 0 unless an error occurs. | Record the *starting* state so a later `verify` can compute a transition matrix. |

When a `baseline.json` is present in the snapshot directory, `verify` reads it and adds the **transition counts** to `summary.json`.

## When to use which

`verify` answers "did anything regress?". `baseline` answers "what was the state on this revision?". For local development you usually only need `verify`. For grading an optimization PR — where you care about which tests changed status, not just the totals — you want `baseline` followed by `verify`.

The usual CI flow:

```mermaid
graph LR
    A[capture<br/>on revision R₀] --> B
    B[baseline<br/>on revision R₀] --> C
    C[apply patch] --> D
    D[verify<br/>on revision R₁] --> E
    E[transition matrix<br/>in summary.json]
```

## The transition matrix

`baseline` and the per-test statuses from `verify` use the same values: `pass`, `fail`, `skip`. `compute_transitions(baseline, verify)` gives a count for each `<baseline>-to-<verify>` pair that occurs in the data. Pairs that do not occur are left out of `summary.json`; the `verify` log prints all nine.

The cells, in `summary.json` ordering:

|              | → pass | → fail | → skip |
|--------------|--------|--------|--------|
| **from pass** | `pass-to-pass` | `pass-to-fail` | `pass-to-skip` |
| **from fail** | `fail-to-pass` | `fail-to-fail` | `fail-to-skip` |
| **from skip** | `skip-to-pass` | `skip-to-fail` | `skip-to-skip` |

Interpreting cells:

| Cell | What it usually means |
|------|----------------------|
| `pass-to-pass` | Stable correct test — your patch left this test intact. |
| `pass-to-fail` | **Regression** — a previously-passing test now produces a different output. |
| `pass-to-skip` | A test that passed in the baseline was skipped in verify, for example because it timed out. Investigate it. |
| `fail-to-pass` | Your patch fixed a previously-failing test. This is the win condition for optimization PRs that include corrections. |
| `fail-to-fail` | Still failing. The counts do not record the specific error, so the failure can be different. |
| `fail-to-skip` | Failed in the baseline; skipped in verify, for example because it timed out in verify. |
| `skip-to-pass` | Previously skipped (no snapshot or failed capture); now produces a comparable output. |
| `skip-to-fail` | Skipped during baseline (likely no snapshot), but now the test produces an output that's different from the **original capture**. Diagnose by inspecting what's in `.snapshots/`. |
| `skip-to-skip` | Persistently un-snapshottable — e.g., a benchmark that returns a generator. |

Only `test_id`s present in **both** `baseline.json` and the verify run are counted. Tests added or removed between the two runs are dropped without a warning.

## `test_id`

A `test_id` is a stable, on-disk-path-like string identifying one `(benchmark, parameters)` row:

```
<module_path>/[<class_name>.]<benchmark_name>/<param-hash>
```

The `param-hash` is the first 16 hex characters of the MD5 of `str(tuple(parameters))`. Snapshots use the same function, so a snapshot and a baseline entry for the same test always have the same `test_id`.

Two examples:

- A free function `time_compute()` in `benchmarks/foo.py` with no parameters:
  `foo/time_compute/bcd8b0c2eb1fce71`
- A method `time_union(self, n, geom_type)` on class `TimeUnion` in `benchmarks/geometry/union.py`, parameters `(100, "polygon")`:
  `geometry.union/TimeUnion.time_union/7204f98887b5c4d7` (the MD5 prefix of `"(100, 'polygon')"`)

You don't normally construct these by hand — `SnapshotManager.get_test_id(...)` does it consistently.

## What `baseline.json` looks like

```json
{
  "schema": "snapshot_tool/baseline@2",
  "timestamp": "2026-05-14T10:11:00.123456",
  "entries": {
    "geometry.union/TimeUnion.time_union/2f3a...": "pass",
    "geometry.union/TimeUnion.time_union/8c1e...": "fail",
    "coordinates/time_separation/bcd8b0c2eb1fce71": "skip"
  },
  "meta": {
    "counts": {"total": 124, "pass": 118, "fail": 2, "skip": 4},
    "snapshot_dir": ".snapshots",
    "benchmark_dir": "benchmarks"
  }
}
```

The schema tag (`snapshot_tool/baseline@2`) names the format. The reader accepts any payload with an `entries` dict and does not use the `meta` block.

## Status normalization

`compute_transitions` runs every status through `_normalize_status` before matching:

- The status `"failed_to_pass"` is mapped to `"fail"`, so baseline files that use that name still work.
- Any other value is kept as it is. `baseline` and `verify` write only `pass`, `fail`, and `skip`.

## Wiring into CI

A minimal optimization-grading workflow:

```yaml
# Pseudocode for a CI job
- name: Capture and baseline on main
  run: |
    git checkout main
    snapshot-tool capture  benchmarks --snapshot-dir .snapshots
    snapshot-tool baseline benchmarks --snapshot-dir .snapshots

- name: Verify on the PR branch
  run: |
    git checkout ${{ github.head_ref }}
    snapshot-tool verify   benchmarks --snapshot-dir .snapshots --summary summary.json

- name: Print transition counts
  run: |
    jq '. | {passed, failed, "pass-to-fail", "fail-to-pass"}' summary.json
```

The `verify` step will exit non-zero if any test failed. If you only want to grade transitions without failing the build on `pass-to-fail`, run `verify` with `|| true` and inspect `summary.json` yourself.

## Common pitfalls

- **`baseline` before `capture`.** `baseline` reads existing snapshots; it does not create them. If you run `baseline` before `capture`, every entry is `skip` (no snapshot) and the transition counts tell you nothing.
- **Different `--snapshot-dir` between `baseline` and `verify`.** Both must point at the same directory — `baseline.json` lives inside it.
- **Different `--filter` between `baseline` and `verify`.** Only `test_id`s in both runs are counted, so a different filter makes the overlap smaller without a warning.
- **Non-deterministic benchmarks.** If a benchmark returns a value that depends on randomness, on the iteration order of a `set`, or on the wall clock, it appears as `pass-to-fail` even when nothing changed. See [Determinism](determinism.md) for the RNG reseeding. Wall-clock and set-order problems must be fixed in the benchmark itself.

## Next steps

- [CLI](cli.md) — Per-subcommand reference.
- [Snapshot Storage](storage.md) — Where `baseline.json` lives and how `test_id`s are constructed.
- [Determinism](determinism.md) — Why RNG reseeding is necessary for stable transitions.
