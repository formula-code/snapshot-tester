# snapshot-tool

`snapshot-tool` checks that [ASV](https://github.com/airspeed-velocity/asv) benchmarks still give the same results after a code change. It runs each benchmark, captures a return value, stores it, and compares it on a later run with numerical tolerance. ASV measures speed; `snapshot-tool` checks correctness.

## Installation

```bash
uv pip install git+https://github.com/formula-code/snapshot-tester.git
# or
pip install git+https://github.com/formula-code/snapshot-tester.git
```

For development:

```bash
git clone https://github.com/formula-code/snapshot-tester.git
cd snapshot-tester
uv sync --group dev && uv pip install -e .
```

The tool has no required runtime dependencies. It supports Python 3.8 to 3.13.

## Usage

```bash
snapshot-tool list     path/to/benchmarks          # show discovered benchmarks
snapshot-tool capture  path/to/benchmarks          # store reference outputs
snapshot-tool baseline path/to/benchmarks          # optional: record pass/fail/skip per test
snapshot-tool verify   path/to/benchmarks          # compare against the stored outputs
```

`capture`, `baseline` and `verify` accept `--filter REGEX`, `--snapshot-dir DIR`, `--timeout SEC` and `--parallel [--workers N]`. `baseline` and `verify` also accept `--tolerance RTOL ATOL`. `verify` writes `summary.json` and exits with code 1 when a test fails. When `baseline.json` exists, the summary also holds the baseline-to-verify transition counts, for example `pass-to-fail`.

Snapshots go to `.snapshots/snapshots.db`, a SQLite database of gzipped pickles that are deduplicated by sha256.

## Documentation

The full documentation is in [`docs/`](docs/index.md) and builds with MkDocs (`mkdocs.yml`):

- [CLI reference](docs/guide/cli.md)
- [Python API quickstart](docs/getting-started/quickstart.md)
- [Configuration](docs/guide/configuration.md)
- [Baseline and verify](docs/guide/baseline-and-verify.md)
- [Discovery](docs/guide/discovery.md), [Tracing](docs/guide/tracing.md), [Comparison](docs/guide/comparison.md), [Determinism](docs/guide/determinism.md), [Storage](docs/guide/storage.md)

## Limitations

- The benchmarked project must be importable.
- Some benchmarks do not return a value that can be captured.
- Tracing makes benchmarks run slower.
- Pickled snapshots can fail to load on a different Python or library version. Capture and verify with the same environment.

## Tests

```bash
uv run pytest -v --ignore=tests/test_repos/ --ignore=tests/test_cli_roundtrip.py
```
