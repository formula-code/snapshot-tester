# Installation

`snapshot-tool` is pure Python (3.8 → 3.13) with no required runtime dependencies. Numpy is optional; the comparator uses it if it is installed.

## Prerequisites

- **Python 3.8 or newer** (3.8 through 3.13 are CI-tested)
- **[uv](https://astral.sh/uv/)** — recommended for development; the project uses `uv` for dependency management and the `uv_build` backend
- **ASV benchmarks** to test against — any directory of files with `time_*`, `timeraw_*`, `mem_*`, `peakmem_*`, or `track_*` functions/methods

## 1. Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

To use `pip` instead, skip this step and see [Using pip](#using-pip).

## 2. Clone and install

```bash
git clone https://github.com/formula-code/snapshot-tester.git
cd snapshot-tester

# Install the dev environment (creates .venv, installs all deps)
uv sync --group dev
uv pip install -e .
```

This installs the `snapshot-tool` console entry point along with the full dev toolchain: `pytest`, `ruff`, `mypy`, and the optional comparison libraries (`numpy`, `shapely`, `astropy`, `scipy`, `pandas`) used by the test suite.

## 3. Verify

Confirm the CLI is on your `$PATH`:

```bash
uv run snapshot-tool --help
```

You should see the top-level usage with six subcommands: `list`, `capture`, `verify`, `baseline`, `clean`, `config`.

Run the unit tests without the slow real-repo roundtrips (the same command as the `tests.yml` CI job):

```bash
uv run pytest -v --ignore=tests/test_repos/ --ignore=tests/test_cli_roundtrip.py
```

## 4. Smoke test against a bundled benchmark repo

The repo vendors three real benchmark suites under `tests/test_repos/` (astropy, pandas, shapely). The `customtest.sh` script runs the `list → capture → baseline → verify` roundtrip against the shapely suite:

```bash
bash customtest.sh
```

You can also point the CLI at any of them directly:

```bash
uv run snapshot-tool list tests/test_repos/shapely_benchmarks
uv run snapshot-tool capture tests/test_repos/shapely_benchmarks
uv run snapshot-tool verify tests/test_repos/shapely_benchmarks
```

!!! note
    The `tests/test_repos/` directories are excluded from `pytest` collection (`norecursedirs` in `pyproject.toml`) — they are vendored benchmark sources, not test files. They are exercised through `tests/test_cli_roundtrip.py`, which CI runs in dedicated per-suite jobs sharded by benchmark module regex.

## Using pip

If you don't use `uv`:

```bash
git clone https://github.com/formula-code/snapshot-tester.git
cd snapshot-tester
pip install -e .

# Or directly from the repo
pip install git+https://github.com/formula-code/snapshot-tester.git
```

The runtime has no required dependencies. The test suite needs the `dev` dependency group (`pytest`, `numpy`, `shapely`, `astropy`, and others). Install them by hand or with `uv sync --group dev`.

## Development tasks

The lint commands that the `lint.yml` CI job runs (check only, no automatic fixes):

```bash
uv run ruff format --check src/ tests/
uv run ruff check src/ tests/
```

CI runs the tests on Python 3.8 to 3.13. Locally, one interpreter is enough:

```bash
uv run pytest -v --ignore=tests/test_repos/ --ignore=tests/test_cli_roundtrip.py
```

The slow real-repo roundtrips (matching the per-suite CI jobs):

```bash
uv run pytest tests/test_cli_roundtrip.py -v

# Filter inside the roundtrip via env vars (the same knobs CI uses):
SNAPSHOT_TOOL_FILTER='^benchmarks\.(coordinates|units)' \
SNAPSHOT_TOOL_TIMEOUT=10 \
  uv run pytest tests/test_cli_roundtrip.py::TestAstropyRoundtrip -x
```

## Next steps

- [**CLI guide**](../guide/cli.md) — Run `list → capture → verify` against your own benchmarks.
- [Python API Quickstart](quickstart.md) — Use the modules programmatically.
- [Configuration](../guide/configuration.md) — `snapshot_config.json` and every CLI flag.
