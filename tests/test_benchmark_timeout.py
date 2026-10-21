"""A benchmark past its timeout is stopped, not only abandoned."""

import time
from pathlib import Path

from snapshot_tool.discovery import BenchmarkDiscovery
from snapshot_tool.runner import BenchmarkRunner


def _bench_dir(tmp_path: Path) -> Path:
    d = tmp_path / "benchmarks"
    d.mkdir()
    (d / "__init__.py").write_text("")
    (d / "bench_slow.py").write_text(
        "import time\n"
        "DONE = []\n"
        "def time_sleeps():\n"
        "    try:\n"
        "        time.sleep(30)\n"
        "    except Exception:\n"
        "        pass\n"
        "    DONE.append(1)\n"
        "    return 1\n"
        "def time_spins():\n"
        "    n = 0\n"
        "    while True:\n"
        "        n += 1\n"
        "def time_fast():\n"
        "    return 2\n"
    )
    return d


def _bench(d: Path, name: str):
    return next(b for b in BenchmarkDiscovery(d).discover_all() if b.name == name)


def test_a_slow_benchmark_is_stopped_at_its_timeout(tmp_path):
    d = _bench_dir(tmp_path)
    runner = BenchmarkRunner(d, project_dir=tmp_path, timeout=0.5)
    for name in ("time_sleeps", "time_spins"):
        t0 = time.monotonic()
        result = runner.run_benchmark(_bench(d, name))
        assert time.monotonic() - t0 < 5
        assert not result.success and isinstance(result.error, TimeoutError)
    time.sleep(1)
    module = runner._load_module(_bench(d, "time_sleeps").module_path)
    assert module.DONE == []
    fast = runner.run_benchmark(_bench(d, "time_fast"))
    assert fast.success and fast.return_value == 2
