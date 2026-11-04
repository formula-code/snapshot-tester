"""Run ASV benchmarks with tracing, setup methods, parameter combinations and timeouts."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import pickle
import signal
import sys
import threading
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .discovery import BenchmarkDiscovery, BenchmarkInfo
from .rng_patcher import RNGPatcher
from .tracer import ExecutionTracer, TraceResult

logger = logging.getLogger(__name__)


class BenchmarkTimeout(BaseException):
    """Raised inside a benchmark by the timer.

    It is a BaseException, so the benchmark's own ``except Exception`` cannot catch it.
    """


class BenchmarkRunner:
    """Executes ASV benchmarks with tracing enabled."""

    def __init__(
        self,
        benchmark_dir: Path,
        project_dir: Optional[Path] = None,
        seed: int = 42,
        timeout: Optional[float] = None,
    ):
        self.benchmark_dir = Path(benchmark_dir)
        self.project_dir = project_dir or benchmark_dir.parent
        self.discovery = BenchmarkDiscovery(self.benchmark_dir)
        self.tracer = ExecutionTracer()
        self.seed = seed
        self.timeout = timeout  # seconds per benchmark; None means no limit
        self.rng_patcher = RNGPatcher(seed=seed)

        self._module_cache: dict[str, Any] = {}

        # setup_cache() results, keyed by "module.Class"
        self._setup_cache: dict[str, Any] = {}

        if str(self.project_dir) not in sys.path:
            sys.path.insert(0, str(self.project_dir))

    def _reset_random_state(self):
        """Reseed Python random, the numpy legacy API, PyTorch and TensorFlow."""
        self.rng_patcher.patch_all()

    def _evaluate_params_at_runtime(self, benchmark: BenchmarkInfo, module) -> list[list[Any]]:
        """Evaluate params at runtime by accessing the class attribute."""
        if not benchmark.needs_runtime_eval or not benchmark.class_name:
            return benchmark.params

        try:
            benchmark_class = getattr(module, benchmark.class_name)
            params = getattr(benchmark_class, "params", None)
            if params is None:
                return None

            if isinstance(params, (list, tuple)):
                if len(params) > 0 and isinstance(params[0], (list, tuple)):
                    # One list per parameter
                    return [list(p) if isinstance(p, tuple) else p for p in params]
                else:
                    # One parameter
                    return [list(params) if isinstance(params, tuple) else params]
            else:
                return [[params]]

        except Exception as e:
            logger.warning(f"Failed to evaluate params at runtime for {benchmark.name}: {e}")
            return None

    def _run_with_timeout(
        self,
        benchmark: BenchmarkInfo,
        parameters: Optional[tuple[Any, ...]] = None,
    ) -> Optional[TraceResult]:
        """Run the benchmark; on timeout return a failed TraceResult with a TimeoutError."""
        if self.timeout is None:
            return self._run_benchmark_internal(benchmark, parameters)
        if hasattr(signal, "setitimer") and threading.current_thread() is threading.main_thread():
            return self._run_with_alarm(benchmark, parameters)

        # Off the main thread no signal can stop the benchmark; it keeps running after the wait ends
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self._run_benchmark_internal, benchmark, parameters)
        try:
            result = future.result(timeout=self.timeout)
            executor.shutdown(wait=True)
            return result
        except FuturesTimeoutError:
            logger.warning(f"Benchmark {benchmark.name} timed out after {self.timeout} seconds")
            future.cancel()
            executor.shutdown(wait=False)
            return TraceResult(
                return_value=None,
                function_name=benchmark.name,
                module_name=benchmark.module_path,
                depth=0,
                success=False,
                error=TimeoutError(f"Benchmark execution exceeded {self.timeout} seconds"),
            )
        except Exception as e:
            logger.error(f"Unexpected error in timeout wrapper: {e}")
            executor.shutdown(wait=False)
            return TraceResult(
                return_value=None,
                function_name=benchmark.name,
                module_name=benchmark.module_path,
                depth=0,
                success=False,
                error=e,
            )

    def _run_with_alarm(
        self, benchmark: BenchmarkInfo, parameters: Optional[tuple[Any, ...]] = None
    ) -> Optional[TraceResult]:
        """Run the benchmark in this thread and stop it with SIGALRM when the timeout passes."""

        def fire(_signum, _frame):
            raise BenchmarkTimeout()

        # Import before the timer: an import stopped halfway breaks the module for later benchmarks
        try:
            self._reset_random_state()
            self._load_module(benchmark.module_path)
        except Exception:
            pass
        previous = signal.signal(signal.SIGALRM, fire)
        try:
            signal.setitimer(signal.ITIMER_REAL, self.timeout)
            try:
                result = self._run_benchmark_internal(benchmark, parameters)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            return result
        except BenchmarkTimeout:
            sys.settrace(None)
            logger.warning(f"Benchmark {benchmark.name} stopped after {self.timeout} seconds")
            return TraceResult(
                return_value=None,
                function_name=benchmark.name,
                module_name=benchmark.module_path,
                depth=0,
                success=False,
                error=TimeoutError(f"Benchmark execution exceeded {self.timeout} seconds"),
            )
        finally:
            signal.signal(signal.SIGALRM, previous)

    def run_benchmark(
        self, benchmark: BenchmarkInfo, parameters: Optional[tuple[Any, ...]] = None
    ) -> Optional[TraceResult]:
        """Run a single benchmark with tracing and optional timeout."""
        return self._run_with_timeout(benchmark, parameters)

    def _run_benchmark_internal(
        self, benchmark: BenchmarkInfo, parameters: Optional[tuple[Any, ...]] = None
    ) -> Optional[TraceResult]:
        """Run a single benchmark with tracing and no timeout."""

        try:
            self._reset_random_state()

            module = self._load_module(benchmark.module_path)

            if benchmark.needs_runtime_eval:
                runtime_params = self._evaluate_params_at_runtime(benchmark, module)
                if runtime_params:
                    benchmark.params = runtime_params
                    benchmark.needs_runtime_eval = False

            # A parameterized method called without parameters uses the first combination
            if benchmark.params and parameters is None and benchmark.benchmark_type == "method":
                param_combinations = self.get_param_combinations(benchmark)
                if param_combinations:
                    parameters = param_combinations[0]
                    logger.warning(
                        f"Benchmark {benchmark.name} has parameters but none provided. "
                        f"Using first parameter combination: {parameters}"
                    )

            if benchmark.benchmark_type == "function":
                return self._run_function_benchmark(module, benchmark)
            elif benchmark.benchmark_type == "method":
                return self._run_method_benchmark(module, benchmark, parameters)
            else:
                raise ValueError(f"Unknown benchmark type: {benchmark.benchmark_type}")

        except Exception as e:
            error_type = self._categorize_error(e)
            logger.error(f"Error running benchmark {benchmark.name}: {error_type} - {e}")
            if error_type == "parameter_error":
                pass
            else:
                traceback.print_exc()

            return TraceResult(
                return_value=None,
                function_name=benchmark.name,
                module_name=benchmark.module_path,
                depth=0,
                success=False,
                error=e,
            )

    def run_all_benchmarks(
        self, filter_pattern: Optional[str] = None
    ) -> dict[str, list[TraceResult]]:
        """Run all discovered benchmarks; filter_pattern is a substring of the benchmark name."""

        benchmarks = self.discovery.discover_all()

        if filter_pattern:
            benchmarks = [b for b in benchmarks if filter_pattern in b.name]

        results = {}

        for benchmark in benchmarks:
            logger.info(f"Running benchmark: {benchmark.module_path}.{benchmark.name}")

            if benchmark.needs_runtime_eval:
                try:
                    module = self._load_module(benchmark.module_path)
                    runtime_params = self._evaluate_params_at_runtime(benchmark, module)
                    if runtime_params:
                        benchmark.params = runtime_params
                        benchmark.needs_runtime_eval = False
                    else:
                        logger.warning("  Failed to evaluate params at runtime, skipping")
                        continue
                except Exception as e:
                    logger.error(f"  Failed to load module for runtime evaluation: {e}")
                    continue

            if benchmark.params:
                param_combinations = self.get_param_combinations(benchmark)
                benchmark_results = []

                for params in param_combinations:
                    logger.info(f"  Parameters: {params}")
                    result = self.run_benchmark(benchmark, params)
                    if result:
                        benchmark_results.append(result)

                results[f"{benchmark.module_path}.{benchmark.name}"] = benchmark_results
            else:
                result = self.run_benchmark(benchmark)
                if result:
                    results[f"{benchmark.module_path}.{benchmark.name}"] = [result]

        return results

    def get_param_combinations(self, benchmark: BenchmarkInfo) -> list[tuple[Any, ...]]:
        """Return parameter combinations, evaluating runtime params if needed."""
        if benchmark.needs_runtime_eval:
            module = self._load_module(benchmark.module_path)
            runtime_params = self._evaluate_params_at_runtime(benchmark, module)
            if runtime_params:
                benchmark.params = runtime_params
                benchmark.needs_runtime_eval = False
        if not benchmark.params:
            return [()]
        import itertools

        return list(itertools.product(*benchmark.params))

    def _load_module(self, module_path: str) -> Any:
        """Load a benchmark module."""
        if module_path in self._module_cache:
            return self._module_cache[module_path]

        module_file = self.benchmark_dir / f"{module_path.replace('.', '/')}.py"

        if not module_file.exists():
            raise FileNotFoundError(f"Module file not found: {module_file}")

        # Ensure parent packages exist in sys.modules to support relative imports
        # in benchmark files (e.g., `from .pandas_vb_common import setup`).
        import types

        parts = module_path.split(".")

        for i in range(1, len(parts)):
            pkg_name = ".".join(parts[:i])
            if pkg_name not in sys.modules:
                pkg = types.ModuleType(pkg_name)
                pkg_dir = self.benchmark_dir / "/".join(parts[:i])
                if pkg_dir.exists() and pkg_dir.is_dir():
                    pkg.__path__ = [str(pkg_dir)]
                sys.modules[pkg_name] = pkg

        spec = importlib.util.spec_from_file_location(module_path, module_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module: {module_path}")

        module = importlib.util.module_from_spec(spec)

        # __package__ must be set for relative imports such as `from .utils import ...`
        if len(parts) > 1:
            module.__package__ = ".".join(parts[:-1])
        else:
            # A top-level module uses the name of benchmark_dir as its package, with or
            # without __init__.py, so relative imports work
            if (self.benchmark_dir / "__init__.py").exists():
                module.__package__ = self.benchmark_dir.name
            else:
                module.__package__ = self.benchmark_dir.name

        # Relative imports need the module in sys.modules before it runs
        sys.modules[module_path] = module

        if module.__package__ and module.__package__ not in sys.modules:
            pkg = types.ModuleType(module.__package__)
            if len(parts) > 1:
                pkg_dir = self.benchmark_dir / "/".join(parts[:-1])
            else:
                pkg_dir = self.benchmark_dir
            if pkg_dir.exists() and pkg_dir.is_dir():
                pkg.__path__ = [str(pkg_dir)]
            sys.modules[module.__package__] = pkg

        spec.loader.exec_module(module)

        self._module_cache[module_path] = module

        return module

    def _run_function_benchmark(
        self, module: Any, benchmark: BenchmarkInfo
    ) -> Optional[TraceResult]:
        """Run a function-level benchmark."""

        benchmark_func = getattr(module, benchmark.name, None)
        if benchmark_func is None:
            raise AttributeError(f"Benchmark function {benchmark.name} not found")

        self.tracer.start_tracing()

        try:
            # Rewrite a trailing expression statement into a return and use that value directly
            import ast
            import inspect

            try:
                source = inspect.getsource(benchmark_func)

                tree = ast.parse(source)

                func_def = tree.body[0]

                if func_def.body:
                    last_stmt = func_def.body[-1]

                    if isinstance(last_stmt, ast.Expr):
                        return_stmt = ast.Return(value=last_stmt.value)
                        func_def.body[-1] = return_stmt

                        ast.fix_missing_locations(tree)

                        modified_code = compile(tree, f"<modified_{benchmark.name}>", "exec")

                        namespace = module.__dict__.copy()
                        exec(modified_code, namespace)

                        modified_func = namespace[benchmark.name]

                        result_value = modified_func()

                        result = TraceResult(
                            return_value=result_value,
                            function_name=benchmark.name,
                            module_name=benchmark.module_path,
                            depth=0,
                            success=True,
                        )

                        self.tracer.stop_tracing()
                        return result

            except Exception:
                pass

            # The rewrite was not possible: run the original function under the tracer
            benchmark_func()

            result = self.tracer.stop_tracing()

            # Nothing captured (for example, a timing-only benchmark): success with no value
            if result is None:
                return TraceResult(
                    return_value=None,
                    function_name=benchmark.name,
                    module_name=benchmark.module_path,
                    depth=0,
                    success=True,
                    error=None,
                )

            return result

        except Exception as e:
            self.tracer.stop_tracing()
            logger.error(f"Benchmark {benchmark.name} failed: {e}")
            traceback.print_exc()
            return TraceResult(
                return_value=None,
                function_name=benchmark.name,
                module_name=benchmark.module_path,
                depth=0,
                success=False,
                error=e,
            )

    def _run_method_benchmark(
        self, module: Any, benchmark: BenchmarkInfo, parameters: Optional[tuple[Any, ...]] = None
    ) -> Optional[TraceResult]:
        """Run a method-level benchmark."""

        benchmark_class = getattr(module, benchmark.class_name, None)
        if benchmark_class is None:
            raise AttributeError(f"Benchmark class {benchmark.class_name} not found")

        benchmark_method = getattr(benchmark_class, benchmark.name, None)
        if benchmark_method is None:
            raise AttributeError(f"Benchmark method {benchmark.name} not found")

        # Rewrite a trailing expression into a return, so the value is captured even when the
        # inner calls run in C or Cython, which sys.settrace cannot see. On any error, skip it.
        try:
            import ast
            import inspect
            import textwrap

            try:
                src = inspect.getsource(benchmark_method)
            except (OSError, TypeError):
                src = None

            if src:
                dedented = textwrap.dedent(src)
                tree = ast.parse(dedented)
                if tree.body and isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_def = tree.body[0]
                    if func_def.body:
                        last_stmt = func_def.body[-1]
                        if isinstance(last_stmt, ast.Expr):
                            func_def.body[-1] = ast.Return(value=last_stmt.value)
                            ast.fix_missing_locations(tree)
                            code = compile(
                                tree,
                                f"<modified_{benchmark.class_name}.{benchmark.name}>",
                                "exec",
                            )
                            namespace = module.__dict__.copy()
                            exec(code, namespace)
                            new_func = namespace.get(benchmark.name)
                            if new_func is not None:
                                setattr(benchmark_class, benchmark.name, new_func)
                                benchmark_method = getattr(benchmark_class, benchmark.name)
        except Exception:
            pass

        instance = benchmark_class()

        # setup_cache runs once per class; its result is reused
        cached_state = None
        if benchmark.has_setup_cache:
            cache_key = f"{benchmark.module_path}.{benchmark.class_name}"
            if cache_key not in self._setup_cache:
                self._reset_random_state()
                setup_cache_method = getattr(instance, "setup_cache", None)
                if setup_cache_method:
                    try:
                        cached_state = setup_cache_method()
                        self._setup_cache[cache_key] = cached_state
                        logger.debug(f"Cached setup_cache result for {cache_key}")
                    except Exception as e:
                        logger.warning(f"setup_cache failed for {benchmark.class_name}: {e}")
                        traceback.print_exc()
            else:
                cached_state = self._setup_cache[cache_key]

        if benchmark.has_setup and benchmark.setup_method:
            self._reset_random_state()

            setup_method = getattr(instance, benchmark.setup_method, None)
            if setup_method:
                # setup takes the setup_cache result if its first parameter is "state" or "_state"
                import inspect

                setup_expects_state = False
                if cached_state is not None:
                    try:
                        sig = inspect.signature(setup_method)
                        params = list(sig.parameters.keys())
                        if params and params[0] == "self":
                            params = params[1:]
                        if params and (params[0] == "state" or params[0] == "_state"):
                            setup_expects_state = True
                    except (ValueError, TypeError):
                        pass

                if parameters:
                    if setup_expects_state and cached_state is not None:
                        try:
                            setup_method(cached_state, *parameters)
                        except NotImplementedError:
                            # ASV convention: NotImplementedError in setup skips this combination
                            logger.debug(
                                f"Setup raised NotImplementedError for parameters {parameters}, skipping"
                            )
                            return TraceResult(
                                return_value=None,
                                function_name=benchmark.name,
                                module_name=benchmark.module_path,
                                depth=0,
                                success=False,
                                error=NotImplementedError(
                                    f"Parameter combination {parameters} not supported"
                                ),
                            )
                        except TypeError as e:
                            try:
                                setup_method(*parameters)
                            except NotImplementedError:
                                logger.debug(
                                    f"Setup raised NotImplementedError for parameters {parameters}, skipping"
                                )
                                return TraceResult(
                                    return_value=None,
                                    function_name=benchmark.name,
                                    module_name=benchmark.module_path,
                                    depth=0,
                                    success=False,
                                    error=NotImplementedError(
                                        f"Parameter combination {parameters} not supported"
                                    ),
                                )
                            except TypeError:
                                logger.warning(
                                    f"Setup method failed with state and parameters: {e}"
                                )
                    elif benchmark.param_names:
                        # Positional first, then keyword arguments from param_names
                        try:
                            setup_method(*parameters)
                        except NotImplementedError:
                            logger.debug(
                                f"Setup raised NotImplementedError for parameters {parameters}, skipping"
                            )
                            return TraceResult(
                                return_value=None,
                                function_name=benchmark.name,
                                module_name=benchmark.module_path,
                                depth=0,
                                success=False,
                                error=NotImplementedError(
                                    f"Parameter combination {parameters} not supported"
                                ),
                            )
                        except TypeError as e:
                            try:
                                param_dict = dict(zip(benchmark.param_names, parameters))
                                setup_method(**param_dict)
                            except NotImplementedError:
                                logger.debug(
                                    f"Setup raised NotImplementedError for parameters {parameters}, skipping"
                                )
                                return TraceResult(
                                    return_value=None,
                                    function_name=benchmark.name,
                                    module_name=benchmark.module_path,
                                    depth=0,
                                    success=False,
                                    error=NotImplementedError(
                                        f"Parameter combination {parameters} not supported"
                                    ),
                                )
                            except TypeError:
                                logger.warning(
                                    f"Setup method failed with parameters {parameters}: {e}"
                                )
                                try:
                                    setup_method()
                                except TypeError:
                                    logger.warning("Setup method also failed without parameters")
                    else:
                        try:
                            setup_method(*parameters)
                        except NotImplementedError:
                            logger.debug(
                                f"Setup raised NotImplementedError for parameters {parameters}, skipping"
                            )
                            return TraceResult(
                                return_value=None,
                                function_name=benchmark.name,
                                module_name=benchmark.module_path,
                                depth=0,
                                success=False,
                                error=NotImplementedError(
                                    f"Parameter combination {parameters} not supported"
                                ),
                            )
                        except TypeError as e:
                            logger.warning(f"Setup method failed with parameters {parameters}: {e}")
                            try:
                                setup_method()
                            except TypeError:
                                logger.warning("Setup method also failed without parameters")
                else:
                    if setup_expects_state and cached_state is not None:
                        try:
                            setup_method(cached_state)
                        except NotImplementedError:
                            logger.debug("Setup raised NotImplementedError, skipping")
                            return TraceResult(
                                return_value=None,
                                function_name=benchmark.name,
                                module_name=benchmark.module_path,
                                depth=0,
                                success=False,
                                error=NotImplementedError("Parameter combination not supported"),
                            )
                        except TypeError as e:
                            logger.warning(f"Setup method failed with state: {e}")
                            try:
                                setup_method()
                            except TypeError:
                                logger.warning("Setup method also failed without state")
                    else:
                        try:
                            setup_method()
                        except NotImplementedError:
                            logger.debug("Setup raised NotImplementedError, skipping")
                            return TraceResult(
                                return_value=None,
                                function_name=benchmark.name,
                                module_name=benchmark.module_path,
                                depth=0,
                                success=False,
                                error=NotImplementedError("Parameter combination not supported"),
                            )
                        except TypeError as e:
                            logger.warning(f"Setup method failed: {e}")

        self.tracer.start_tracing()

        try:
            # The method takes the setup_cache result if its first parameter is "state" or "_state"
            import inspect

            method_expects_state = False
            if cached_state is not None:
                try:
                    sig = inspect.signature(benchmark_method)
                    params = list(sig.parameters.keys())
                    # Skip 'self' if present
                    if params and params[0] == "self":
                        params = params[1:]
                    # Check if first parameter is 'state' or '_state'
                    if params and (params[0] == "state" or params[0] == "_state"):
                        method_expects_state = True
                except (ValueError, TypeError):
                    # Fall back to the parameter names found by discovery
                    method_expects_state = (
                        benchmark.method_params
                        and len(benchmark.method_params) > 0
                        and (
                            benchmark.method_params[0] == "state"
                            or benchmark.method_params[0] == "_state"
                        )
                    )

            # Parameters are passed when the method declares parameters or the class has no
            # param_names; otherwise only setup receives them.
            if method_expects_state and cached_state is not None:
                if parameters:
                    benchmark_method(instance, cached_state, *parameters)
                else:
                    benchmark_method(instance, cached_state)
            elif parameters and (
                (benchmark.method_params and len(benchmark.method_params) > 0)
                or (not benchmark.param_names)
            ):
                benchmark_method(instance, *parameters)
            else:
                benchmark_method(instance)

            result = self.tracer.stop_tracing()

            # Nothing captured (for example, a timing-only benchmark): success with no value
            if result is None:
                return TraceResult(
                    return_value=None,
                    function_name=benchmark.name,
                    module_name=benchmark.module_path,
                    depth=0,
                    success=True,
                    error=None,
                )

            return result

        except Exception as e:
            self.tracer.stop_tracing()
            logger.error(f"Benchmark {benchmark.name} failed: {e}")
            traceback.print_exc()
            return TraceResult(
                return_value=None,
                function_name=benchmark.name,
                module_name=benchmark.module_path,
                depth=0,
                success=False,
                error=e,
            )

    def get_benchmark_info(self, benchmark_name: str) -> Optional[BenchmarkInfo]:
        """Get information about a specific benchmark."""
        return self.discovery.get_benchmark_by_name(benchmark_name)

    def list_benchmarks(self) -> list[BenchmarkInfo]:
        """List all discovered benchmarks."""
        return self.discovery.discover_all()

    def get_module_benchmarks(self, module_path: str) -> list[BenchmarkInfo]:
        """Get all benchmarks from a specific module."""
        return self.discovery.get_benchmarks_by_module(module_path)

    def clear_cache(self) -> None:
        """Clear the module cache."""
        self._module_cache.clear()

    def get_trace_stats(self) -> dict[str, Any]:
        """Get tracing statistics."""
        return self.tracer.get_trace_stats()

    def _categorize_error(self, error: Exception) -> str:
        """Return a short error category for logging."""
        error_str = str(error).lower()

        if isinstance(error, TimeoutError):
            return "timeout_error"
        elif "missing" in error_str and "required positional argument" in error_str:
            return "parameter_error"
        elif "no module named" in error_str:
            return "missing_dependency"
        elif "no such file or directory" in error_str:
            return "missing_file"
        elif "attributeerror" in error_str.lower():
            return "attribute_error"
        else:
            return "unknown_error"


# Parallel execution (CLI --parallel). A process pool gives each worker its own sys.settrace,
# RNG and import state; the main process does every SQLite write (SQLite has one writer).


def default_worker_count() -> int:
    """Return min(cpu_count, 8)."""
    return min(os.cpu_count() or 1, 8)


@dataclass
class TaskResult:
    """Picklable outcome of running one (benchmark, params) in a worker."""

    ok: bool
    serialized_value: Any = None  # serialize_value(return_value); always picklable
    failure_reason: Optional[str] = None


def _result_to_task_result(result: Optional[TraceResult]) -> TaskResult:
    from .storage import serialize_value

    if result is None:
        return TaskResult(ok=False, failure_reason="Unknown error (no result)")
    if not result.success:
        err = result.error
        if err is not None:
            etype = type(err).__name__
            emsg = str(err)
            reason = f"{etype}: {emsg}" if emsg else etype
        else:
            reason = "Unknown error (no exception details)"
        return TaskResult(ok=False, failure_reason=reason)
    return TaskResult(ok=True, serialized_value=serialize_value(result.return_value))


def _execute_one(
    runner: BenchmarkRunner,
    benchmark: BenchmarkInfo,
    parameters: Optional[tuple],
) -> TaskResult:
    try:
        result = runner.run_benchmark(benchmark, parameters)
    except Exception as e:  # run_benchmark catches most errors itself
        return TaskResult(ok=False, failure_reason=f"{type(e).__name__}: {e}")
    return _result_to_task_result(result)


# Built once per worker by the pool initializer, so its module and setup_cache caches are reused
_WORKER_RUNNER: Optional[BenchmarkRunner] = None


def _worker_init(
    benchmark_dir: str, project_dir: Optional[str], seed: int, timeout: Optional[float]
) -> None:
    global _WORKER_RUNNER
    _WORKER_RUNNER = BenchmarkRunner(
        Path(benchmark_dir),
        project_dir=Path(project_dir) if project_dir else None,
        seed=seed,
        timeout=timeout,
    )


def _worker_run(benchmark: BenchmarkInfo, parameters: Optional[tuple]) -> TaskResult:
    assert _WORKER_RUNNER is not None, "worker not initialized"
    return _execute_one(_WORKER_RUNNER, benchmark, parameters)


def iter_task_results(
    benchmark_dir: Path,
    project_dir: Optional[Path],
    seed: int,
    timeout: Optional[float],
    tasks: list,  # list of (idx, BenchmarkInfo, params_tuple_or_None)
    max_workers: int,
):
    """Yield ``(idx, TaskResult)`` for every task.

    Runs in the main process when ``max_workers`` <= 1 or there is one task, and for tasks
    whose ``(benchmark, params)`` cannot be pickled.
    """
    if max_workers <= 1 or len(tasks) <= 1:
        runner = BenchmarkRunner(benchmark_dir, project_dir=project_dir, seed=seed, timeout=timeout)
        for idx, bench, params in tasks:
            yield idx, _execute_one(runner, bench, params)
        return

    picklable, inline = [], []
    for idx, bench, params in tasks:
        try:
            pickle.dumps((bench, params))
            picklable.append((idx, bench, params))
        except Exception:
            inline.append((idx, bench, params))

    if inline:
        fallback = BenchmarkRunner(
            benchmark_dir, project_dir=project_dir, seed=seed, timeout=timeout
        )
        for idx, bench, params in inline:
            yield idx, _execute_one(fallback, bench, params)

    if not picklable:
        return

    # One dead worker (killed for memory, or a crash in native code) breaks the whole pool.
    # Then the tasks that did not finish run again serially in the main process.
    done_idx = set()
    pool_broken = False
    fut_to_task = {}

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_worker_init,
        initargs=(
            str(benchmark_dir),
            str(project_dir) if project_dir else None,
            seed,
            timeout,
        ),
    ) as ex:
        fut_to_task = {
            ex.submit(_worker_run, bench, params): (idx, bench, params)
            for idx, bench, params in picklable
        }
        for fut in as_completed(fut_to_task):
            idx, _, _ = fut_to_task[fut]
            try:
                result = fut.result()
            except BrokenProcessPool:
                pool_broken = True
                break
            except Exception as e:
                done_idx.add(idx)
                yield (
                    idx,
                    TaskResult(
                        ok=False,
                        failure_reason=f"worker crashed: {type(e).__name__}: {e}",
                    ),
                )
                continue
            done_idx.add(idx)
            yield idx, result

    if pool_broken:
        remaining = sorted(
            (
                (idx, bench, params)
                for idx, bench, params in fut_to_task.values()
                if idx not in done_idx
            ),
            key=lambda t: t[0],
        )
        logger.warning(
            "Worker pool broke (likely an OOM-killed benchmark); falling back to "
            f"in-process serial execution for the remaining {len(remaining)} task(s)"
        )
        serial_runner = BenchmarkRunner(
            benchmark_dir, project_dir=project_dir, seed=seed, timeout=timeout
        )
        for idx, bench, params in remaining:
            yield idx, _execute_one(serial_runner, bench, params)
