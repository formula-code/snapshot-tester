"""
Benchmark runner for executing ASV benchmarks with tracing.

This module executes benchmarks with tracing enabled and handles setup methods,
parameter combinations, and global variable initialization.
"""
from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any, Optional

from .discovery import BenchmarkDiscovery, BenchmarkInfo
from .rng_patcher import RNGPatcher
from .tracer import ExecutionTracer, TraceResult

logger = logging.getLogger(__name__)


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
        self.seed = seed  # Deterministic seed for reproducibility
        self.timeout = timeout  # Maximum execution time in seconds

        # Initialize RNG patcher for deterministic execution
        self.rng_patcher = RNGPatcher(seed=seed)

        # Cache for loaded modules
        self._module_cache: dict[str, Any] = {}
        
        # Cache for setup_cache results (per class instance)
        self._setup_cache: dict[str, Any] = {}

        # Add project directory to Python path for imports
        if str(self.project_dir) not in sys.path:
            sys.path.insert(0, str(self.project_dir))

    def _reset_random_state(self):
        """
        Reset random state to ensure deterministic benchmark execution.

        This patches ALL random number generators (numpy legacy, numpy Generator API,
        PyTorch, TensorFlow) to use deterministic seeds.
        """
        # Use the RNG patcher to ensure all RNGs are deterministic
        # This handles both legacy and modern numpy RNG, plus PyTorch and TensorFlow
        self.rng_patcher.patch_all()

    def _evaluate_params_at_runtime(self, benchmark: BenchmarkInfo, module) -> list[list[Any]]:
        """Evaluate params at runtime by accessing the class attribute."""
        if not benchmark.needs_runtime_eval or not benchmark.class_name:
            return benchmark.params

        try:
            # Get the class from the module
            benchmark_class = getattr(module, benchmark.class_name)
            # Access the params attribute
            params = getattr(benchmark_class, "params", None)
            if params is None:
                return None

            # Convert to the expected format
            if isinstance(params, (list, tuple)):
                if len(params) > 0 and isinstance(params[0], (list, tuple)):
                    # Already in the correct format - convert to list of lists
                    return [list(p) if isinstance(p, tuple) else p for p in params]
                else:
                    # Single parameter list
                    return [list(params) if isinstance(params, tuple) else params]
            else:
                # Single value
                return [[params]]

        except Exception as e:
            logger.warning(f"Failed to evaluate params at runtime for {benchmark.name}: {e}")
            return None

    def _run_with_timeout(
        self,
        benchmark: BenchmarkInfo,
        parameters: Optional[tuple[Any, ...]] = None,
    ) -> Optional[TraceResult]:
        """
        Execute benchmark with timeout.

        Returns None if timeout is exceeded, otherwise returns TraceResult.
        """
        if self.timeout is None:
            # No timeout configured, run directly
            return self._run_benchmark_internal(benchmark, parameters)

        # Use ThreadPoolExecutor to run with timeout
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self._run_benchmark_internal, benchmark, parameters)
        try:
            result = future.result(timeout=self.timeout)
            executor.shutdown(wait=True)
            return result
        except FuturesTimeoutError:
            logger.warning(
                f"Benchmark {benchmark.name} timed out after {self.timeout} seconds"
            )
            # Cancel the future and shutdown without waiting
            future.cancel()
            executor.shutdown(wait=False)
            # Return a failed trace result for timeout
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

    def run_benchmark(
        self, benchmark: BenchmarkInfo, parameters: Optional[tuple[Any, ...]] = None
    ) -> Optional[TraceResult]:
        """Run a single benchmark with tracing and optional timeout."""
        return self._run_with_timeout(benchmark, parameters)

    def _run_benchmark_internal(
        self, benchmark: BenchmarkInfo, parameters: Optional[tuple[Any, ...]] = None
    ) -> Optional[TraceResult]:
        """Run a single benchmark with tracing (internal implementation without timeout wrapper)."""

        try:
            # Reset random state before each benchmark run for determinism
            self._reset_random_state()

            # Load the benchmark module
            module = self._load_module(benchmark.module_path)

            # Evaluate params at runtime if needed
            if benchmark.needs_runtime_eval:
                runtime_params = self._evaluate_params_at_runtime(benchmark, module)
                if runtime_params:
                    benchmark.params = runtime_params
                    benchmark.needs_runtime_eval = False  # Mark as evaluated

            # Handle parameterized benchmarks called without parameters
            # If benchmark has params but none provided, use first param combination
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
                # Don't print full traceback for parameter errors
                pass
            else:
                traceback.print_exc()

            # Return a failed trace result instead of None
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
        """Run all discovered benchmarks."""

        # Discover benchmarks
        benchmarks = self.discovery.discover_all()

        if filter_pattern:
            benchmarks = [b for b in benchmarks if filter_pattern in b.name]

        results = {}

        for benchmark in benchmarks:
            logger.info(f"Running benchmark: {benchmark.module_path}.{benchmark.name}")

            # Handle runtime evaluation of params
            if benchmark.needs_runtime_eval:
                # Load module to evaluate params
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
                # Run with all parameter combinations
                param_combinations = self.get_param_combinations(benchmark)
                benchmark_results = []

                for params in param_combinations:
                    logger.info(f"  Parameters: {params}")
                    result = self.run_benchmark(benchmark, params)
                    if result:
                        benchmark_results.append(result)

                results[f"{benchmark.module_path}.{benchmark.name}"] = benchmark_results
            else:
                # Run without parameters
                result = self.run_benchmark(benchmark)
                if result:
                    results[f"{benchmark.module_path}.{benchmark.name}"] = [result]

        return results

    def get_param_combinations(self, benchmark: BenchmarkInfo) -> list[tuple[Any, ...]]:
        """Return parameter combinations, evaluating runtime params if needed."""
        # Ensure params are evaluated if needed
        if benchmark.needs_runtime_eval:
            module = self._load_module(benchmark.module_path)
            runtime_params = self._evaluate_params_at_runtime(benchmark, module)
            if runtime_params:
                benchmark.params = runtime_params
                benchmark.needs_runtime_eval = False
        # Fallback if still no params
        if not benchmark.params:
            return [()]
        # Build Cartesian product
        import itertools

        return list(itertools.product(*benchmark.params))

    def _load_module(self, module_path: str) -> Any:
        """Load a benchmark module."""
        if module_path in self._module_cache:
            return self._module_cache[module_path]

        # Find the module file
        module_file = self.benchmark_dir / f"{module_path.replace('.', '/')}.py"

        if not module_file.exists():
            raise FileNotFoundError(f"Module file not found: {module_file}")

        # Ensure parent packages exist in sys.modules to support relative imports
        # in benchmark files (e.g., `from .pandas_vb_common import setup`).
        import types
        parts = module_path.split('.')
        
        # Create parent packages if they don't exist
        for i in range(1, len(parts)):
            pkg_name = '.'.join(parts[:i])
            if pkg_name not in sys.modules:
                pkg = types.ModuleType(pkg_name)
                # Mark as a package by setting __path__ to the directory
                pkg_dir = self.benchmark_dir / '/'.join(parts[:i])
                if pkg_dir.exists() and pkg_dir.is_dir():
                    pkg.__path__ = [str(pkg_dir)]
                sys.modules[pkg_name] = pkg

        # Load the module
        spec = importlib.util.spec_from_file_location(module_path, module_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module: {module_path}")

        module = importlib.util.module_from_spec(spec)
        
        # Set __package__ attribute for relative imports to work
        # This is critical for modules using relative imports like `from .utils import ...`
        if len(parts) > 1:
            # Module is in a subpackage - parent is the package
            module.__package__ = '.'.join(parts[:-1])
        else:
            # Module is at root of benchmark_dir
            # If benchmark_dir has __init__.py, it's a package
            # Modules in it need __package__ set to allow relative imports
            # Use benchmark_dir name as package name
            if (self.benchmark_dir / "__init__.py").exists():
                # benchmark_dir is a package - use its name as package name
                module.__package__ = self.benchmark_dir.name
            else:
                # No __init__.py, but modules might still use relative imports
                # Create a synthetic package based on directory name
                module.__package__ = self.benchmark_dir.name
        
        # Add module to sys.modules before execution (required for relative imports)
        sys.modules[module_path] = module
        
        # Ensure the package exists in sys.modules for relative imports to work
        if module.__package__ and module.__package__ not in sys.modules:
            pkg = types.ModuleType(module.__package__)
            # Set __path__ to benchmark_dir or the appropriate parent directory
            if len(parts) > 1:
                pkg_dir = self.benchmark_dir / '/'.join(parts[:-1])
            else:
                pkg_dir = self.benchmark_dir
            if pkg_dir.exists() and pkg_dir.is_dir():
                pkg.__path__ = [str(pkg_dir)]
            sys.modules[module.__package__] = pkg
        
        spec.loader.exec_module(module)

        # Cache the module
        self._module_cache[module_path] = module

        return module

    def _run_function_benchmark(
        self, module: Any, benchmark: BenchmarkInfo
    ) -> Optional[TraceResult]:
        """Run a function-level benchmark."""

        # Get the benchmark function
        benchmark_func = getattr(module, benchmark.name, None)
        if benchmark_func is None:
            raise AttributeError(f"Benchmark function {benchmark.name} not found")

        # Start tracing
        self.tracer.start_tracing()

        try:
            # For benchmarks that don't return values, we need to capture the result
            # of the deepest function call. We'll use AST manipulation to modify
            # the benchmark function to return its result.
            import ast
            import inspect

            # Get the source code of the benchmark function
            try:
                source = inspect.getsource(benchmark_func)

                # Parse the AST
                tree = ast.parse(source)

                # Find the function definition
                func_def = tree.body[0]

                # Find the last statement in the function body
                if func_def.body:
                    last_stmt = func_def.body[-1]

                    # If the last statement is an expression (like fitting.LevMarLSQFitter()),
                    # modify it to return the result
                    if isinstance(last_stmt, ast.Expr):
                        # Create a return statement with the same expression
                        return_stmt = ast.Return(value=last_stmt.value)
                        func_def.body[-1] = return_stmt

                        # Fix missing line numbers for the new AST nodes
                        ast.fix_missing_locations(tree)

                        # Compile the modified function
                        modified_code = compile(tree, f"<modified_{benchmark.name}>", "exec")

                        # Execute in the module's namespace
                        namespace = module.__dict__.copy()
                        exec(modified_code, namespace)

                        # Get the modified function
                        modified_func = namespace[benchmark.name]

                        # Execute the modified function
                        result_value = modified_func()

                        # Create a TraceResult with the captured value
                        result = TraceResult(
                            return_value=result_value,
                            function_name=benchmark.name,
                            module_name=benchmark.module_path,
                            depth=0,
                            success=True,
                        )

                        # Stop tracing
                        self.tracer.stop_tracing()
                        return result

            except Exception:
                # Fall back to original execution
                pass

            # Fallback: execute the original function
            benchmark_func()

            # Stop tracing and get result
            result = self.tracer.stop_tracing()

            # If tracer didn't capture anything, return success with no data
            # (e.g., timing-only benchmarks, state modification benchmarks)
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
            # Stop tracing even if benchmark failed
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

        # Get the benchmark class
        benchmark_class = getattr(module, benchmark.class_name, None)
        if benchmark_class is None:
            raise AttributeError(f"Benchmark class {benchmark.class_name} not found")

        # Get the benchmark method
        benchmark_method = getattr(benchmark_class, benchmark.name, None)
        if benchmark_method is None:
            raise AttributeError(f"Benchmark method {benchmark.name} not found")

        # Attempt to rewrite the benchmark method so that its final expression
        # becomes an explicit return statement. This lets us capture the value
        # even if inner calls execute in C/Cython (unseen by sys.settrace).
        # Falls back silently if rewriting is not possible.
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
                    # Skip empty bodies
                    if func_def.body:
                        last_stmt = func_def.body[-1]
                        if isinstance(last_stmt, ast.Expr):
                            # Replace tail expression with return of that expression
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
                                # Refresh local reference to the possibly replaced method
                                benchmark_method = getattr(benchmark_class, benchmark.name)
        except Exception:
            # If anything goes wrong, ignore and continue with original method.
            pass

        # Instantiate the class
        instance = benchmark_class()
        
        # Run setup_cache if it exists (once per class, cached)
        cached_state = None
        if benchmark.has_setup_cache:
            cache_key = f"{benchmark.module_path}.{benchmark.class_name}"
            if cache_key not in self._setup_cache:
                # Reset random state before setup_cache to ensure determinism
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

        # Run setup if it exists
        if benchmark.has_setup and benchmark.setup_method:
            # Reset random state before setup to ensure determinism
            self._reset_random_state()

            setup_method = getattr(instance, benchmark.setup_method, None)
            if setup_method:
                # Check if setup expects cached state as first parameter
                # Inspect the setup method's signature
                import inspect
                setup_expects_state = False
                if cached_state is not None:
                    try:
                        sig = inspect.signature(setup_method)
                        params = list(sig.parameters.keys())
                        # Skip 'self' if present
                        if params and params[0] == 'self':
                            params = params[1:]
                        # Check if first parameter is 'state' or '_state'
                        if params and (params[0] == 'state' or params[0] == '_state'):
                            setup_expects_state = True
                    except (ValueError, TypeError):
                        # If signature inspection fails, try heuristic
                        # If we have cached_state and setup has parameters, assume it expects state
                        pass
                
                if parameters:
                    # Call setup with parameters
                    if setup_expects_state and cached_state is not None:
                        # Pass cached state as first argument, then parameters
                        try:
                            setup_method(cached_state, *parameters)
                        except NotImplementedError:
                            # Setup explicitly indicates this parameter combination is not supported
                            # Skip this benchmark run
                            logger.debug(f"Setup raised NotImplementedError for parameters {parameters}, skipping")
                            return TraceResult(
                                return_value=None,
                                function_name=benchmark.name,
                                module_name=benchmark.module_path,
                                depth=0,
                                success=False,
                                error=NotImplementedError(f"Parameter combination {parameters} not supported"),
                            )
                        except TypeError as e:
                            # Try without state if that fails
                            try:
                                setup_method(*parameters)
                            except NotImplementedError:
                                logger.debug(f"Setup raised NotImplementedError for parameters {parameters}, skipping")
                                return TraceResult(
                                    return_value=None,
                                    function_name=benchmark.name,
                                    module_name=benchmark.module_path,
                                    depth=0,
                                    success=False,
                                    error=NotImplementedError(f"Parameter combination {parameters} not supported"),
                                )
                            except TypeError:
                                logger.warning(f"Setup method failed with state and parameters: {e}")
                    elif benchmark.param_names:
                        # Use param_names for keyword arguments
                        try:
                            setup_method(*parameters)
                        except NotImplementedError:
                            logger.debug(f"Setup raised NotImplementedError for parameters {parameters}, skipping")
                            return TraceResult(
                                return_value=None,
                                function_name=benchmark.name,
                                module_name=benchmark.module_path,
                                depth=0,
                                success=False,
                                error=NotImplementedError(f"Parameter combination {parameters} not supported"),
                            )
                        except TypeError as e:
                            try:
                                param_dict = dict(zip(benchmark.param_names, parameters))
                                setup_method(**param_dict)
                            except NotImplementedError:
                                logger.debug(f"Setup raised NotImplementedError for parameters {parameters}, skipping")
                                return TraceResult(
                                    return_value=None,
                                    function_name=benchmark.name,
                                    module_name=benchmark.module_path,
                                    depth=0,
                                    success=False,
                                    error=NotImplementedError(f"Parameter combination {parameters} not supported"),
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
                        # No param_names - pass parameters as positional arguments
                        try:
                            setup_method(*parameters)
                        except NotImplementedError:
                            logger.debug(f"Setup raised NotImplementedError for parameters {parameters}, skipping")
                            return TraceResult(
                                return_value=None,
                                function_name=benchmark.name,
                                module_name=benchmark.module_path,
                                depth=0,
                                success=False,
                                error=NotImplementedError(f"Parameter combination {parameters} not supported"),
                            )
                        except TypeError as e:
                            logger.warning(f"Setup method failed with parameters {parameters}: {e}")
                            try:
                                setup_method()
                            except TypeError:
                                logger.warning("Setup method also failed without parameters")
                else:
                    # Call setup without parameters (or with state if it expects it)
                    if setup_expects_state and cached_state is not None:
                        try:
                            setup_method(cached_state)
                        except NotImplementedError:
                            logger.debug(f"Setup raised NotImplementedError, skipping")
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
                            logger.debug(f"Setup raised NotImplementedError, skipping")
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

        # Start tracing
        self.tracer.start_tracing()

        try:
            # Check if benchmark method expects cached state as first parameter
            # Inspect the benchmark method's signature
            import inspect
            method_expects_state = False
            if cached_state is not None:
                try:
                    sig = inspect.signature(benchmark_method)
                    params = list(sig.parameters.keys())
                    # Skip 'self' if present
                    if params and params[0] == 'self':
                        params = params[1:]
                    # Check if first parameter is 'state' or '_state'
                    if params and (params[0] == 'state' or params[0] == '_state'):
                        method_expects_state = True
                except (ValueError, TypeError):
                    # If signature inspection fails, fall back to method_params check
                    method_expects_state = (
                        benchmark.method_params 
                        and len(benchmark.method_params) > 0 
                        and (benchmark.method_params[0] == "state" or benchmark.method_params[0] == "_state")
                    )
            
            # Execute the benchmark method with parameters
            # Rule:
            # - If method expects state and we have cached_state, pass state first
            # - If method defines its own parameters (method_params), pass them regardless of param_names
            # - Else if there are no param_names (unnamed param pattern), pass them
            # - Otherwise, call without parameters (setup-only pattern)
            if method_expects_state and cached_state is not None:
                # Pass cached state as first argument, then parameters
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

            # Stop tracing and get result
            result = self.tracer.stop_tracing()

            # If tracer didn't capture anything, return success with no data
            # (e.g., timing-only benchmarks, state modification benchmarks)
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
            # Stop tracing even if benchmark failed
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
        """Categorize errors for better handling."""
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
