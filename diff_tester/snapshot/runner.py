"""
Benchmark runner for executing ASV benchmarks with tracing.

This module executes benchmarks with tracing enabled and handles setup methods,
parameter combinations, and global variable initialization.
"""

import sys
import importlib.util
import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import traceback

from .discovery import BenchmarkInfo, BenchmarkDiscovery
from .tracer import ExecutionTracer, TraceResult


class BenchmarkRunner:
    """Executes ASV benchmarks with tracing enabled."""
    
    def __init__(self, benchmark_dir: Path, project_dir: Optional[Path] = None):
        self.benchmark_dir = Path(benchmark_dir)
        self.project_dir = project_dir or benchmark_dir.parent
        self.discovery = BenchmarkDiscovery(self.benchmark_dir)
        self.tracer = ExecutionTracer()
        
        # Cache for loaded modules
        self._module_cache: Dict[str, Any] = {}
        
        # Add project directory to Python path for imports
        if str(self.project_dir) not in sys.path:
            sys.path.insert(0, str(self.project_dir))
    
    def _evaluate_params_at_runtime(self, benchmark: BenchmarkInfo, module) -> List[List[Any]]:
        """Evaluate params at runtime by accessing the class attribute."""
        if not benchmark.needs_runtime_eval or not benchmark.class_name:
            return benchmark.params
        
        try:
            # Get the class from the module
            benchmark_class = getattr(module, benchmark.class_name)
            # Access the params attribute
            params = getattr(benchmark_class, 'params', None)
            if params is None:
                return None
            
            # Convert to the expected format
            if isinstance(params, list):
                if len(params) > 0 and isinstance(params[0], list):
                    # Already in the correct format
                    return params
                else:
                    # Single parameter list
                    return [params]
            else:
                # Single value
                return [[params]]
                
        except Exception as e:
            print(f"Warning: Failed to evaluate params at runtime for {benchmark.name}: {e}")
            return None
    
    def run_benchmark(self, 
                     benchmark: BenchmarkInfo,
                     parameters: Optional[Tuple[Any, ...]] = None) -> Optional[TraceResult]:
        """Run a single benchmark with tracing."""
        
        try:
            # Load the benchmark module
            module = self._load_module(benchmark.module_path)
            
            # Evaluate params at runtime if needed
            if benchmark.needs_runtime_eval:
                runtime_params = self._evaluate_params_at_runtime(benchmark, module)
                if runtime_params:
                    benchmark.params = runtime_params
                    benchmark.needs_runtime_eval = False  # Mark as evaluated
            
            if benchmark.benchmark_type == 'function':
                return self._run_function_benchmark(module, benchmark)
            elif benchmark.benchmark_type == 'method':
                return self._run_method_benchmark(module, benchmark, parameters)
            else:
                raise ValueError(f"Unknown benchmark type: {benchmark.benchmark_type}")
                
        except Exception as e:
            error_type = self._categorize_error(e)
            print(f"Error running benchmark {benchmark.name}: {error_type} - {e}")
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
                error=e
            )
    
    def run_all_benchmarks(self, 
                          filter_pattern: Optional[str] = None) -> Dict[str, List[TraceResult]]:
        """Run all discovered benchmarks."""
        
        # Discover benchmarks
        benchmarks = self.discovery.discover_all()
        
        if filter_pattern:
            benchmarks = [b for b in benchmarks if filter_pattern in b.name]
        
        results = {}
        
        for benchmark in benchmarks:
            print(f"Running benchmark: {benchmark.module_path}.{benchmark.name}")
            
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
                        print(f"  Failed to evaluate params at runtime, skipping")
                        continue
                except Exception as e:
                    print(f"  Failed to load module for runtime evaluation: {e}")
                    continue
            
            if benchmark.params:
                # Run with all parameter combinations
                param_combinations = self.get_param_combinations(benchmark)
                benchmark_results = []
                
                for params in param_combinations:
                    print(f"  Parameters: {params}")
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

    def get_param_combinations(self, benchmark: BenchmarkInfo) -> List[Tuple[Any, ...]]:
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
        
        # Load the module
        spec = importlib.util.spec_from_file_location(module_path, module_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module: {module_path}")
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Cache the module
        self._module_cache[module_path] = module
        
        return module
    
    def _run_function_benchmark(self, module: Any, benchmark: BenchmarkInfo) -> Optional[TraceResult]:
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
                            module_path=benchmark.module_path,
                            depth=0,
                            success=True
                        )
                        
                        # Stop tracing
                        self.tracer.stop_tracing()
                        return result
                        
            except Exception as ast_error:
                # Fall back to original execution
                pass
            
            # Fallback: execute the original function
            benchmark_func()
            
            # Stop tracing and get result
            result = self.tracer.stop_tracing()
            
            return result
            
        except Exception as e:
            # Stop tracing even if benchmark failed
            self.tracer.stop_tracing()
            print(f"Benchmark {benchmark.name} failed: {e}")
            traceback.print_exc()
            return TraceResult(
                return_value=None,
                function_name=benchmark.name,
                module_name=benchmark.module_path,
                depth=0,
                success=False,
                error=e
            )
    
    def _run_method_benchmark(self, 
                            module: Any, 
                            benchmark: BenchmarkInfo,
                            parameters: Optional[Tuple[Any, ...]] = None) -> Optional[TraceResult]:
        """Run a method-level benchmark."""
        
        # Get the benchmark class
        benchmark_class = getattr(module, benchmark.class_name, None)
        if benchmark_class is None:
            raise AttributeError(f"Benchmark class {benchmark.class_name} not found")
        
        # Get the benchmark method
        benchmark_method = getattr(benchmark_class, benchmark.name, None)
        if benchmark_method is None:
            raise AttributeError(f"Benchmark method {benchmark.name} not found")
        
        # Instantiate the class
        instance = benchmark_class()
        
        # Run setup if it exists
        if benchmark.has_setup and benchmark.setup_method:
            setup_method = getattr(instance, benchmark.setup_method, None)
            if setup_method:
                if parameters:
                    # Call setup with parameters
                    if benchmark.param_names:
                        # Use param_names for keyword arguments
                        try:
                            setup_method(*parameters)
                        except TypeError as e:
                            try:
                                param_dict = dict(zip(benchmark.param_names, parameters))
                                setup_method(**param_dict)
                            except TypeError:
                                print(f"Warning: Setup method failed with parameters {parameters}: {e}")
                                try:
                                    setup_method()
                                except TypeError:
                                    print(f"Warning: Setup method also failed without parameters")
                    else:
                        # No param_names - pass parameters as positional arguments
                        try:
                            setup_method(*parameters)
                        except TypeError as e:
                            print(f"Warning: Setup method failed with parameters {parameters}: {e}")
                            try:
                                setup_method()
                            except TypeError:
                                print(f"Warning: Setup method also failed without parameters")
                else:
                    # Call setup without parameters
                    try:
                        setup_method()
                    except TypeError as e:
                        print(f"Warning: Setup method failed: {e}")
        
        # Start tracing
        self.tracer.start_tracing()
        
        try:
            # Execute the benchmark method with parameters
            # Rule:
            # - If method defines its own parameters (method_params), pass them regardless of param_names
            # - Else if there are no param_names (unnamed param pattern), pass them
            # - Otherwise, call without parameters (setup-only pattern)
            if parameters and (
                (benchmark.method_params and len(benchmark.method_params) > 0)
                or (not benchmark.param_names)
            ):
                benchmark_method(instance, *parameters)
            else:
                benchmark_method(instance)
            
            # Stop tracing and get result
            result = self.tracer.stop_tracing()
            return result
            
        except Exception as e:
            # Stop tracing even if benchmark failed
            self.tracer.stop_tracing()
            print(f"Benchmark {benchmark.name} failed: {e}")
            traceback.print_exc()
            return TraceResult(
                return_value=None,
                function_name=benchmark.name,
                module_name=benchmark.module_path,
                depth=0,
                success=False,
                error=e
            )
    
    def get_benchmark_info(self, benchmark_name: str) -> Optional[BenchmarkInfo]:
        """Get information about a specific benchmark."""
        return self.discovery.get_benchmark_by_name(benchmark_name)
    
    def list_benchmarks(self) -> List[BenchmarkInfo]:
        """List all discovered benchmarks."""
        return self.discovery.discover_all()
    
    def get_module_benchmarks(self, module_path: str) -> List[BenchmarkInfo]:
        """Get all benchmarks from a specific module."""
        return self.discovery.get_benchmarks_by_module(module_path)
    
    def clear_cache(self) -> None:
        """Clear the module cache."""
        self._module_cache.clear()
    
    def get_trace_stats(self) -> Dict[str, Any]:
        """Get tracing statistics."""
        return self.tracer.get_trace_stats()
    
    def _categorize_error(self, error: Exception) -> str:
        """Categorize errors for better handling."""
        error_str = str(error).lower()
        
        if "missing" in error_str and "required positional argument" in error_str:
            return "parameter_error"
        elif "no module named" in error_str:
            return "missing_dependency"
        elif "no such file or directory" in error_str:
            return "missing_file"
        elif "attributeerror" in error_str.lower():
            return "attribute_error"
        else:
            return "unknown_error"
