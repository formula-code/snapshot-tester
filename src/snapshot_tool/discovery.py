"""
Benchmark discovery engine for ASV benchmarks.

This module parses ASV benchmark files to discover benchmark classes,
functions, parameters, and setup methods.
"""
from __future__ import annotations

import ast
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkInfo:
    """Information about a discovered benchmark."""

    name: str
    module_path: str
    benchmark_type: str  # 'function' or 'method'
    class_name: Optional[str] = None
    params: Optional[list[list[Any]]] = None
    param_names: Optional[list[str]] = None
    setup_method: Optional[str] = None
    has_setup: bool = False
    method_params: Optional[list[str]] = None  # Parameters expected by the method itself
    needs_runtime_eval: bool = False  # Whether params need runtime evaluation


class BenchmarkDiscovery:
    """Discovers ASV benchmarks in a directory tree."""

    # ASV benchmark prefixes
    BENCHMARK_PREFIXES = {"time_", "timeraw_", "mem_", "peakmem_", "track_"}

    def __init__(self, benchmark_dir: str | Path):
        self.benchmark_dir = Path(benchmark_dir)
        self.discovered_benchmarks: list[BenchmarkInfo] = []

    def discover_all(self) -> list[BenchmarkInfo]:
        """Discover all benchmarks in the benchmark directory."""
        self.discovered_benchmarks = []

        # Walk through all Python files in the benchmark directory
        for py_file in self.benchmark_dir.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue

            try:
                benchmarks = self._discover_in_file(py_file)
                self.discovered_benchmarks.extend(benchmarks)
            except Exception as e:
                logger.warning(f"Failed to parse {py_file}: {e}")
                continue

        return self.discovered_benchmarks

    def _discover_in_file(self, file_path: Path) -> list[BenchmarkInfo]:
        """Discover benchmarks in a single Python file."""
        benchmarks = []

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content, filename=str(file_path))

            # Calculate module path relative to benchmark directory
            rel_path = file_path.relative_to(self.benchmark_dir)
            module_path = str(rel_path.with_suffix("")).replace(os.sep, ".")

            # Discover function-level benchmarks (only at module level, not inside classes)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    if self._is_benchmark_function(node.name):
                        benchmark = BenchmarkInfo(
                            name=node.name, module_path=module_path, benchmark_type="function"
                        )
                        benchmarks.append(benchmark)

            # Discover class-level benchmarks
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    class_benchmarks = self._discover_class_benchmarks(node, module_path)
                    benchmarks.extend(class_benchmarks)

        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")

        return benchmarks

    def _discover_class_benchmarks(
        self, class_node: ast.ClassDef, module_path: str
    ) -> list[BenchmarkInfo]:
        """Discover benchmarks within a class."""
        benchmarks = []

        # Check if class has params and param_names attributes
        params = None
        param_names = None
        setup_method = None

        # Check if we need to evaluate params at runtime
        needs_runtime_eval = False

        for node in class_node.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id == "params" and (
                            isinstance(node.value, ast.List) or isinstance(node.value, ast.Tuple)
                        ):
                            # Check if params contain complex expressions that need runtime evaluation
                            if self._needs_runtime_evaluation(node.value):
                                needs_runtime_eval = True
                                params = None  # Will be evaluated at runtime
                            else:
                                params = self._extract_params(node.value)
                        elif target.id == "param_names" and isinstance(node.value, ast.List):
                            param_names = self._extract_param_names(node.value)

            elif isinstance(node, ast.FunctionDef) and node.name == "setup":
                setup_method = node.name

        # Find benchmark methods in the class
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                if self._is_benchmark_function(node.name):
                    # Extract method parameters
                    method_params = self._get_method_parameters(node)

                    benchmark = BenchmarkInfo(
                        name=node.name,
                        module_path=module_path,
                        benchmark_type="method",
                        class_name=class_node.name,
                        params=params,
                        param_names=param_names,
                        setup_method=setup_method,
                        has_setup=setup_method is not None,
                        method_params=method_params,
                        needs_runtime_eval=needs_runtime_eval,
                    )
                    benchmarks.append(benchmark)

        return benchmarks

    def _get_method_parameters(self, method_node: ast.FunctionDef) -> list[str]:
        """Extract parameter names from a method definition."""
        params = []
        for arg in method_node.args.args:
            if arg.arg != "self":  # Skip self parameter
                params.append(arg.arg)
        return params

    def _is_benchmark_function(self, name: str) -> bool:
        """Check if a function name indicates it's a benchmark."""
        return any(name.startswith(prefix) for prefix in self.BENCHMARK_PREFIXES)

    def _extract_params(self, params_node) -> list[list[Any]]:
        """Extract parameter values from AST."""
        params = []
        # Handle both List and Tuple nodes
        if isinstance(params_node, (ast.List, ast.Tuple)):
            # Check if this is a simple list (single parameter) or nested list (multiple parameters)
            if len(params_node.elts) > 0 and isinstance(params_node.elts[0], (ast.List, ast.Tuple)):
                # Nested list - multiple parameters
                for elt in params_node.elts:
                    param_values = []
                    for val in elt.elts:
                        param_values.append(self._extract_literal_value(val))
                    params.append(param_values)
            else:
                # Simple list - single parameter
                param_values = []
                for val in params_node.elts:
                    param_values.append(self._extract_literal_value(val))
                params.append(param_values)
        return params

    def _extract_param_names(self, param_names_node: ast.List) -> list[str]:
        """Extract parameter names from AST."""
        param_names = []
        for elt in param_names_node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                param_names.append(elt.value)
        return param_names

    def _needs_runtime_evaluation(self, params_node) -> bool:
        """Check if params contain complex expressions that need runtime evaluation."""
        if isinstance(params_node, (ast.List, ast.Tuple)):
            for elt in params_node.elts:
                if isinstance(elt, (ast.List, ast.Tuple)):
                    # Check nested lists/tuples
                    if self._needs_runtime_evaluation(elt):
                        return True
                elif not isinstance(elt, (ast.Constant, ast.NameConstant, ast.Num, ast.Str)):
                    # If it's not a simple literal, it needs runtime evaluation
                    return True
        return False

    def _extract_literal_value(self, node: ast.AST) -> Any:
        """Extract literal value from AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.List):
            return [self._extract_literal_value(elt) for elt in node.elts]
        elif isinstance(node, ast.Tuple):
            return tuple(self._extract_literal_value(elt) for elt in node.elts)
        elif isinstance(node, ast.NameConstant):  # Python < 3.8
            return node.value
        elif isinstance(node, ast.Num):  # Python < 3.8
            return node.n
        elif isinstance(node, ast.Str):  # Python < 3.8
            return node.s
        else:
            # For complex expressions, we can't evaluate them statically
            # Return a placeholder that indicates this needs runtime evaluation
            return f"<runtime_eval:{ast.unparse(node) if hasattr(ast, 'unparse') else str(node)}>"

    def generate_parameter_combinations(self, benchmark: BenchmarkInfo) -> list[tuple[Any, ...]]:
        """Generate all parameter combinations for a parameterized benchmark."""
        if not benchmark.params:
            return [()]

        if benchmark.needs_runtime_eval:
            # For benchmarks that need runtime evaluation, we can't generate combinations statically
            # Return a placeholder that indicates runtime evaluation is needed
            return [("<runtime_eval>",)]

        import itertools

        return list(itertools.product(*benchmark.params))

    def get_benchmark_by_name(self, name: str) -> Optional[BenchmarkInfo]:
        """Get a benchmark by its name."""
        for benchmark in self.discovered_benchmarks:
            if benchmark.name == name:
                return benchmark
        return None

    def get_benchmarks_by_module(self, module_path: str) -> list[BenchmarkInfo]:
        """Get all benchmarks from a specific module."""
        return [b for b in self.discovered_benchmarks if b.module_path == module_path]
