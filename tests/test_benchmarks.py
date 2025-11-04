"""
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')
Minimal test benchmarks to verify the snapshot testing tool works.
"""

import numpy as np


def time_simple_function():
    """Simple function benchmark that returns a value."""
    return np.array([1, 2, 3, 4, 5])


def time_simple_calculation():
    """Simple calculation benchmark."""
    x = np.linspace(0, 10, 100)
    y = np.sin(x) * np.exp(-x / 5)
    return y


class SimpleBenchmark:
    """Simple class-based benchmark."""

    def setup(self):
        """Setup method."""
        self.data = np.random.random(1000)
        self.multiplier = 2.5

    def time_calculation(self):
        """Benchmark method."""
        result = self.data * self.multiplier
        return result


class ParameterizedBenchmark:
    """Parameterized benchmark."""

    params = ([1, 2, 3], [10, 100, 1000])
    param_names = ["multiplier", "size"]

    def setup(self, multiplier, size):
        """Setup with parameters."""
        self.multiplier = multiplier
        self.data = np.random.random(size)

    def time_parameterized_calculation(self):
        """Parameterized benchmark method."""
        result = self.data * self.multiplier
        return result
