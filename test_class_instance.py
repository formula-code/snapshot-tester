"""
Test benchmark that returns class instances to verify class capture functionality.
"""

import numpy as np


class TestClass:
    """A simple test class to verify class instance capture."""
    
    def __init__(self, value):
        self.value = value
        self.data = np.array([1, 2, 3, 4, 5]) * value
    
    def __repr__(self):
        return f"TestClass(value={self.value})"
    
    def __eq__(self, other):
        if not isinstance(other, TestClass):
            return False
        return self.value == other.value and np.array_equal(self.data, other.data)


def create_test_class(value):
    """Function that creates and returns a TestClass instance."""
    return TestClass(value)


def time_class_creation():
    """Benchmark that creates a class instance."""
    return create_test_class(42)


def time_class_with_data():
    """Benchmark that creates a class instance with data."""
    return create_test_class(100)


class ClassBenchmark:
    """Benchmark class that returns class instances."""
    
    def time_class_method(self):
        """Method that returns a class instance."""
        return create_test_class(200)
    
    def time_class_with_params(self):
        """Method that returns a class instance with parameters."""
        return create_test_class(300)
