"""
Tests for the RNG patcher module.

This test suite verifies that all random number generators are properly
monkey-patched to produce deterministic results across multiple runs.
"""

import random

import numpy as np
import pytest

from snapshot_tool import RNGPatcher, patch_all_rngs, reset_all_rngs, unpatch_all_rngs


class TestRNGPatcher:
    """Test the RNG patcher functionality."""

    def setup_method(self):
        """Ensure clean state before each test."""
        # Unpatch any global patching from previous tests
        unpatch_all_rngs()

    def teardown_method(self):
        """Clean up after each test."""
        # Ensure we unpatch after each test
        unpatch_all_rngs()

    def test_python_random_determinism(self):
        """Test that Python's random module produces deterministic results."""
        patcher = RNGPatcher(seed=12345)
        patcher.patch_all()

        # Generate random numbers
        values1 = [random.random() for _ in range(10)]

        # Reset and generate again
        patcher.patch_all()
        values2 = [random.random() for _ in range(10)]

        # Should be identical
        assert values1 == values2

        patcher.unpatch_all()

    def test_numpy_legacy_random_determinism(self):
        """Test that NumPy's legacy random API produces deterministic results."""
        patcher = RNGPatcher(seed=12345)
        patcher.patch_all()

        # Generate random numbers
        values1 = np.random.rand(10)

        # Reset and generate again
        patcher.patch_all()
        values2 = np.random.rand(10)

        # Should be identical
        np.testing.assert_array_equal(values1, values2)

        patcher.unpatch_all()

    def test_numpy_generator_pcg64_determinism(self):
        """Test that NumPy Generator with PCG64 produces deterministic results."""
        patcher = RNGPatcher(seed=12345)
        patcher.patch_all()

        # Create generators with different seeds (should be ignored)
        gen1 = np.random.Generator(np.random.PCG64(999))
        values1 = gen1.uniform(0, 20, 10)

        gen2 = np.random.Generator(np.random.PCG64(888))
        values2 = gen2.uniform(0, 20, 10)

        # Should be identical despite different seed arguments
        np.testing.assert_array_equal(values1, values2)

        patcher.unpatch_all()

    def test_numpy_generator_mt19937_determinism(self):
        """Test that NumPy Generator with MT19937 produces deterministic results."""
        patcher = RNGPatcher(seed=12345)
        patcher.patch_all()

        # Create generators with different seeds (should be ignored)
        gen1 = np.random.Generator(np.random.MT19937(999))
        values1 = gen1.uniform(0, 20, 10)

        gen2 = np.random.Generator(np.random.MT19937(888))
        values2 = gen2.uniform(0, 20, 10)

        # Should be identical despite different seed arguments
        np.testing.assert_array_equal(values1, values2)

        patcher.unpatch_all()

    def test_numpy_generator_philox_determinism(self):
        """Test that NumPy Generator with Philox produces deterministic results."""
        patcher = RNGPatcher(seed=12345)
        patcher.patch_all()

        # Create generators with different seeds (should be ignored)
        gen1 = np.random.Generator(np.random.Philox(999))
        values1 = gen1.uniform(0, 20, 10)

        gen2 = np.random.Generator(np.random.Philox(888))
        values2 = gen2.uniform(0, 20, 10)

        # Should be identical despite different seed arguments
        np.testing.assert_array_equal(values1, values2)

        patcher.unpatch_all()

    def test_numpy_generator_sfc64_determinism(self):
        """Test that NumPy Generator with SFC64 produces deterministic results."""
        patcher = RNGPatcher(seed=12345)
        patcher.patch_all()

        # Create generators with different seeds (should be ignored)
        gen1 = np.random.Generator(np.random.SFC64(999))
        values1 = gen1.uniform(0, 20, 10)

        gen2 = np.random.Generator(np.random.SFC64(888))
        values2 = gen2.uniform(0, 20, 10)

        # Should be identical despite different seed arguments
        np.testing.assert_array_equal(values1, values2)

        patcher.unpatch_all()

    def test_context_manager(self):
        """Test that the context manager properly patches and unpatches."""
        # Generate without patching
        gen1 = np.random.Generator(np.random.PCG64(999))
        values_before = gen1.uniform(0, 20, 5)

        # Use context manager
        with RNGPatcher(seed=12345):
            gen2 = np.random.Generator(np.random.PCG64(999))
            values_patched1 = gen2.uniform(0, 20, 5)

            gen3 = np.random.Generator(np.random.PCG64(888))
            values_patched2 = gen3.uniform(0, 20, 5)

            # Should be identical when patched
            np.testing.assert_array_equal(values_patched1, values_patched2)

        # After context manager, should work normally again
        gen4 = np.random.Generator(np.random.PCG64(999))
        values_after = gen4.uniform(0, 20, 5)

        # Values before and after should be the same (same seed, unpatched)
        np.testing.assert_array_equal(values_before, values_after)

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        patcher1 = RNGPatcher(seed=12345)
        patcher1.patch_all()
        gen1 = np.random.Generator(np.random.PCG64(999))
        values1 = gen1.uniform(0, 20, 10)
        patcher1.unpatch_all()

        patcher2 = RNGPatcher(seed=54321)
        patcher2.patch_all()
        gen2 = np.random.Generator(np.random.PCG64(999))
        values2 = gen2.uniform(0, 20, 10)
        patcher2.unpatch_all()

        # Different seeds should produce different results
        assert not np.array_equal(values1, values2)

    def test_global_patch_function(self):
        """Test the global patch_all_rngs function."""
        patch_all_rngs(seed=12345)

        gen1 = np.random.Generator(np.random.PCG64(999))
        values1 = gen1.uniform(0, 20, 10)

        gen2 = np.random.Generator(np.random.PCG64(888))
        values2 = gen2.uniform(0, 20, 10)

        # Should be identical
        np.testing.assert_array_equal(values1, values2)

        unpatch_all_rngs()

    def test_reset_all_rngs_function(self):
        """Test the reset_all_rngs function."""
        reset_all_rngs(seed=12345)

        # Legacy numpy random should be deterministic
        values1 = np.random.rand(10)

        reset_all_rngs(seed=12345)
        values2 = np.random.rand(10)

        np.testing.assert_array_equal(values1, values2)

    def test_unpatch_restores_original_behavior(self):
        """Test that unpatching restores original RNG behavior."""
        # Create a generator with a specific seed
        gen1 = np.random.Generator(np.random.PCG64(999))
        values1 = gen1.uniform(0, 20, 10)

        # Patch
        patcher = RNGPatcher(seed=12345)
        patcher.patch_all()

        gen2 = np.random.Generator(np.random.PCG64(999))
        values2 = gen2.uniform(0, 20, 10)

        # Values should be different (patched vs unpatched)
        assert not np.array_equal(values1, values2)

        # Unpatch
        patcher.unpatch_all()

        # After unpatching, same seed should produce same values as before
        gen3 = np.random.Generator(np.random.PCG64(999))
        values3 = gen3.uniform(0, 20, 10)

        np.testing.assert_array_equal(values1, values3)

    def test_multiple_patch_calls_safe(self):
        """Test that calling patch_all multiple times is safe."""
        patcher = RNGPatcher(seed=12345)

        # Should not raise an error
        patcher.patch_all()
        patcher.patch_all()  # Second call should be safe

        patcher.unpatch_all()

    def test_benchmark_scenario_from_wcs(self):
        """
        Test the exact scenario from the wcs.py benchmark.

        This reproduces the code pattern from tests/test_repos/astropy_benchmarks/benchmarks/wcs.py
        to verify that our patching makes it deterministic.
        """
        patcher = RNGPatcher(seed=12345)
        patcher.patch_all()

        # First run - mimics the setup() method
        np.random.seed(12345)
        gen1 = np.random.Generator(np.random.PCG64(12345))
        px1 = gen1.uniform(0, 20, 100)
        py1 = gen1.uniform(0, 20, 100)

        # Second run - should produce identical results
        np.random.seed(12345)
        gen2 = np.random.Generator(np.random.PCG64(12345))
        px2 = gen2.uniform(0, 20, 100)
        py2 = gen2.uniform(0, 20, 100)

        # Values should be identical
        np.testing.assert_array_equal(px1, px2)
        np.testing.assert_array_equal(py1, py2)

        patcher.unpatch_all()


class TestRNGPatcherEdgeCases:
    """Test edge cases and error handling."""

    def setup_method(self):
        """Ensure clean state before each test."""
        unpatch_all_rngs()

    def teardown_method(self):
        """Clean up after each test."""
        unpatch_all_rngs()

    def test_unpatch_without_patch_safe(self):
        """Test that unpatching without patching is safe."""
        patcher = RNGPatcher(seed=12345)
        # Should not raise an error
        patcher.unpatch_all()

    def test_context_manager_with_exception(self):
        """Test that context manager properly unpatches even with exceptions."""
        try:
            with RNGPatcher(seed=12345):
                gen1 = np.random.Generator(np.random.PCG64(999))
                gen1.uniform(0, 20, 10)
                raise ValueError("Test exception")
        except ValueError:
            pass

        # After exception, should be unpatched
        gen2 = np.random.Generator(np.random.PCG64(999))
        values1 = gen2.uniform(0, 20, 10)

        gen3 = np.random.Generator(np.random.PCG64(999))
        values2 = gen3.uniform(0, 20, 10)

        # Should produce same values (unpatched, same seed)
        np.testing.assert_array_equal(values1, values2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
