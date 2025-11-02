"""
Random Number Generator (RNG) Patcher for Deterministic Execution.

This module provides monkey-patching capabilities to ensure all random number
generators across different libraries (numpy, PyTorch, TensorFlow, etc.) produce
deterministic results. This is critical for snapshot testing where reproducibility
is required.

The patcher handles:
- Python's built-in `random` module
- NumPy's legacy random API (np.random.seed, np.random.rand, etc.)
- NumPy's modern Generator API (Generator, PCG64, MT19937, etc.)
- PyTorch random number generation (if installed)
- TensorFlow random number generation (if installed)
"""

import logging
import sys
from functools import wraps
from typing import Any

logger = logging.getLogger(__name__)


class RNGPatcher:
    """Monkey-patches random number generators for deterministic execution."""

    def __init__(self, seed: int = 42):
        """
        Initialize the RNG patcher.

        Args:
            seed: The deterministic seed to use for all RNG sources.
        """
        self.seed = seed
        self._original_functions: dict[str, Any] = {}
        self._patched = False

    def patch_all(self):
        """Apply all RNG patches to make execution deterministic."""
        if self._patched:
            logger.warning("RNG patcher already applied, re-seeding RNGs")
            # Re-seed even if already patched
            self._patch_python_random()
            self._patch_numpy_legacy()
            # Update the class variable so patched functions use new seed
            RNGPatcher._current_seed = self.seed
            return

        logger.debug("Applying RNG patches with seed=%d", self.seed)

        # Update the class variable before patching
        RNGPatcher._current_seed = self.seed

        self._patch_python_random()
        self._patch_numpy_legacy()
        self._patch_numpy_generator()
        self._patch_torch()
        self._patch_tensorflow()

        self._patched = True

    def unpatch_all(self):
        """Restore all original RNG functions."""
        if not self._patched:
            return

        logger.debug("Restoring original RNG functions")

        # Restore numpy Generator and BitGenerators
        if "numpy.random" in sys.modules:
            import numpy as np
            if "numpy.random.Generator" in self._original_functions:
                np.random.Generator = self._original_functions["numpy.random.Generator"]
            if "numpy.random.PCG64" in self._original_functions:
                np.random.PCG64 = self._original_functions["numpy.random.PCG64"]
            if "numpy.random.MT19937" in self._original_functions:
                np.random.MT19937 = self._original_functions["numpy.random.MT19937"]
            if "numpy.random.Philox" in self._original_functions:
                np.random.Philox = self._original_functions["numpy.random.Philox"]
            if "numpy.random.SFC64" in self._original_functions:
                np.random.SFC64 = self._original_functions["numpy.random.SFC64"]

        self._original_functions.clear()
        self._patched = False

    def _patch_python_random(self):
        """Patch Python's built-in random module."""
        import random
        random.seed(self.seed)
        logger.debug("Patched Python random module")

    def _patch_numpy_legacy(self):
        """Patch NumPy's legacy random API."""
        try:
            import numpy as np
            np.random.seed(self.seed)
            logger.debug("Patched NumPy legacy random API")
        except ImportError:
            pass

    def _patch_numpy_generator(self):
        """
        Patch NumPy's modern Generator API.

        Since BitGenerator classes are immutable C types, we patch at the module level
        by replacing the class references with wrapper classes that override seeds.
        We use classes (not functions) to maintain compatibility with type annotations.
        """
        try:
            import numpy as np

            # Store original classes
            if not hasattr(np.random, "PCG64"):
                return

            # Store originals only if not already stored
            if "numpy.random.PCG64" not in self._original_functions:
                self._original_functions["numpy.random.PCG64"] = np.random.PCG64
                self._original_functions["numpy.random.MT19937"] = np.random.MT19937
                self._original_functions["numpy.random.Philox"] = np.random.Philox
                self._original_functions["numpy.random.SFC64"] = np.random.SFC64
                self._original_functions["numpy.random.Generator"] = np.random.Generator

            # Get original classes
            original_pcg64 = self._original_functions["numpy.random.PCG64"]
            original_mt19937 = self._original_functions["numpy.random.MT19937"]
            original_philox = self._original_functions["numpy.random.Philox"]
            original_sfc64 = self._original_functions["numpy.random.SFC64"]
            original_generator = self._original_functions["numpy.random.Generator"]

            # Create wrapper classes that maintain class-like behavior
            # This is important for scipy and other libraries that use type annotations
            class WrappedPCG64(original_pcg64):
                """Wrapper for PCG64 that overrides seed."""
                def __new__(cls, seed=None):  # pylint: disable=unused-argument
                    return original_pcg64(RNGPatcher._get_instance_seed())

            class WrappedMT19937(original_mt19937):
                """Wrapper for MT19937 that overrides seed."""
                def __new__(cls, seed=None):  # pylint: disable=unused-argument
                    return original_mt19937(RNGPatcher._get_instance_seed())

            class WrappedPhilox(original_philox):
                """Wrapper for Philox that overrides seed."""
                def __new__(cls, seed=None, counter=None, key=None):  # pylint: disable=unused-argument
                    return original_philox(seed=RNGPatcher._get_instance_seed(), counter=counter, key=key)

            class WrappedSFC64(original_sfc64):
                """Wrapper for SFC64 that overrides seed."""
                def __new__(cls, seed=None):  # pylint: disable=unused-argument
                    return original_sfc64(RNGPatcher._get_instance_seed())

            class WrappedGenerator(original_generator):
                """Wrapper for Generator that ensures seeded BitGenerator."""
                def __new__(cls, bit_generator=None):
                    if bit_generator is None:
                        bit_generator = original_pcg64(RNGPatcher._get_instance_seed())
                    return original_generator(bit_generator)

            # Replace at module level
            np.random.PCG64 = WrappedPCG64  # type: ignore
            np.random.MT19937 = WrappedMT19937  # type: ignore
            np.random.Philox = WrappedPhilox  # type: ignore
            np.random.SFC64 = WrappedSFC64  # type: ignore
            np.random.Generator = WrappedGenerator  # type: ignore

            logger.debug("Patched NumPy Generator API (PCG64, MT19937, Philox, SFC64, Generator)")

        except (ImportError, AttributeError) as e:
            logger.debug("Could not patch NumPy Generator API: %s", str(e))

    def _patch_torch(self):
        """Patch PyTorch random number generation."""
        try:
            import torch  # type: ignore
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
            logger.debug("Patched PyTorch random")
        except ImportError:
            pass

    def _patch_tensorflow(self):
        """Patch TensorFlow random number generation."""
        try:
            import tensorflow as tf  # type: ignore
            tf.random.set_seed(self.seed)
            logger.debug("Patched TensorFlow random")
        except ImportError:
            pass

    @staticmethod
    def _get_instance_seed():
        """
        Get the seed from the current RNG patcher instance.

        This is a workaround since we need to access the instance seed
        from within the patched methods. We use a class variable to store it.
        """
        return getattr(RNGPatcher, "_current_seed", 42)

    def __enter__(self):
        """Context manager entry - apply patches."""
        # Store the seed in a class variable so patched methods can access it
        RNGPatcher._current_seed = self.seed
        self.patch_all()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # pylint: disable=unused-argument
        """Context manager exit - restore original functions."""
        self.unpatch_all()
        # Clean up the class variable
        if hasattr(RNGPatcher, "_current_seed"):
            delattr(RNGPatcher, "_current_seed")
        return False


# Global patcher instance for convenience
_global_patcher: RNGPatcher | None = None


def patch_all_rngs(seed: int = 42):
    """
    Apply all RNG patches globally.

    Args:
        seed: The deterministic seed to use.

    Note:
        This applies patches globally. Use unpatch_all_rngs() to restore original behavior.
        For temporary patching, use the RNGPatcher context manager instead.
    """
    global _global_patcher

    if _global_patcher is not None:
        logger.warning("Global RNG patcher already applied")
        return

    _global_patcher = RNGPatcher(seed=seed)
    _global_patcher.patch_all()


def unpatch_all_rngs():
    """Restore all original RNG functions."""
    global _global_patcher

    if _global_patcher is None:
        return

    _global_patcher.unpatch_all()
    _global_patcher = None


def reset_all_rngs(seed: int = 42):
    """
    Reset all RNG states without monkey-patching.

    This is a lighter-weight alternative that just reseeds all known RNGs
    without modifying their behavior.

    Args:
        seed: The deterministic seed to use.
    """
    import random
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    try:
        import tensorflow as tf  # type: ignore
        tf.random.set_seed(seed)
    except ImportError:
        pass
