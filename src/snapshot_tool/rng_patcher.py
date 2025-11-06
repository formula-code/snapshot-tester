"""
Random Number Generator (RNG) Patcher for Deterministic Execution.

This module provides seeding capabilities to ensure all random number
generators across different libraries (numpy, PyTorch, TensorFlow, etc.) produce
deterministic results. This is critical for snapshot testing where reproducibility
is required.

The patcher handles:
- Python's built-in `random` module
- NumPy's legacy random API (np.random.seed) - compatible with numpy 1.12+ (2017)
- PyTorch random number generation (if installed)
- TensorFlow random number generation (if installed)
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class RNGPatcher:
    """Seeds random number generators for deterministic execution."""

    def __init__(self, seed: int = 42):
        """
        Initialize the RNG patcher.

        Args:
            seed: The deterministic seed to use for all RNG sources.
        """
        self.seed = seed
        self._patched = False

    def patch_all(self):
        """Apply all RNG patches to make execution deterministic."""
        if self._patched:
            logger.debug("RNG patcher already applied, re-seeding RNGs")
            # Re-seed even if already patched
            self._patch_python_random()
            self._patch_numpy_legacy()
            self._patch_torch()
            self._patch_tensorflow()
            return

        logger.debug("Applying RNG patches with seed=%d", self.seed)

        self._patch_python_random()
        self._patch_numpy_legacy()
        self._patch_torch()
        self._patch_tensorflow()

        self._patched = True

    def unpatch_all(self):
        """Restore all original RNG functions."""
        if not self._patched:
            return

        logger.debug("Restoring original RNG functions")
        self._patched = False

    def _patch_python_random(self):
        """Patch Python's built-in random module."""
        import random
        random.seed(self.seed)
        logger.debug("Patched Python random module")

    def _patch_numpy_legacy(self):
        """Patch NumPy's legacy random API (compatible with numpy 1.12+/2017)."""
        try:
            import numpy as np
            np.random.seed(self.seed)
            logger.debug("Patched NumPy legacy random API")
        except ImportError:
            logger.debug("NumPy not available, skipping numpy random patching")

    def _patch_torch(self):
        """Patch PyTorch random number generation."""
        try:
            import torch  # type: ignore
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
            logger.debug("Patched PyTorch random")
        except ImportError:
            logger.debug("PyTorch not available, skipping torch random patching")

    def _patch_tensorflow(self):
        """Patch TensorFlow random number generation."""
        try:
            import tensorflow as tf  # type: ignore
            tf.random.set_seed(self.seed)
            logger.debug("Patched TensorFlow random")
        except ImportError:
            logger.debug("TensorFlow not available, skipping tensorflow random patching")

    def __enter__(self):
        """Context manager entry - apply patches."""
        self.patch_all()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # pylint: disable=unused-argument
        """Context manager exit - restore original functions."""
        self.unpatch_all()
        return False


# Global patcher instance for convenience
_global_patcher: Optional[RNGPatcher] = None


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
