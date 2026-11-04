"""Reseed random number generators so benchmark outputs are reproducible.

Seeds Python ``random``, the numpy legacy API (``np.random.seed``), PyTorch and TensorFlow (when
installed). ``numpy.random.Generator`` instances are not seeded. Nothing is monkey-patched.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class RNGPatcher:
    """Seeds random number generators for deterministic execution."""

    def __init__(self, seed: int = 42):
        """Store the seed used for every RNG."""
        self.seed = seed
        self._patched = False

    def patch_all(self):
        """Reseed all supported RNGs."""
        if self._patched:
            logger.debug("RNG patcher already applied, re-seeding RNGs")
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
        """Clear the patched flag; RNG state is not restored."""
        if not self._patched:
            return

        logger.debug("Restoring original RNG functions")
        self._patched = False

    def _patch_python_random(self):
        """Seed Python's random module."""
        import random

        random.seed(self.seed)
        logger.debug("Patched Python random module")

    def _patch_numpy_legacy(self):
        """Seed numpy's legacy random API."""
        try:
            import numpy as np

            np.random.seed(self.seed)
            logger.debug("Patched NumPy legacy random API")
        except ImportError:
            logger.debug("NumPy not available, skipping numpy random patching")

    def _patch_torch(self):
        """Seed PyTorch, including CUDA when available."""
        try:
            import torch  # type: ignore

            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
            logger.debug("Patched PyTorch random")
        except ImportError:
            logger.debug("PyTorch not available, skipping torch random patching")

    def _patch_tensorflow(self):
        """Seed TensorFlow."""
        try:
            import tensorflow as tf  # type: ignore

            tf.random.set_seed(self.seed)
            logger.debug("Patched TensorFlow random")
        except ImportError:
            logger.debug("TensorFlow not available, skipping tensorflow random patching")

    def __enter__(self):
        """Reseed on entry."""
        self.patch_all()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # pylint: disable=unused-argument
        """Clear the patched flag on exit; RNG state is not restored."""
        self.unpatch_all()
        return False


_global_patcher: Optional[RNGPatcher] = None


def patch_all_rngs(seed: int = 42):
    """Reseed all RNGs once; later calls only warn until unpatch_all_rngs() is called."""
    global _global_patcher

    if _global_patcher is not None:
        logger.warning("Global RNG patcher already applied")
        return

    _global_patcher = RNGPatcher(seed=seed)
    _global_patcher.patch_all()


def unpatch_all_rngs():
    """Drop the global patcher so patch_all_rngs() reseeds again; RNG state is not restored."""
    global _global_patcher

    if _global_patcher is None:
        return

    _global_patcher.unpatch_all()
    _global_patcher = None


def reset_all_rngs(seed: int = 42):
    """Reseed the same RNGs as RNGPatcher.patch_all(), without a patcher object."""
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
