"""Unified array interface for NumPy/CuPy."""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Try to import CuPy
try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


def get_array_module(arr: Any) -> Any:
    """Get appropriate array module (numpy or cupy) for an array.

    Args:
        arr: NumPy or CuPy array

    Returns:
        numpy or cupy module
    """
    if CUPY_AVAILABLE and hasattr(arr, "__cuda_array_interface__"):
        return cp
    return np


def to_numpy(arr: Any) -> np.ndarray:
    """Convert array to NumPy (from GPU if needed).

    Args:
        arr: NumPy or CuPy array

    Returns:
        NumPy array
    """
    if CUPY_AVAILABLE and hasattr(arr, "get"):
        result: np.ndarray = arr.get()
        return result
    return np.asarray(arr)


def to_gpu(arr: Any, force: bool = False) -> Any:
    """Convert array to GPU if available.

    Args:
        arr: NumPy or CuPy array
        force: Raise error if CuPy not available

    Returns:
        CuPy array if available, otherwise NumPy array
    """
    if not CUPY_AVAILABLE:
        if force:
            msg = "CuPy not available - cannot transfer to GPU"
            raise RuntimeError(msg)
        logger.warning("CuPy not available - keeping array on CPU")
        return arr

    if hasattr(arr, "__cuda_array_interface__"):
        return arr  # Already on GPU

    return cp.asarray(arr)


def ensure_array(data: list | tuple | np.ndarray | Any, use_gpu: bool = False) -> Any:
    """Ensure data is an array (NumPy or CuPy).

    Args:
        data: List, tuple, or array
        use_gpu: Create on GPU if available

    Returns:
        NumPy or CuPy array
    """
    if use_gpu and CUPY_AVAILABLE:
        if hasattr(data, "__cuda_array_interface__"):
            return data
        return cp.asarray(data)

    return np.asarray(data)
