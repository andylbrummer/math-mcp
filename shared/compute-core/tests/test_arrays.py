"""Tests for array utilities."""

from unittest.mock import patch

import numpy as np
import pytest
from compute_core.arrays import ensure_array, get_array_module, to_gpu, to_numpy


def test_get_array_module_numpy() -> None:
    """Test get_array_module returns numpy for numpy arrays."""
    arr = np.array([1, 2, 3])
    xp = get_array_module(arr)
    assert xp is np


@pytest.mark.gpu
def test_get_array_module_cupy() -> None:
    """Test get_array_module returns cupy for cupy arrays."""
    try:
        import cupy as cp  # noqa: PLC0415

        arr = cp.array([1, 2, 3])
        xp = get_array_module(arr)
        assert xp is cp
    except ImportError:
        pytest.skip("CuPy not available")


def test_to_numpy() -> None:
    """Test converting to numpy."""
    arr = np.array([1, 2, 3])
    result = to_numpy(arr)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(arr, result)


@pytest.mark.gpu
def test_to_numpy_from_gpu() -> None:
    """Test converting from GPU to numpy."""
    try:
        import cupy as cp  # noqa: PLC0415

        arr = cp.array([1, 2, 3])
        result = to_numpy(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(cp.asnumpy(arr), result)
    except ImportError:
        pytest.skip("CuPy not available")


def test_to_gpu_without_cupy() -> None:
    """Test to_gpu fallback when CuPy is unavailable."""
    with patch("compute_core.arrays.CUPY_AVAILABLE", False):
        arr = np.array([1, 2, 3])
        result = to_gpu(arr)
        # Should return numpy array with warning when CuPy unavailable
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(arr, result)


@pytest.mark.gpu
def test_to_gpu_with_cupy() -> None:
    """Test to_gpu with CuPy available."""
    try:
        import cupy as cp  # noqa: PLC0415

        arr = np.array([1, 2, 3])
        result = to_gpu(arr)
        assert hasattr(result, "__cuda_array_interface__")
        np.testing.assert_array_equal(arr, cp.asnumpy(result))
    except ImportError:
        pytest.skip("CuPy not available")


def test_ensure_array_from_list() -> None:
    """Test ensure_array from list."""
    data = [1, 2, 3, 4, 5]
    result = ensure_array(data)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array(data))


@pytest.mark.gpu
def test_ensure_array_gpu() -> None:
    """Test ensure_array on GPU."""
    try:
        import cupy as cp  # noqa: PLC0415

        data = [1, 2, 3, 4, 5]
        result = ensure_array(data, use_gpu=True)
        assert hasattr(result, "__cuda_array_interface__")
        np.testing.assert_array_equal(np.array(data), cp.asnumpy(result))
    except ImportError:
        pytest.skip("CuPy not available")
