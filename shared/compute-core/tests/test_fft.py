"""Tests for FFT operations."""

import numpy as np
import pytest
from compute_core.fft import fft, fft2, fftfreq, ifft, ifft2, irfft, rfft

rng = np.random.default_rng()


def test_fft_1d() -> None:
    """Test 1D FFT."""
    x = np.array([1, 2, 3, 4, 5])
    result = fft(x)
    expected = np.fft.fft(x)
    np.testing.assert_array_almost_equal(result, expected)


def test_ifft_1d() -> None:
    """Test 1D IFFT."""
    x = np.array([1, 2, 3, 4, 5])
    fft_result = fft(x)
    result = ifft(fft_result)
    np.testing.assert_array_almost_equal(result, x)


def test_fft_ifft_roundtrip() -> None:
    """Test FFT-IFFT roundtrip."""
    x = rng.random(100)
    result = ifft(fft(x))
    np.testing.assert_array_almost_equal(result, x)


def test_fft2_2d() -> None:
    """Test 2D FFT."""
    x = rng.random((10, 10))
    result = fft2(x)
    expected = np.fft.fft2(x)
    np.testing.assert_array_almost_equal(result, expected)


def test_ifft2_2d() -> None:
    """Test 2D IFFT."""
    x = rng.random((10, 10))
    fft_result = fft2(x)
    result = ifft2(fft_result)
    np.testing.assert_array_almost_equal(result, x)


def test_rfft_real_input() -> None:
    """Test real FFT with real input."""
    x = rng.random(100)
    result = rfft(x)
    expected = np.fft.rfft(x)
    np.testing.assert_array_almost_equal(result, expected)


def test_irfft_roundtrip() -> None:
    """Test RFFT-IRFFT roundtrip."""
    x = rng.random(100)
    result = irfft(rfft(x))
    np.testing.assert_array_almost_equal(result, x)


def test_fftfreq() -> None:
    """Test FFT frequency calculation."""
    n = 10
    d = 0.1
    result = fftfreq(n, d)
    expected = np.fft.fftfreq(n, d)
    np.testing.assert_array_almost_equal(result, expected)


@pytest.mark.gpu
def test_fft_gpu() -> None:
    """Test FFT on GPU."""
    try:
        import cupy as cp  # noqa: PLC0415
        from compute_core.arrays import to_gpu  # noqa: PLC0415

        x = rng.random(100)
        x_gpu = to_gpu(x)
        result = fft(x_gpu)

        assert hasattr(result, "__cuda_array_interface__")

        # Verify correctness
        expected = np.fft.fft(x)
        np.testing.assert_array_almost_equal(cp.asnumpy(result), expected)
    except ImportError:
        pytest.skip("CuPy not available")


@pytest.mark.gpu
def test_fft2_gpu() -> None:
    """Test 2D FFT on GPU."""
    try:
        import cupy as cp  # noqa: PLC0415
        from compute_core.arrays import to_gpu  # noqa: PLC0415

        x = rng.random((50, 50))
        x_gpu = to_gpu(x)
        result = fft2(x_gpu)

        assert hasattr(result, "__cuda_array_interface__")

        # Verify correctness
        expected = np.fft.fft2(x)
        np.testing.assert_array_almost_equal(cp.asnumpy(result), expected)
    except ImportError:
        pytest.skip("CuPy not available")
