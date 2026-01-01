"""FFT operations with unified NumPy/CuPy interface."""

import logging
from typing import Any

import numpy as np

from compute_core.arrays import get_array_module

logger = logging.getLogger(__name__)


def fft(
    arr: Any,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
) -> Any:
    """1D Fast Fourier Transform.

    Args:
        arr: Input array
        n: Length of transformed axis
        axis: Axis along which to compute FFT
        norm: Normalization mode ('backward', 'ortho', 'forward')

    Returns:
        Transformed array (same type as input)
    """
    xp = get_array_module(arr)
    return xp.fft.fft(arr, n=n, axis=axis, norm=norm)


def ifft(
    arr: Any,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
) -> Any:
    """1D Inverse Fast Fourier Transform.

    Args:
        arr: Input array
        n: Length of transformed axis
        axis: Axis along which to compute IFFT
        norm: Normalization mode ('backward', 'ortho', 'forward')

    Returns:
        Transformed array (same type as input)
    """
    xp = get_array_module(arr)
    return xp.fft.ifft(arr, n=n, axis=axis, norm=norm)


def fft2(
    arr: Any,
    s: tuple[int, ...] | None = None,
    axes: tuple[int, int] = (-2, -1),
    norm: str | None = None,
) -> Any:
    """2D Fast Fourier Transform.

    Args:
        arr: Input array
        s: Shape of output
        axes: Axes along which to compute FFT
        norm: Normalization mode

    Returns:
        Transformed array (same type as input)
    """
    xp = get_array_module(arr)
    return xp.fft.fft2(arr, s=s, axes=axes, norm=norm)


def ifft2(
    arr: Any,
    s: tuple[int, ...] | None = None,
    axes: tuple[int, int] = (-2, -1),
    norm: str | None = None,
) -> Any:
    """2D Inverse Fast Fourier Transform.

    Args:
        arr: Input array
        s: Shape of output
        axes: Axes along which to compute IFFT
        norm: Normalization mode

    Returns:
        Transformed array (same type as input)
    """
    xp = get_array_module(arr)
    return xp.fft.ifft2(arr, s=s, axes=axes, norm=norm)


def rfft(
    arr: Any,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
) -> Any:
    """1D Real FFT (input is real, output is complex).

    Args:
        arr: Real input array
        n: Length of transformed axis
        axis: Axis along which to compute RFFT
        norm: Normalization mode

    Returns:
        Complex transformed array
    """
    xp = get_array_module(arr)
    return xp.fft.rfft(arr, n=n, axis=axis, norm=norm)


def irfft(
    arr: Any,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
) -> Any:
    """1D Inverse Real FFT (input is complex, output is real).

    Args:
        arr: Complex input array
        n: Length of transformed axis
        axis: Axis along which to compute IRFFT
        norm: Normalization mode

    Returns:
        Real transformed array
    """
    xp = get_array_module(arr)
    return xp.fft.irfft(arr, n=n, axis=axis, norm=norm)


def fftfreq(n: int, d: float = 1.0, use_gpu: bool = False) -> Any:
    """Return FFT sample frequencies.

    Args:
        n: Window length
        d: Sample spacing
        use_gpu: Return on GPU if available

    Returns:
        Array of frequencies
    """
    if use_gpu:
        try:
            import cupy as cp  # noqa: PLC0415

            return cp.fft.fftfreq(n, d=d)
        except ImportError:
            pass

    return np.fft.fftfreq(n, d=d)


def fftshift(arr: Any, axes: tuple[int, ...] | None = None) -> Any:
    """Shift zero-frequency component to center.

    Args:
        arr: Input array
        axes: Axes over which to shift

    Returns:
        Shifted array
    """
    xp = get_array_module(arr)
    return xp.fft.fftshift(arr, axes=axes)


def ifftshift(arr: Any, axes: tuple[int, ...] | None = None) -> Any:
    """Inverse of fftshift.

    Args:
        arr: Input array
        axes: Axes over which to shift

    Returns:
        Shifted array
    """
    xp = get_array_module(arr)
    return xp.fft.ifftshift(arr, axes=axes)
