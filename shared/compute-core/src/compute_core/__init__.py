"""Unified computational primitives for CPU/GPU."""

from compute_core.arrays import get_array_module, to_gpu, to_numpy
from compute_core.fft import fft, fft2, ifft, ifft2, irfft, rfft
from compute_core.linalg import cholesky, eig, matmul, solve, svd

__all__ = [
    "cholesky",
    "eig",
    "fft",
    "fft2",
    "get_array_module",
    "ifft",
    "ifft2",
    "irfft",
    "matmul",
    "rfft",
    "solve",
    "svd",
    "to_gpu",
    "to_numpy",
]
