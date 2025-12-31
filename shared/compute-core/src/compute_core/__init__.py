"""Unified computational primitives for CPU/GPU."""

from compute_core.arrays import get_array_module, to_gpu, to_numpy
from compute_core.fft import fft, fft2, ifft, ifft2, irfft, rfft
from compute_core.linalg import cholesky, eig, matmul, solve, svd

__all__ = [
    "get_array_module",
    "to_numpy",
    "to_gpu",
    "fft",
    "ifft",
    "fft2",
    "ifft2",
    "rfft",
    "irfft",
    "matmul",
    "solve",
    "eig",
    "svd",
    "cholesky",
]
