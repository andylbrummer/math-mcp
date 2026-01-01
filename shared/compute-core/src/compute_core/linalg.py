"""Linear algebra operations with unified NumPy/CuPy interface."""

import logging
from typing import Any

from compute_core.arrays import get_array_module

logger = logging.getLogger(__name__)


def matmul(a: Any, b: Any) -> Any:
    """Matrix multiplication.

    Args:
        a: First matrix
        b: Second matrix

    Returns:
        Product matrix (same type as inputs)
    """
    xp = get_array_module(a)
    return xp.matmul(a, b)


def solve(a: Any, b: Any) -> Any:
    """Solve linear system Ax = b.

    Args:
        a: Coefficient matrix (N x N)
        b: Right-hand side (N,) or (N, K)

    Returns:
        Solution x (same type as inputs)
    """
    xp = get_array_module(a)
    return xp.linalg.solve(a, b)


def eig(a: Any) -> tuple[Any, Any]:
    """Compute eigenvalues and eigenvectors.

    Args:
        a: Square matrix

    Returns:
        Tuple of (eigenvalues, eigenvectors)
    """
    xp = get_array_module(a)
    result: tuple[Any, Any] = xp.linalg.eig(a)
    return result


def svd(a: Any, full_matrices: bool = True) -> tuple[Any, Any, Any]:
    """Singular Value Decomposition.

    Args:
        a: Input matrix
        full_matrices: If True, compute full-sized U and Vh

    Returns:
        Tuple of (U, S, Vh) where a = U @ diag(S) @ Vh
    """
    xp = get_array_module(a)
    result: tuple[Any, Any, Any] = xp.linalg.svd(a, full_matrices=full_matrices)
    return result


def cholesky(a: Any) -> Any:
    """Cholesky decomposition.

    Args:
        a: Symmetric positive-definite matrix

    Returns:
        Lower triangular matrix L where a = L @ L.T
    """
    xp = get_array_module(a)
    return xp.linalg.cholesky(a)


def inv(a: Any) -> Any:
    """Matrix inverse.

    Args:
        a: Square matrix

    Returns:
        Inverse matrix
    """
    xp = get_array_module(a)
    return xp.linalg.inv(a)


def det(a: Any) -> Any:
    """Matrix determinant.

    Args:
        a: Square matrix

    Returns:
        Determinant (scalar)
    """
    xp = get_array_module(a)
    return xp.linalg.det(a)


def norm(a: Any, order: Any | None = None, axis: Any | None = None) -> Any:
    """Matrix or vector norm.

    Args:
        a: Input array
        order: Order of the norm
        axis: Axis along which to compute norm

    Returns:
        Norm value(s)
    """
    xp = get_array_module(a)
    return xp.linalg.norm(a, ord=order, axis=axis)


def qr(a: Any, mode: str = "reduced") -> tuple[Any, Any]:
    """QR decomposition.

    Args:
        a: Input matrix
        mode: 'reduced' or 'complete'

    Returns:
        Tuple of (Q, R) where a = Q @ R
    """
    xp = get_array_module(a)
    result: tuple[Any, Any] = xp.linalg.qr(a, mode=mode)
    return result
