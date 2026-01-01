"""Tests for linear algebra operations."""

import numpy as np
import pytest
from compute_core.linalg import cholesky, det, eig, inv, matmul, norm, solve, svd

rng = np.random.default_rng()


def test_matmul() -> None:
    """Test matrix multiplication."""
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    result = matmul(a, b)
    expected = np.matmul(a, b)
    np.testing.assert_array_equal(result, expected)


def test_solve_linear_system() -> None:
    """Test solving linear system."""
    a = np.array([[3, 1], [1, 2]], dtype=float)
    b = np.array([9, 8], dtype=float)
    x = solve(a, b)

    # Verify Ax = b
    np.testing.assert_array_almost_equal(np.matmul(a, x), b)


def test_eig_symmetric() -> None:
    """Test eigenvalue decomposition."""
    # Symmetric matrix for real eigenvalues
    a = np.array([[2, 1], [1, 2]], dtype=float)
    eigenvalues, eigenvectors = eig(a)

    # Verify A v = λ v
    for i in range(len(eigenvalues)):
        lhs = np.matmul(a, eigenvectors[:, i])
        rhs = eigenvalues[i] * eigenvectors[:, i]
        np.testing.assert_array_almost_equal(lhs, rhs)


def test_svd_decomposition() -> None:
    """Test SVD decomposition."""
    a = rng.random((5, 3))
    u, s, vh = svd(a, full_matrices=False)

    # Reconstruct matrix
    reconstructed = u @ np.diag(s) @ vh
    np.testing.assert_array_almost_equal(a, reconstructed)


def test_cholesky_decomposition() -> None:
    """Test Cholesky decomposition."""
    # Create symmetric positive-definite matrix
    a = np.array([[4, 2], [2, 3]], dtype=float)
    lower = cholesky(a)

    # Verify A = L @ L.T
    reconstructed = lower @ lower.T
    np.testing.assert_array_almost_equal(a, reconstructed)


def test_matrix_inverse() -> None:
    """Test matrix inversion."""
    a = np.array([[1, 2], [3, 4]], dtype=float)
    a_inv = inv(a)

    # Verify A @ A^-1 = I
    identity = matmul(a, a_inv)
    np.testing.assert_array_almost_equal(identity, np.eye(2))


def test_determinant() -> None:
    """Test determinant calculation."""
    a = np.array([[1, 2], [3, 4]], dtype=float)
    result = det(a)
    expected = np.linalg.det(a)
    np.testing.assert_almost_equal(result, expected)


def test_vector_norm() -> None:
    """Test vector norm."""
    v = np.array([3, 4])
    result = norm(v)
    expected = np.linalg.norm(v)
    np.testing.assert_almost_equal(result, expected)


def test_matrix_norm() -> None:
    """Test matrix norm."""
    a = np.array([[1, 2], [3, 4]], dtype=float)
    result = norm(a)
    expected = np.linalg.norm(a)
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.gpu
def test_matmul_gpu() -> None:
    """Test matrix multiplication on GPU."""
    try:
        import cupy as cp  # noqa: PLC0415
        from compute_core.arrays import to_gpu  # noqa: PLC0415

        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])

        a_gpu = to_gpu(a)
        b_gpu = to_gpu(b)

        result = matmul(a_gpu, b_gpu)
        assert hasattr(result, "__cuda_array_interface__")

        expected = np.matmul(a, b)
        np.testing.assert_array_equal(cp.asnumpy(result), expected)
    except ImportError:
        pytest.skip("CuPy not available")


@pytest.mark.gpu
def test_solve_gpu() -> None:
    """Test solving linear system on GPU."""
    try:
        import cupy as cp  # noqa: PLC0415
        from compute_core.arrays import to_gpu  # noqa: PLC0415

        a = np.array([[3, 1], [1, 2]], dtype=float)
        b = np.array([9, 8], dtype=float)

        a_gpu = to_gpu(a)
        b_gpu = to_gpu(b)

        x = solve(a_gpu, b_gpu)
        assert hasattr(x, "__cuda_array_interface__")

        # Verify Ax = b
        ax = cp.matmul(a_gpu, x)
        np.testing.assert_array_almost_equal(cp.asnumpy(ax), b)
    except ImportError:
        pytest.skip("CuPy not available")
