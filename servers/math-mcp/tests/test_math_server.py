"""Tests for Math MCP server."""

import ast

import numpy as np
import pytest

# Import tools directly for testing
from math_mcp.server import (
    _array_cache,
    _tool_create_array,
    _tool_fft,
    _tool_find_roots,
    _tool_ifft,
    _tool_info,
    _tool_matrix_multiply,
    _tool_optimize_function,
    _tool_solve_linear_system,
    _tool_symbolic_diff,
    _tool_symbolic_integrate,
    _tool_symbolic_simplify,
    _tool_symbolic_solve,
)


@pytest.mark.asyncio
async def test_info_overview() -> None:
    """Test info tool with overview."""
    result = await _tool_info({"topic": "overview"})
    assert len(result) == 1
    assert "symbolic" in str(result[0]["text"])


@pytest.mark.asyncio
async def test_info_category() -> None:
    """Test info tool with category."""
    result = await _tool_info({"topic": "symbolic"})
    assert len(result) == 1
    assert "symbolic_solve" in str(result[0]["text"])


@pytest.mark.asyncio
async def test_symbolic_solve_quadratic() -> None:
    """Test solving quadratic equation."""
    result = await _tool_symbolic_solve({"equations": "x**2 - 4"})
    assert len(result) == 1
    text = str(result[0]["text"])
    assert "solutions" in text
    # Should find x = -2 and x = 2
    assert "2" in text or "-2" in text


@pytest.mark.asyncio
async def test_symbolic_solve_system() -> None:
    """Test solving system of equations."""
    result = await _tool_symbolic_solve(
        {"equations": ["x + y = 5", "x - y = 1"], "variables": ["x", "y"]}
    )
    assert len(result) == 1
    text = str(result[0]["text"])
    assert "solutions" in text


@pytest.mark.asyncio
async def test_symbolic_diff() -> None:
    """Test symbolic differentiation."""
    result = await _tool_symbolic_diff({"expression": "x**2 + 2*x + 1", "variable": "x"})
    assert len(result) == 1
    text = str(result[0]["text"])
    assert "derivative" in text
    # d/dx(x^2 + 2x + 1) = 2x + 2
    assert "2*x" in text or "2x" in text


@pytest.mark.asyncio
async def test_symbolic_diff_second_order() -> None:
    """Test second-order derivative."""
    result = await _tool_symbolic_diff({"expression": "x**3", "variable": "x", "order": 2})
    assert len(result) == 1
    text = str(result[0]["text"])
    # d²/dx²(x^3) = 6x
    assert "6*x" in text or "6x" in text


@pytest.mark.asyncio
async def test_symbolic_integrate_indefinite() -> None:
    """Test indefinite integration."""
    result = await _tool_symbolic_integrate({"expression": "x**2", "variable": "x"})
    assert len(result) == 1
    text = str(result[0]["text"])
    assert "integral" in text
    assert "indefinite" in text


@pytest.mark.asyncio
async def test_symbolic_integrate_definite() -> None:
    """Test definite integration."""
    result = await _tool_symbolic_integrate(
        {"expression": "x**2", "variable": "x", "limits": [0, 1]}
    )
    assert len(result) == 1
    text = str(result[0]["text"])
    assert "definite" in text


@pytest.mark.asyncio
async def test_symbolic_simplify() -> None:
    """Test expression simplification."""
    result = await _tool_symbolic_simplify({"expression": "x**2 - 2*x*y + y**2"})
    assert len(result) == 1
    text = str(result[0]["text"])
    assert "simplified" in text


@pytest.mark.asyncio
async def test_create_array_zeros() -> None:
    """Test creating zero array."""
    result = await _tool_create_array({"shape": [10, 10], "fill_type": "zeros"})
    assert len(result) == 1
    text = str(result[0]["text"])
    assert "array_id" in text
    assert "array://" in text


@pytest.mark.asyncio
async def test_create_array_random() -> None:
    """Test creating random array."""
    result = await _tool_create_array({"shape": [5, 5], "fill_type": "random"})
    assert len(result) == 1
    text = str(result[0]["text"])
    assert "array_id" in text


@pytest.mark.asyncio
async def test_create_array_linspace() -> None:
    """Test creating linspace array."""
    result = await _tool_create_array(
        {"shape": [100], "fill_type": "linspace", "linspace_range": [0, 10]}
    )
    assert len(result) == 1
    text = str(result[0]["text"])
    assert "array_id" in text


@pytest.mark.asyncio
async def test_create_array_function() -> None:
    """Test creating array from function."""
    result = await _tool_create_array(
        {"shape": [10, 10], "fill_type": "function", "function": "x + y"}
    )
    assert len(result) == 1
    text = str(result[0]["text"])
    assert "array_id" in text


@pytest.mark.asyncio
async def test_matrix_multiply() -> None:
    """Test matrix multiplication."""
    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    result = await _tool_matrix_multiply({"a": a, "b": b})
    assert len(result) == 1
    text = str(result[0]["text"])
    assert "array_id" in text


@pytest.mark.asyncio
async def test_solve_linear_system() -> None:
    """Test solving linear system."""
    a = [[3, 1], [1, 2]]
    b = [9, 8]
    result = await _tool_solve_linear_system({"a": a, "b": b})
    assert len(result) == 1
    text = str(result[0]["text"])
    assert "solution" in text


@pytest.mark.asyncio
async def test_fft_direct_array() -> None:
    """Test FFT with direct array input."""
    arr = [1, 2, 3, 4, 5]
    result = await _tool_fft({"array": arr})
    assert len(result) == 1
    text = str(result[0]["text"])
    assert "array_id" in text


@pytest.mark.asyncio
async def test_fft_ifft_roundtrip() -> None:
    """Test FFT-IFFT roundtrip."""
    # Create array
    create_result = await _tool_create_array({"shape": [100], "fill_type": "random"})
    text = str(create_result[0]["text"])
    data = ast.literal_eval(text)
    array_id = data["array_id"]

    # FFT
    fft_result = await _tool_fft({"array": array_id})
    fft_text = str(fft_result[0]["text"])
    fft_data = ast.literal_eval(fft_text)
    fft_array_id = fft_data["array_id"]

    # IFFT
    ifft_result = await _tool_ifft({"array": fft_array_id})
    assert len(ifft_result) == 1
    assert "array_id" in str(ifft_result[0]["text"])


@pytest.mark.asyncio
async def test_optimize_function() -> None:
    """Test function optimization."""
    # Minimize (x-2)^2, minimum at x=2
    result = await _tool_optimize_function(
        {"function": "(x - 2)**2", "variables": ["x"], "initial_guess": [0]}
    )
    assert len(result) == 1
    text = str(result[0]["text"])
    assert "success" in text
    # Solution should be near x=2
    assert "x" in text


@pytest.mark.asyncio
async def test_find_roots() -> None:
    """Test root finding."""
    # Find root of x^2 - 4 = 0, roots at x=±2
    result = await _tool_find_roots(
        {"function": "x**2 - 4", "variables": ["x"], "initial_guess": [1]}
    )
    assert len(result) == 1
    text = str(result[0]["text"])
    assert "roots" in text


@pytest.mark.asyncio
async def test_array_cache() -> None:
    """Test that arrays are cached."""
    initial_cache_size = len(_array_cache)
    await _tool_create_array({"shape": [5, 5], "fill_type": "zeros"})
    assert len(_array_cache) == initial_cache_size + 1


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_create_array_gpu() -> None:
    """Test creating array on GPU."""
    result = await _tool_create_array({"shape": [100, 100], "fill_type": "random", "use_gpu": True})
    assert len(result) == 1
    text = str(result[0]["text"])
    assert "array_id" in text


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_matrix_multiply_gpu() -> None:
    """Test matrix multiplication on GPU."""
    rng = np.random.default_rng()
    a = rng.random((100, 100)).tolist()
    b = rng.random((100, 100)).tolist()
    result = await _tool_matrix_multiply({"a": a, "b": b, "use_gpu": True})
    assert len(result) == 1
    text = str(result[0]["text"])
    assert "array_id" in text
