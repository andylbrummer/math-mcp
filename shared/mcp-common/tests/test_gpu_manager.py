"""Tests for GPUManager."""

import pytest
from mcp_common.gpu_manager import GPUManager


def test_gpu_manager_singleton() -> None:
    """Test GPUManager is a singleton."""
    gpu1 = GPUManager.get_instance()
    gpu2 = GPUManager.get_instance()
    assert gpu1 is gpu2


def test_gpu_manager_properties() -> None:
    """Test GPUManager basic properties."""
    gpu = GPUManager.get_instance()

    # Should not raise
    _ = gpu.cuda_available
    _ = gpu.device_count

    assert isinstance(gpu.cuda_available, bool)
    assert isinstance(gpu.device_count, int)
    assert gpu.device_count >= 0


def test_gpu_memory_info() -> None:
    """Test memory info retrieval."""
    gpu = GPUManager.get_instance()
    info = gpu.get_memory_info()

    assert isinstance(info, dict)
    assert "total" in info
    assert "free" in info
    assert "used" in info


def test_set_memory_fraction() -> None:
    """Test setting memory fraction."""
    gpu = GPUManager.get_instance()

    # Valid fraction
    gpu.set_memory_fraction(0.5)

    # Invalid fractions
    with pytest.raises(ValueError, match="Memory fraction must be"):
        gpu.set_memory_fraction(0.0)

    with pytest.raises(ValueError, match="Memory fraction must be"):
        gpu.set_memory_fraction(1.5)


def test_clear_memory_pool() -> None:
    """Test clearing memory pool."""
    gpu = GPUManager.get_instance()
    # Should not raise
    gpu.clear_memory_pool()
