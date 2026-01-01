"""Tests for array serialization."""

import numpy as np
import pytest
from mcp_common.serialization import deserialize_array, serialize_array

rng = np.random.default_rng()


def test_serialize_small_array_inline() -> None:
    """Test small array serialization (inline)."""
    arr = rng.random((100, 100))  # ~80KB
    array_id = "test-array-1"

    metadata = serialize_array(arr, array_id)

    assert metadata["format"] == "inline"
    assert metadata["array_id"] == array_id
    assert metadata["shape"] == [100, 100]
    assert metadata["dtype"] == "float64"
    assert "data" in metadata


def test_serialize_large_array_disk() -> None:
    """Test large array serialization (disk)."""
    # Create array >100MB
    arr = rng.random((2000, 2000, 7))  # ~224MB
    array_id = "test-array-large"

    metadata = serialize_array(arr, array_id)

    assert metadata["format"] == "disk"
    assert metadata["array_id"] == array_id
    assert "file_path" in metadata


def test_serialize_deserialize_roundtrip() -> None:
    """Test serialization-deserialization roundtrip."""
    arr = rng.random((50, 50))
    array_id = "test-roundtrip"

    # Serialize
    metadata = serialize_array(arr, array_id)

    # Deserialize
    arr_restored = deserialize_array(metadata)

    # Verify
    np.testing.assert_array_equal(arr, arr_restored)


def test_force_inline() -> None:
    """Test forcing inline serialization."""
    # Medium array that would normally use memory format
    arr = rng.random((1000, 1000, 15))  # ~120MB
    array_id = "test-force-inline"

    metadata = serialize_array(arr, array_id, force_inline=True)

    # Should be inline despite size
    assert metadata["format"] == "inline"


@pytest.mark.gpu
def test_serialize_cupy_array() -> None:
    """Test serializing CuPy array."""
    try:
        import cupy as cp  # noqa: PLC0415

        arr = cp.random.rand(100, 100)
        array_id = "test-cupy"

        metadata = serialize_array(arr, array_id)

        assert metadata["format"] == "inline"
        assert metadata["shape"] == [100, 100]

    except ImportError:
        pytest.skip("CuPy not available")


@pytest.mark.gpu
def test_deserialize_to_gpu() -> None:
    """Test deserializing to GPU."""
    try:
        import cupy as cp  # noqa: PLC0415

        arr = rng.random((50, 50))
        array_id = "test-gpu-deserialize"

        metadata = serialize_array(arr, array_id)
        arr_gpu = deserialize_array(metadata, use_gpu=True)

        assert hasattr(arr_gpu, "__cuda_array_interface__")
        np.testing.assert_array_almost_equal(arr, cp.asnumpy(arr_gpu))

    except ImportError:
        pytest.skip("CuPy not available")
