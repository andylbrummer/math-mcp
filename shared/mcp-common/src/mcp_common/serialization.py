"""Array serialization utilities - size-based strategy."""

import base64
import io
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Thresholds for serialization strategy
INLINE_THRESHOLD = 10 * 1024 * 1024  # 10MB
MEMORY_THRESHOLD = 100 * 1024 * 1024  # 100MB

# Cache directory for large arrays
ARRAY_CACHE_DIR = Path("/tmp/mcp-array-cache")
ARRAY_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def serialize_array(
    arr: Any,
    array_id: str,
    force_inline: bool = False,
) -> dict[str, Any]:
    """Serialize array using size-based strategy.

    Strategy:
    - Small (<10MB): Inline as base64-encoded bytes
    - Medium (10MB-100MB): Return array_id, cache in memory
    - Large (>100MB): Write to disk, return file path

    Args:
        arr: NumPy or CuPy array
        array_id: Unique identifier for the array
        force_inline: Force inline serialization regardless of size

    Returns:
        Serialization metadata with format, data, or reference
    """
    # Convert CuPy to NumPy for serialization
    arr_np = arr.get() if hasattr(arr, "get") else np.asarray(arr)

    arr_bytes = arr_np.nbytes
    shape = arr_np.shape
    dtype = str(arr_np.dtype)

    # Small arrays: inline serialization
    if force_inline or arr_bytes < INLINE_THRESHOLD:
        buffer = io.BytesIO()
        np.save(buffer, arr_np)
        data_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "format": "inline",
            "array_id": array_id,
            "shape": list(shape),
            "dtype": dtype,
            "size_bytes": arr_bytes,
            "data": data_b64,
        }

    # Large arrays: write to disk
    if arr_bytes >= MEMORY_THRESHOLD:
        file_path = ARRAY_CACHE_DIR / f"{array_id}.npy"
        np.save(file_path, arr_np)

        logger.info("Saved large array to disk: %s (%.2f MB)", file_path, arr_bytes / 1e6)

        return {
            "format": "disk",
            "array_id": array_id,
            "shape": list(shape),
            "dtype": dtype,
            "size_bytes": arr_bytes,
            "file_path": str(file_path),
        }

    # Medium arrays: return ID (cached in memory by caller)
    return {
        "format": "memory",
        "array_id": array_id,
        "shape": list(shape),
        "dtype": dtype,
        "size_bytes": arr_bytes,
    }


def deserialize_array(
    metadata: dict[str, Any],
    use_gpu: bool = False,
) -> Any:
    """Deserialize array from metadata.

    Args:
        metadata: Serialization metadata from serialize_array
        use_gpu: Load array on GPU if available

    Returns:
        NumPy or CuPy array
    """
    fmt = metadata["format"]

    if fmt == "inline":
        # Decode base64 and load
        data_b64 = metadata["data"]
        data_bytes = base64.b64decode(data_b64)
        buffer = io.BytesIO(data_bytes)
        arr = np.load(buffer)

    elif fmt == "disk":
        # Load from file
        file_path = Path(metadata["file_path"])
        if not file_path.exists():
            msg = f"Array file not found: {file_path}"
            raise FileNotFoundError(msg)
        arr = np.load(file_path)
        logger.info("Loaded array from disk: %s", file_path)

    elif fmt == "memory":
        # Array should be in caller's memory cache
        msg = "Cannot deserialize memory format - array must be in cache"
        raise ValueError(msg)

    else:
        msg = f"Unknown serialization format: {fmt}"
        raise ValueError(msg)

    # Transfer to GPU if requested
    if use_gpu:
        try:
            import cupy as cp  # noqa: PLC0415

            arr = cp.asarray(arr)
        except ImportError:
            logger.warning("CuPy not available, keeping array on CPU")

    return arr


def cleanup_array_cache(max_age_seconds: int = 3600) -> int:
    """Clean up old array files from disk cache."""
    import time  # noqa: PLC0415

    now = time.time()
    removed = 0

    for file_path in ARRAY_CACHE_DIR.glob("*.npy"):
        try:
            age = now - file_path.stat().st_mtime
            if age > max_age_seconds:
                file_path.unlink()
                removed += 1
        except Exception:
            logger.exception("Failed to remove %s", file_path)

    if removed > 0:
        logger.info("Cleaned up %d old array files", removed)

    return removed
