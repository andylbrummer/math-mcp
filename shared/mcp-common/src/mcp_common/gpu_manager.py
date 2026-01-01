"""GPU Manager - Singleton for CUDA detection and memory management."""

import logging
from threading import Lock
from typing import Any, Optional

logger = logging.getLogger(__name__)


class GPUManager:
    """Singleton GPU manager for CUDA detection and memory pooling."""

    _instance: Optional["GPUManager"] = None
    _lock = Lock()

    def __init__(self) -> None:
        """Initialize GPU manager (use get_instance() instead)."""
        if GPUManager._instance is not None:
            msg = "Use GPUManager.get_instance() instead of direct instantiation"
            raise RuntimeError(msg)

        self._cuda_available = False
        self._device_count = 0
        self._memory_pool: Any | None = None
        self._gpu_memory_fraction = 0.8

        self._detect_cuda()

    @classmethod
    def get_instance(cls) -> "GPUManager":
        """Get singleton instance of GPUManager."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _detect_cuda(self) -> None:
        """Detect CUDA availability and initialize memory pool."""
        try:
            import cupy as cp  # noqa: PLC0415

            self._cuda_available = True
            self._device_count = cp.cuda.runtime.getDeviceCount()

            # Initialize memory pool with 80% of GPU memory
            pool = cp.cuda.MemoryPool()
            total_memory = cp.cuda.Device().mem_info[1]
            pool.set_limit(size=int(total_memory * self._gpu_memory_fraction))
            cp.cuda.set_allocator(pool.malloc)
            self._memory_pool = pool

            logger.info(
                "CUDA detected: %d device(s), memory pool: %.2f GB",
                self._device_count,
                total_memory * self._gpu_memory_fraction / 1e9,
            )
        except ImportError:
            logger.info("CuPy not available - GPU acceleration disabled")
        except Exception:
            logger.warning("CUDA detection failed - falling back to CPU", exc_info=True)

    @property
    def cuda_available(self) -> bool:
        """Check if CUDA is available."""
        return self._cuda_available

    @property
    def device_count(self) -> int:
        """Get number of CUDA devices."""
        return self._device_count

    def set_memory_fraction(self, fraction: float) -> None:
        """Set GPU memory pool fraction (0.0-1.0)."""
        if not 0.0 < fraction <= 1.0:
            msg = "Memory fraction must be between 0.0 and 1.0"
            raise ValueError(msg)

        self._gpu_memory_fraction = fraction

        if self._cuda_available and self._memory_pool is not None:
            import cupy as cp  # noqa: PLC0415

            total_memory = cp.cuda.Device().mem_info[1]
            self._memory_pool.set_limit(size=int(total_memory * fraction))
            logger.info("GPU memory pool updated to %.0f%%", fraction * 100)

    def get_memory_info(self) -> dict[str, int]:
        """Get current GPU memory usage."""
        if not self._cuda_available:
            return {"total": 0, "free": 0, "used": 0}

        try:
            import cupy as cp  # noqa: PLC0415

            free, total = cp.cuda.Device().mem_info
            return {"total": total, "free": free, "used": total - free}
        except Exception:
            logger.exception("Failed to get memory info")
            return {"total": 0, "free": 0, "used": 0}

    def clear_memory_pool(self) -> None:
        """Clear GPU memory pool."""
        if self._cuda_available and self._memory_pool is not None:
            self._memory_pool.free_all_blocks()
            logger.info("GPU memory pool cleared")
