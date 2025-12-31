# scicomp-compute-core

GPU-accelerated computation core for the Math-Physics-ML MCP system.

## Overview

This package provides high-performance array operations and numerical computing utilities with optional GPU acceleration using CuPy. It includes:

- **Array utilities** - Unified interface for NumPy and CuPy arrays
- **GPU acceleration** - Seamless CPU/GPU array conversion
- **Numerical operations** - Optimized mathematical functions
- **Linear algebra** - Matrix operations and decompositions
- **FFT operations** - Fast Fourier transforms with GPU support

## Installation

```bash
# CPU only
pip install scicomp-compute-core

# With GPU support (CUDA 12.x)
pip install scicomp-compute-core[gpu]
```

## Quick Start

```python
from compute_core.arrays import to_gpu, to_numpy
import numpy as np

# Create array on CPU
arr = np.array([1, 2, 3, 4, 5])

# Move to GPU (if available)
gpu_arr = to_gpu(arr)

# Perform GPU operations
result = gpu_arr * 2

# Move back to CPU
cpu_result = to_numpy(result)
```

## Features

- **Automatic device selection** - Works with or without GPU
- **Transparent acceleration** - Same API for CPU and GPU arrays
- **Type safety** - Full type hints for all functions
- **Performance** - Optimized for numerical computing workflows

## Part of Math-Physics-ML MCP System

This package is part of a larger system. See the [full documentation](https://andylbrummer.github.io/math-mcp/) for details on all available components.
