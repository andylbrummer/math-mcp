---
sidebar_position: 1
---

# Installation

This guide will walk you through installing the Math-Physics-ML MCP system.

## Prerequisites

Before installing, ensure you have:

- **Python 3.11 or higher**
- **uv** package manager (recommended) or pip
- **NVIDIA GPU** with CUDA 12.x (optional, for GPU acceleration)
- **Git** for cloning the repository

## System Requirements

### Minimum Requirements
- Python 3.11+
- 8GB RAM
- 5GB disk space

### Recommended for GPU Acceleration
- NVIDIA GPU with 8GB+ VRAM
- CUDA Toolkit 12.x
- cuDNN 8.x

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/beagle/math-mcp.git
cd math-mcp
```

### 2. Install uv (if not already installed)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or use pip
pip install uv
```

### 3. Install Dependencies

```bash
# Install all dependencies (CPU-only by default)
uv sync

# For GPU support, install with GPU extras
uv sync --all-extras
```

This will:
- Create a virtual environment in `.venv/`
- Install all 4 MCP servers
- Install shared packages (mcp-common, compute-core)
- Install development dependencies

### 4. Verify Installation

```bash
# Activate the virtual environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# Check that MCP servers are installed
which math-mcp quantum-mcp molecular-mcp neural-mcp

# Test a server
math-mcp --version
```

## GPU Setup (Optional)

For GPU acceleration, you need to install CUDA support:

### Install CUDA Toolkit

Follow the official NVIDIA CUDA installation guide for your OS:
- [CUDA Downloads](https://developer.nvidia.com/cuda-downloads)

### Verify CUDA Installation

```bash
# Check CUDA version
nvcc --version

# Check GPU availability
nvidia-smi
```

### Install GPU Python Packages

```bash
# CuPy for GPU-accelerated arrays
uv pip install cupy-cuda12x

# Verify GPU support
python -c "import cupy; print(cupy.cuda.is_available())"
```

## Development Installation

For development work on the MCP servers:

```bash
# Install in editable mode
uv pip install -e shared/mcp-common -e shared/compute-core
uv pip install -e servers/math-mcp
uv pip install -e servers/quantum-mcp
uv pip install -e servers/molecular-mcp
uv pip install -e servers/neural-mcp

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

## Troubleshooting

### Python Version Issues

If you encounter Python version errors:

```bash
# Check Python version
python --version

# Install Python 3.11+ using your system package manager
# or use pyenv
pyenv install 3.11
pyenv local 3.11
```

### GPU Not Detected

If CUDA is installed but not detected:

```bash
# Check CUDA library path
echo $LD_LIBRARY_PATH

# Add CUDA to library path
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Reinstall CuPy
pip uninstall cupy
pip install cupy-cuda12x
```

### Import Errors

If you see module import errors:

```bash
# Reinstall dependencies
uv sync --reinstall

# Or reinstall specific package
uv pip install --force-reinstall mcp-common
```

## Next Steps

After installation:

1. **[Configure the MCP servers](configuration)** - Set up server configurations
2. **[Quick Start Guide](quick-start)** - Run your first computations
3. **[Architecture Overview](../concepts/architecture)** - Understand the system design

## Uninstallation

To remove the Math-Physics-ML MCP system:

```bash
# Remove virtual environment
rm -rf .venv

# Remove installed packages (if installed globally)
pip uninstall math-mcp quantum-mcp molecular-mcp neural-mcp
pip uninstall mcp-common compute-core
```
