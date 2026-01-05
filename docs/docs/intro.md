---
sidebar_position: 1
slug: /
---

# Math-Physics-ML MCP System

Welcome to the **Math-Physics-ML MCP System** documentation!

The Math-Physics-ML MCP System is a comprehensive, GPU-accelerated platform providing Model Context Protocol (MCP) servers for computational mathematics, physics simulations, and machine learning.

## Overview

This system consists of **5 specialized MCP servers** that work together to provide a complete computational science platform:

### 🧮 Math MCP
Symbolic algebra (SymPy) and GPU-accelerated numerical computing
- 14 tools for symbolic math, linear algebra, FFT, and optimization
- Foundation layer used by other MCPs

### ⚛️ Quantum MCP
Wave mechanics and Schrödinger equation simulations
- 12 tools for quantum simulations
- Split-step Fourier solver for time-dependent wave evolution
- Support for 1D and 2D systems

### 🔬 Molecular MCP
Classical molecular dynamics simulations
- 15 tools for particle systems and MD simulations
- NVE, NVT, and NPT ensemble support
- Analysis tools for RDF, MSD, and phase transitions

### 🧠 Neural MCP
Neural network training and experimentation
- 16 tools for deep learning workflows
- Pre-built architectures (ResNet, MobileNet) and custom models
- Hyperparameter tuning and evaluation

### 📝 LLM MCP
Language model training, fine-tuning, and experimentation
- 33 tools for LLM research and development
- GPT (Transformer) and Mamba (State Space Model) architectures
- Training, evaluation, attention analysis, and text generation

## Key Features

- **GPU Acceleration**: Automatic CUDA detection with graceful CPU fallback (10-100x speedup)
- **Async Tasks**: Long-running operations use MCP Tasks primitive
- **Token Efficiency**: URI-based references for large arrays and simulation data
- **Cross-MCP Workflows**: Servers can exchange data and build on each other
- **Progressive Discovery**: Built-in `info` tools for capability exploration
- **Comprehensive Testing**: 75+ tests covering all functionality

## Quick Start

Get started with the Math-Physics-ML MCP system:

1. **[Installation](getting-started/installation)** - Set up the system and dependencies
2. **[Configuration](getting-started/configuration)** - Configure the MCP servers
3. **[Quick Start Guide](getting-started/quick-start)** - Run your first computations

## Architecture

The system is built as a monorepo with shared infrastructure:

```
math-mcp/
├── servers/
│   ├── math-mcp/          # Symbolic & numerical computing
│   ├── quantum-mcp/       # Quantum mechanics
│   ├── molecular-mcp/     # Molecular dynamics
│   ├── neural-mcp/        # Machine learning
│   └── llm-mcp/           # Language model training
├── shared/
│   ├── mcp-common/        # GPU manager, async tasks, config
│   └── compute-core/      # Unified array interface, FFT, linalg
```

Learn more in the [Architecture](concepts/architecture) guide.

## API Reference

Explore the complete API documentation for each MCP server:

- **[Math MCP API](api/math-mcp)** - Symbolic algebra and numerical computing
- **[Quantum MCP API](api/quantum-mcp)** - Wave mechanics and simulations
- **[Molecular MCP API](api/molecular-mcp)** - Molecular dynamics
- **[Neural MCP API](api/neural-mcp)** - Neural network training
- **[LLM MCP API](api/llm-mcp)** - Language model training and fine-tuning

## Example: Quantum Wave Scattering

Here's a simple example combining Math MCP and Quantum MCP:

```python
# Create a Gaussian potential barrier (Math MCP)
potential = create_array(
    shape=[256],
    fill_type='function',
    function='10*exp(-(x-128)**2/100)'
)

# Create a Gaussian wave packet (Quantum MCP)
wavepacket = create_gaussian_wavepacket(
    grid_size=[256],
    position=[64],
    momentum=[2.0],
    width=5.0
)

# Solve Schrödinger equation (Quantum MCP)
simulation = solve_schrodinger(
    potential=potential['array_id'],
    initial_state=wavepacket,
    time_steps=1000,
    dt=0.1,
    use_gpu=True
)
```

## Performance

The GPU acceleration provides significant speedups:

| MCP | Operation | CPU Time | GPU Time | Speedup |
|-----|-----------|----------|----------|---------|
| Math | Matrix multiply (1000×1000) | ~100ms | ~1ms | 100x |
| Quantum | 1D Schrödinger (1000 steps, 256 grid) | ~30s | ~5s | 6x |
| Molecular | MD (100k steps) | ~minutes | ~seconds | >10x |

## Status

All 5 MCP servers are fully implemented and tested:
- ✅ 90+ tests passing
- ✅ GPU and CPU modes supported
- ✅ Pre-commit hooks configured
- ✅ Ready for production use

## License

MIT License - see LICENSE file for details.
