# scicomp-math-mcp

mcp-name: io.github.andylbrummer/math-mcp

MCP server for symbolic algebra and GPU-accelerated numerical computing.

## Overview

This server provides tools for mathematical computation combining symbolic algebra with numerical computing:

- **Symbolic mathematics** - Equations, simplification, differentiation, integration (via SymPy)
- **GPU-accelerated numerics** - Fast array operations and linear algebra
- **Mathematical transforms** - FFT, optimization, root finding
- **Linear systems** - Solving systems of equations with GPU acceleration

## Installation & Usage

```bash
# Run directly with uvx (no installation required)
uvx scicomp-math-mcp

# Or install with pip
pip install scicomp-math-mcp

# Then run as a command
scicomp-math-mcp
```

## Available Tools

### Symbolic Computation
- `symbolic_solve` - Solve symbolic equations
- `symbolic_diff` - Compute derivatives
- `symbolic_integrate` - Compute integrals
- `symbolic_simplify` - Simplify expressions

### Numerical Computing
- `create_array` - Create arrays with various patterns
- `matrix_multiply` - GPU-accelerated matrix multiplication
- `solve_linear_system` - Solve Ax = b
- `fft` / `ifft` - Fast Fourier transforms
- `optimize_function` - Function minimization
- `find_roots` - Find function roots

## Configuration

Set the `MCP_USE_GPU` environment variable to enable GPU acceleration:

```bash
MCP_USE_GPU=1 scicomp-math-mcp
```

## Examples

### 📖 Code Examples
Practical tutorials in [EXAMPLES.md](EXAMPLES.md):
- Projectile motion with symbolic + numerical computation
- Chemical equilibrium analysis
- Fourier signal analysis
- Circuit analysis with linear systems
- Progressive difficulty (beginner → advanced)

### 📚 Full Documentation
See the [API documentation](https://andylbrummer.github.io/math-mcp/api/math-mcp) for complete API reference.

## Part of Math-Physics-ML MCP System

Part of a comprehensive system for scientific computing. See the [documentation](https://andylbrummer.github.io/math-mcp/) for the complete ecosystem.
