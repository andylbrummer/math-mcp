# scicomp-quantum-mcp

mcp-name: io.github.andylbrummer/quantum-mcp

MCP server for wave mechanics and Schrödinger equation simulations.

## Overview

This server provides tools for quantum mechanics simulations and wavefunction analysis:

- **Schrödinger solver** - Time-dependent wave equation simulations in 1D and 2D
- **Potential creation** - Crystalline lattices, custom potentials, harmonic oscillators
- **Wavefunction analysis** - Observable computation, probability density, energy analysis
- **Wave packets** - Gaussian wave packet creation and evolution
- **GPU acceleration** - Optional CUDA acceleration for large simulations

## Installation & Usage

```bash
# Run directly with uvx (no installation required)
uvx scicomp-quantum-mcp

# Or install with pip
pip install scicomp-quantum-mcp

# With GPU support
pip install scicomp-quantum-mcp[gpu]

# Run as command
scicomp-quantum-mcp
```

## Available Tools

### Potential Creation
- `create_lattice_potential` - Crystalline lattice potentials (square, hexagonal, triangular)
- `create_custom_potential` - Custom potential from mathematical function
- `create_gaussian_wavepacket` - Localized Gaussian wave packets
- `create_plane_wave` - Plane wave states

### Simulation
- `solve_schrodinger` - 1D time-dependent Schrödinger equation
- `solve_schrodinger_2d` - 2D time-dependent Schrödinger equation
- `get_task_status` - Monitor async simulations
- `get_simulation_result` - Retrieve completed simulation data

### Analysis
- `analyze_wavefunction` - Compute observables from wavefunction
- `visualize_potential` - Plot potential energy landscapes
- `render_video` - Animate probability density evolution

## Configuration

Enable GPU acceleration with environment variable:

```bash
MCP_USE_GPU=1 scicomp-quantum-mcp
```

## Examples

### 🎬 Visual Demos
Spectacular animated demonstrations:
- [Single-Slit Diffraction](https://andylbrummer.github.io/math-mcp/docs/demos/single-slit)
- [Double-Slit Interference](https://andylbrummer.github.io/math-mcp/docs/demos/double-slit)
- [Triple-Slit Interference](https://andylbrummer.github.io/math-mcp/docs/demos/triple-slit)
- [Bragg Scattering](https://andylbrummer.github.io/math-mcp/docs/demos/bragg-square)

Run demos with Claude:
```bash
claude -p "Simulate double-slit interference" \
  --allowedTools "mcp__quantum-mcp__*"
```

### 📖 Code Examples
Practical tutorials in [EXAMPLES.md](EXAMPLES.md):
- Particle in a box dynamics
- Double-slit interference patterns
- Quantum tunneling
- Crystal lattice scattering

### 📚 Full Documentation
See the [API documentation](https://andylbrummer.github.io/math-mcp/api/quantum-mcp) for complete reference.

## Part of Math-Physics-ML MCP System

Part of a comprehensive system for scientific computing. See the [documentation](https://andylbrummer.github.io/math-mcp/) for the complete ecosystem.
