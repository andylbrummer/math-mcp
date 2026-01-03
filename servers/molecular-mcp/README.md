# scicomp-molecular-mcp

mcp-name: io.github.andylbrummer/molecular-mcp

MCP server for classical molecular dynamics simulations.

## Overview

This server provides tools for molecular dynamics simulations with support for various ensembles and interaction potentials:

- **Particle systems** - Create and manage systems of interacting particles
- **Interaction potentials** - Lennard-Jones and Coulomb interactions
- **Ensembles** - NVE (microcanonical), NVT (canonical), NPT (isothermal-isobaric)
- **Analysis tools** - Radial distribution functions, mean squared displacement, phase transitions
- **Visualization** - Trajectory rendering and density field visualization
- **GPU acceleration** - Optional CUDA acceleration for large systems

## Installation & Usage

```bash
# Run directly with uvx (no installation required)
uvx scicomp-molecular-mcp

# Or install with pip
pip install scicomp-molecular-mcp

# With GPU support
pip install scicomp-molecular-mcp[gpu]

# Run as command
scicomp-molecular-mcp
```

## Available Tools

### System Setup
- `create_particles` - Initialize particle system with temperature
- `add_potential` - Add Lennard-Jones or Coulomb interactions

### Simulation
- `run_md` - NVE ensemble (constant energy, volume)
- `run_nvt` - NVT ensemble (constant temperature, volume)
- `run_npt` - NPT ensemble (constant temperature, pressure)
- `get_trajectory` - Retrieve simulation trajectory data

### Analysis
- `compute_rdf` - Radial distribution function analysis
- `compute_msd` - Mean squared displacement
- `analyze_temperature` - Thermodynamic properties
- `detect_phase_transition` - Identify phase transitions
- `density_field` - Compute density field visualization

### Visualization
- `render_trajectory` - Animate particle trajectories

## Configuration

Enable GPU acceleration with environment variable:

```bash
MCP_USE_GPU=1 scicomp-molecular-mcp
```

## Examples

### 🎬 Visual Demos
Spectacular animated demonstrations:
- [Galaxy Collision](https://andylbrummer.github.io/math-mcp/docs/demos/galaxy-collision) - N-body gravitational dynamics

Run demos with Claude:
```bash
claude -p "Simulate two galaxies colliding" \
  --allowedTools "mcp__molecular-mcp__*"
```

### 📖 Code Examples
Practical tutorials in [EXAMPLES.md](EXAMPLES.md):
- Simple liquid simulation with Lennard-Jones
- Temperature & pressure control
- Diffusion coefficient calculation
- Ionic systems with Coulomb interactions

### 📚 Full Documentation
See the [API documentation](https://andylbrummer.github.io/math-mcp/api/molecular-mcp) for complete reference.

## Part of Math-Physics-ML MCP System

Part of a comprehensive system for scientific computing. See the [documentation](https://andylbrummer.github.io/math-mcp/) for the complete ecosystem.
