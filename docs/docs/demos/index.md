---
sidebar_position: 1
title: Visual Demos
---

# Visual Demonstrations

Dramatic visualizations showcasing the computational physics capabilities of the Math MCP ecosystem.

These demos demonstrate:
- **Quantum mechanics** simulations using the Schrödinger equation
- **Molecular dynamics** with N-body gravitational interactions
- Real-time visualization and video rendering

## Quantum Mechanics Demos

### Slit Diffraction Series

Experience the most famous experiments in quantum mechanics - wave interference through single, double, and triple slits.

| Demo | Description |
|------|-------------|
| [Single-Slit Diffraction](/demos/single-slit) | Classic diffraction pattern from a single narrow opening |
| [Double-Slit Interference](/demos/double-slit) | The quintessential quantum experiment showing wave-particle duality |
| [Triple-Slit Interference](/demos/triple-slit) | Complex interference patterns from three coherent sources |

### Bragg Scattering Series

Watch quantum wavepackets scatter from crystal lattice structures, revealing the atomic arrangement through diffraction patterns.

| Demo | Description |
|------|-------------|
| [Square Lattice](/demos/bragg-square) | Scattering from a simple cubic-like crystal structure |
| [Hexagonal Lattice](/demos/bragg-hexagonal) | Graphene-like honeycomb atomic arrangement |

## Molecular Dynamics Demos

### Galaxy Collision

Spectacular N-body gravitational simulation of colliding galaxies.

| Demo | Description |
|------|-------------|
| [Galaxy Collision](/demos/galaxy-collision) | Two spiral galaxies merge, creating tidal tails and bridges |

---

## Running the Demos

Run demos interactively with Claude Code:

```bash
claude -p "Simulate double-slit interference and save to /tmp/demo.gif" \
  --allowedTools "mcp__quantum-mcp__*"
```

Or start an interactive session:

```bash
cd /path/to/math-mcp
claude
# Then ask: "Simulate a galaxy collision"
```

---

## The Physics Behind the Demos

### Quantum Simulations

All quantum demos use the **split-step Fourier method** to solve the time-dependent Schrödinger equation. The algorithm alternates between:
1. **Position space**: Apply potential energy operator
2. **Momentum space**: Apply kinetic energy operator (via FFT)

This preserves unitarity and handles both bound and scattering states.

### Molecular Dynamics

The galaxy collision uses **Velocity Verlet integration** with gravitational softening. The softening parameter prevents numerical instabilities when particles pass close together.
