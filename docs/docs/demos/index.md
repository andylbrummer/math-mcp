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
| [Single-Slit Diffraction](./single-slit) | Classic diffraction pattern from a single narrow opening |
| [Double-Slit Interference](./double-slit) | The quintessential quantum experiment showing wave-particle duality |
| [Triple-Slit Interference](./triple-slit) | Complex interference patterns from three coherent sources |

### Bragg Scattering Series

Watch quantum wavepackets scatter from crystal lattice structures, revealing the atomic arrangement through diffraction patterns.

| Demo | Description |
|------|-------------|
| [Square Lattice](./bragg-square) | Scattering from a simple cubic-like crystal structure |
| [Hexagonal Lattice](./bragg-hexagonal) | Graphene-like honeycomb atomic arrangement |
| [Triangular Lattice](./bragg-triangular) | Close-packed 2D crystal structure |

## Molecular Dynamics Demos

### Galaxy Collision

Spectacular N-body gravitational simulation of colliding galaxies.

| Demo | Description |
|------|-------------|
| [Galaxy Collision](./galaxy-collision) | Two spiral galaxies merge, creating tidal tails and bridges |

---

## Running the Demos

All demos are self-contained Python scripts that can be run directly:

```bash
# Quantum slit demos
python demos/quantum/scripts/slit_diffraction.py --demo all

# Bragg scattering demos
python demos/quantum/scripts/bragg_scattering.py --demo all

# Galaxy collision demo
python demos/molecular/scripts/galaxy_collision.py --demo main
```

### Requirements

```bash
pip install numpy matplotlib scipy
# For video export:
pip install ffmpeg-python  # or install ffmpeg system-wide
```

### Output

Demos generate:
- **Static images** in `demos/*/images/`
- **Animated videos** in `demos/*/videos/`

---

## The Physics Behind the Demos

### Quantum Simulations

All quantum demos use the **split-step Fourier method** to solve the time-dependent Schrödinger equation:

$$
i\hbar \frac{\partial \psi}{\partial t} = \left( -\frac{\hbar^2}{2m}\nabla^2 + V \right) \psi
$$

The algorithm alternates between:
1. **Position space**: Apply potential energy operator
2. **Momentum space**: Apply kinetic energy operator (via FFT)

This preserves unitarity and handles both bound and scattering states.

### Molecular Dynamics

The galaxy collision uses **Velocity Verlet integration** with gravitational softening:

$$
\mathbf{a}_i = \sum_{j \neq i} \frac{G m_j (\mathbf{r}_j - \mathbf{r}_i)}{(|\mathbf{r}_j - \mathbf{r}_i|^2 + \epsilon^2)^{3/2}}
$$

The softening parameter $\epsilon$ prevents numerical instabilities when particles pass close together.
