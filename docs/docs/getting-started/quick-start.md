---
sidebar_position: 1
---

# Quick Start

Get started with the Math-Physics-ML MCP servers in minutes. This guide shows you what each server can do and how to use them through Claude.

:::info
**These are MCP tools, not Python functions**

The examples below show MCP tool calls that Claude will invoke when you ask for them. You don't write this code yourself—you describe what you want in natural language, and Claude calls the appropriate tools.

**To use these servers:** Start Claude Code or Claude Desktop and ask for what you want!
:::

## How It Works

```bash
# Just ask Claude naturally:
claude
> "Solve x² + 5x + 6 = 0 and show the steps"
> "Simulate a particle in a 1D box"
> "Train a neural network on MNIST"
```

Claude will automatically call the right tools and return results.

---

## What Each Server Does

| Server | What It Provides |
|--------|-----------------|
| **Math MCP** | Symbolic algebra, arrays, FFT, optimization, linear algebra |
| **Quantum MCP** | Quantum potentials, wavefunctions, Schrödinger equation |
| **Molecular MCP** | Particle systems, molecular dynamics, trajectory analysis |
| **Neural MCP** | Neural networks, datasets, training, evaluation |

---

## Example Conversations

### Math MCP

```
You: "Find when a projectile hits the ground: h(t) = 100 + 50t - 4.9t²"

Claude: [calls symbolic_solve]
→ The projectile hits the ground at t = 11.3 seconds (positive root)

You: "Compute the FFT of a signal with 5Hz and 10Hz components"

Claude: [calls create_array + fft]
→ Returns frequency spectrum showing peaks at 5Hz and 10Hz
```

### Quantum MCP

```
You: "Create a double-slit potential and show the interference pattern"

Claude: [calls create_custom_potential + create_gaussian_wavepacket + solve_schrodinger_2d + render_video]
→ Returns animation showing wave diffraction through two slits
```

### Molecular MCP

```
You: "Simulate 1000 argon atoms at 300K"

Claude: [calls create_particles + add_potential + run_nvt]
→ Returns trajectory data and radial distribution function
```

### Neural MCP

```
You: "Train ResNet18 on CIFAR-10 for 5 epochs"

Claude: [calls load_dataset + define_model + train_model + evaluate_model]
→ Returns trained model with ~85% accuracy
```

---

## Available Tools Reference

### Math MCP Tools

**Symbolic Computation:**
- `symbolic_solve` - Solve equations symbolically
- `symbolic_diff` - Compute derivatives
- `symbolic_integrate` - Compute integrals
- `symbolic_simplify` - Simplify expressions

**Numerical Computing:**
- `create_array` - Create arrays (random, linspace, function)
- `matrix_multiply` - GPU-accelerated matrix multiplication
- `solve_linear_system` - Solve Ax = b
- `fft` / `ifft` - Fast Fourier transforms
- `optimize_function` - Find function minima
- `find_roots` - Find equation roots

### Quantum MCP Tools

**Potentials:**
- `create_lattice_potential` - Crystal lattice (square, hexagonal, triangular)
- `create_custom_potential` - Custom potential from function
- `create_gaussian_wavepacket` - Localized quantum state
- `create_plane_wave` - Plane wave state

**Simulation:**
- `solve_schrodinger` - 1D time-dependent Schrödinger equation
- `solve_schrodinger_2d` - 2D time-dependent Schrödinger equation

**Analysis:**
- `analyze_wavefunction` - Compute observables
- `visualize_potential` - Plot potential landscape
- `render_video` - Animate probability density

### Molecular MCP Tools

**Setup:**
- `create_particles` - Initialize N-particle system
- `add_potential` - Add Lennard-Jones or Coulomb interactions

**Simulation:**
- `run_md` - NVE (constant energy)
- `run_nvt` - NVT (constant temperature)
- `run_npt` - NPT (constant pressure)

**Analysis:**
- `compute_rdf` - Radial distribution function
- `compute_msd` - Mean squared displacement
- `analyze_temperature` - Thermodynamic properties
- `render_trajectory` - Create animation

### Neural MCP Tools

**Models:**
- `define_model` - Create architecture (resnet18, mobilenet, custom)
- `load_pretrained` - Load pretrained models

**Data:**
- `load_dataset` - Load CIFAR-10, MNIST, ImageNet
- `create_dataloader` - Batch data with shuffling

**Training:**
- `train_model` - Train with configurable parameters
- `get_experiment_status` - Monitor progress
- `evaluate_model` - Test set evaluation

**Analysis:**
- `compute_metrics` - Precision, recall, F1
- `confusion_matrix` - Classification errors
- `visualize_predictions` - Sample predictions
- `tune_hyperparameters` - Hyperparameter search
- `export_model` - Save to ONNX/TorchScript

---

## GPU Acceleration

For large-scale computations, enable GPU acceleration:

```bash
# Set environment variable before starting Claude
export MCP_USE_GPU=1
claude
```

GPU provides significant speedups for:
- Large matrix operations (Math MCP)
- 2D wavefunction simulations (Quantum MCP)
- Long MD trajectories (Molecular MCP)
- Neural network training (Neural MCP)

---

## Cross-MCP Workflows

MCPs can share data through resource URIs:

```
You: "Create a potential array in Math MCP, then use it in Quantum MCP"

Claude:
1. Calls create_array → Returns resource URI like "array://uuid-1234"
2. Calls solve_schrodinger(potential="array://uuid-1234", ...)
→ Claude handles the resource sharing automatically!
```

You don't need to manage URIs yourself—Claude handles all resource passing between MCPs.

---

## Interactive Demos

For spectacular visual demonstrations, try:

- [Single-Slit Diffraction](../demos/single-slit) - Quantum wave diffraction
- [Double-Slit Interference](../demos/double-slit) - Classic quantum experiment
- [Galaxy Collision](../demos/galaxy-collision) - N-body gravitational dynamics

Run demos with Claude:
```bash
claude -p "Simulate double-slit interference" --allowedTools "mcp__quantum-mcp__*"
```

---

## Next Steps

- 📖 **Examples**: Practical tutorials in each server's [EXAMPLES.md](https://github.com/andylbrummer/math-mcp/tree/main/servers/*/EXAMPLES.md)
- 🎬 **Visual Demos**: [Interactive demonstrations](../demos/index)
- 📚 **API Reference**: [Complete tool documentation](../api/overview)
