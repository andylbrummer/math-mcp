---
sidebar_position: 1
---

# Quick Start

Get started with the Math-Physics-ML MCP servers in minutes.

:::info
**These are MCP tools, not Python functions**

You don't write code. You describe what you want in natural language, and Claude calls the appropriate tools automatically.

**Just start Claude and ask for what you want!**
:::

## How It Works

```bash
claude
> "Analyze this sensor data and find the dominant frequencies"
> "Simulate quantum tunneling through a barrier"
> "Train a neural network on my dataset"
```

Claude invokes the right tools and returns results—no code required.

---

## What Each Server Does

| Server | What It Provides |
|--------|-----------------|
| **Math MCP** | Symbolic algebra, numerical arrays, FFT, optimization, linear algebra |
| **Quantum MCP** | Quantum potentials, wavefunctions, Schrödinger equation solving |
| **Molecular MCP** | Particle systems, molecular dynamics, trajectory analysis |
| **Neural MCP** | Neural networks, datasets, training, evaluation |

---

## Example: FFT Spectral Analysis

**Real problem:** Robot accelerometer data shows potential bearing wear. Find the dominant vibration frequencies.

```
You: "I have sensor data sampled at 1000 Hz. Create an array with these values and find the frequency spectrum: [0.1, 0.3, 0.5, 0.2, ...]"

Claude: [calls create_array with the data, then fft]
→ Returns frequency spectrum showing peaks at 15Hz, 45Hz, and 127Hz

You: "Which frequencies are above the normal threshold of 50Hz?"

Claude: The frequencies above 50Hz are 127Hz and its harmonics.
This may indicate bearing defect frequency - recommend inspection.
```

This is something Claude alone cannot do—it needs actual numerical computation.

---

## Example: Quantum Tunneling

**Real problem:** Estimate tunneling probability for an electron through a potential barrier.

```
You: "Simulate a Gaussian wavepacket with momentum 2.0 hitting a 10-unit barrier from x=50 to x=70. What fraction tunnels through?"

Claude: [calls create_custom_potential + create_gaussian_wavepacket + solve_schrodinger]
→ Returns ~23% transmission probability

You: "At what barrier height would tunneling drop below 1%?"

Claude: [runs parameter sweep]
→ Barrier height of ~15 units would reduce tunneling to <1%
```

Claude can't simulate quantum mechanics without these tools.

---

## Example: Molecular Dynamics

**Real problem:** Simulate how a protein folds under thermal fluctuations.

```
You: "Create a 500-atom system at 310K and run NVT dynamics for 5000 steps"

Claude: [calls create_particles + add_potential + run_nvt]
→ Returns trajectory and analyzes structural properties

You: "Compute the radius of gyration over time"

Claude: [calls analyze_trajectory]
→ Returns time series showing protein compactness
```

LLMs cannot run molecular dynamics simulations.

---

## Example: Neural Network Training

**Real problem:** Train a classifier on a custom dataset.

```
You: "Load my image dataset from /data/images and train a ResNet for 10 epochs"

Claude: [calls load_dataset + define_model + train_model]
→ Returns trained model with training curves

You: "What's the accuracy on the test set?"

Claude: [calls evaluate_model]
→ 94.2% accuracy with confusion matrix showing main error sources
```

Claude can write PyTorch code, but it often has bugs. MCP tools run actual training correctly.

---

## Available Tools

### Math MCP

| Tool | Purpose |
|------|---------|
| `symbolic_solve` | Solve equations symbolically |
| `symbolic_diff` | Compute derivatives |
| `symbolic_integrate` | Compute integrals |
| `create_array` | Create numerical arrays |
| `matrix_multiply` | GPU matrix multiplication |
| `solve_linear_system` | Solve Ax = b |
| `fft` / `ifft` | Fourier transforms |
| `optimize_function` | Find minima |
| `find_roots` | Find equation roots |

### Quantum MCP

| Tool | Purpose |
|------|---------|
| `create_lattice_potential` | Crystal lattices |
| `create_custom_potential` | Custom potentials |
| `create_gaussian_wavepacket` | Quantum states |
| `solve_schrodinger` | 1D Schrödinger equation |
| `solve_schrodinger_2d` | 2D Schrödinger equation |
| `render_video` | Animate wavefunction |

### Molecular MCP

| Tool | Purpose |
|------|---------|
| `create_particles` | Initialize N-particle system |
| `add_potential` | Add LJ/Coulomb forces |
| `run_md` | NVE dynamics |
| `run_nvt` | NVT dynamics |
| `run_npt` | NPT dynamics |
| `compute_rdf` | Radial distribution |
| `render_trajectory` | Animate trajectory |

### Neural MCP

| Tool | Purpose |
|------|---------|
| `define_model` | Create architecture |
| `load_pretrained` | Load pretrained models |
| `load_dataset` | Load CIFAR-10, MNIST |
| `train_model` | Train neural network |
| `evaluate_model` | Test set evaluation |
| `tune_hyperparameters` | Hyperparameter search |
| `export_model` | Export to ONNX |

---

## GPU Acceleration

For large computations, enable GPU:

```bash
export MCP_USE_GPU=1
claude
```

GPU provides 10-100x speedup for:
- Large matrix operations
- 2D quantum simulations
- Long MD trajectories
- Neural network training

---

## Interactive Demos

Watch spectacular visualizations:

- [Single-Slit Diffraction](/demos/single-slit) - Quantum diffraction
- [Double-Slit Interference](/demos/double-slit) - Interference patterns
- [Galaxy Collision](/demos/galaxy-collision) - N-body dynamics

Run demos:
```bash
claude -p "Simulate double-slit interference" --allowedTools "mcp__quantum-mcp__*"
```

---

## Next Steps

- 📖 **Examples**: [EXAMPLES.md](https://github.com/andylbrummer/math-mcp/tree/main/servers/*/EXAMPLES.md) in each server folder
- 🎬 **Visual Demos**: [Interactive demonstrations](/demos/index)
- 📚 **API Reference**: [Complete documentation](/api/overview)
