---
sidebar_position: 1
title: Example Prompts
---

# 1000+ Example Prompts

**You know the physics. You know the math. You just don't want to learn SymPy, NumPy, or PyTorch APIs.**

Browse natural language prompts organized by discipline. Just describe what you want—the system handles the implementation.

## Available Collections

| Domain | Topics | Examples |
|--------|--------|----------|
| [Physics](./examples-physics.md) | Mechanics, E&M, Quantum, Relativity, Fluids, Astrophysics | ~130 |
| [Chemistry](./examples-chemistry.md) | Quantum Chem, MD, Kinetics, Spectroscopy, Biochemistry | ~100 |
| [Mathematics](./examples-mathematics.md) | Calculus, Linear Algebra, PDEs, Number Theory, Optimization | ~130 |
| [Machine Learning](./examples-ml-ai.md) | Transformers, Vision, Generative Models, RL, GNNs | ~130 |
| [Engineering](./examples-engineering.md) | Structural, Electrical, Control Systems, Robotics | ~120 |
| [Biology](./examples-biology.md) | Systems Bio, Genomics, Neuroscience, Epidemiology | ~120 |
| [Finance](./examples-finance.md) | Options, Portfolio, Risk, Algo Trading, Econometrics | ~130 |
| [Data Science](./examples-data-science.md) | Statistics, Clustering, NLP, A/B Testing, Causal Inference | ~130 |

## How to Use

1. **Find your domain** in the table above
2. **Browse examples** for prompts similar to your task
3. **Copy and modify** for your specific needs
4. **The AI handles** all implementation details

## Example Workflow

**You want to:** Simulate quantum tunneling

**You find in Physics → Quantum Mechanics:**
```
Simulate quantum tunneling through a rectangular barrier - what's the transmission probability?
```

**You ask exactly that.** The system:
- Creates the potential barrier
- Initializes a Gaussian wavepacket
- Runs the Schrödinger equation solver
- Computes transmission/reflection coefficients
- Visualizes the probability density evolution

No need to know about `numpy.fft`, split-step Fourier methods, or CuPy GPU arrays.
