# Math-Physics-ML MCP System

[![PyPI - Math MCP](https://img.shields.io/pypi/v/math-mcp?label=math-mcp)](https://pypi.org/project/math-mcp/)
[![PyPI - Quantum MCP](https://img.shields.io/pypi/v/quantum-mcp?label=quantum-mcp)](https://pypi.org/project/quantum-mcp/)
[![PyPI - Molecular MCP](https://img.shields.io/pypi/v/molecular-mcp?label=molecular-mcp)](https://pypi.org/project/molecular-mcp/)
[![PyPI - Neural MCP](https://img.shields.io/pypi/v/neural-mcp?label=neural-mcp)](https://pypi.org/project/neural-mcp/)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://andylbrummer.github.io/math-mcp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

GPU-accelerated [Model Context Protocol](https://modelcontextprotocol.io) servers for computational mathematics, physics simulations, and machine learning.

## Overview

This system provides **4 specialized MCP servers** that bring scientific computing capabilities to AI assistants like Claude:

| Server | Description | Tools |
|--------|-------------|-------|
| **Math MCP** | Symbolic algebra (SymPy) + numerical computing | 14 |
| **Quantum MCP** | Wave mechanics & Schrodinger simulations | 12 |
| **Molecular MCP** | Classical molecular dynamics | 15 |
| **Neural MCP** | Neural network training & evaluation | 16 |

**Key Features:**
- GPU acceleration with automatic CUDA detection (10-100x speedup)
- Async task support for long-running simulations
- Cross-MCP workflows via URI-based data sharing
- Progressive discovery for efficient tool exploration

## Quick Start

### Installation with uvx (Recommended)

Run any MCP server directly without installation:

```bash
# Run individual servers
uvx math-mcp
uvx quantum-mcp
uvx molecular-mcp
uvx neural-mcp
```

### Installation with pip/uv

```bash
# Install individual servers
pip install math-mcp
pip install quantum-mcp
pip install molecular-mcp
pip install neural-mcp

# Or install all at once
pip install math-mcp quantum-mcp molecular-mcp neural-mcp

# With GPU support (requires CUDA)
pip install math-mcp[gpu] quantum-mcp[gpu] molecular-mcp[gpu] neural-mcp[gpu]
```

## Configuration

### Claude Desktop

Add to your Claude Desktop configuration file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "math-mcp": {
      "command": "uvx",
      "args": ["math-mcp"]
    },
    "quantum-mcp": {
      "command": "uvx",
      "args": ["quantum-mcp"]
    },
    "molecular-mcp": {
      "command": "uvx",
      "args": ["molecular-mcp"]
    },
    "neural-mcp": {
      "command": "uvx",
      "args": ["neural-mcp"]
    }
  }
}
```

### Claude Code

Add to your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "math-mcp": {
      "command": "uvx",
      "args": ["math-mcp"]
    },
    "quantum-mcp": {
      "command": "uvx",
      "args": ["quantum-mcp"]
    }
  }
}
```

Or configure globally in `~/.claude/settings.json`.

## Usage Examples

### Math MCP

```python
# Solve equations symbolically
symbolic_solve(equations="x**3 - 6*x**2 + 11*x - 6")
# Result: [1, 2, 3]

# Compute derivatives
symbolic_diff(expression="sin(x)*exp(-x**2)", variable="x")
# Result: cos(x)*exp(-x**2) - 2*x*sin(x)*exp(-x**2)

# GPU-accelerated matrix operations
result = matrix_multiply(a=matrix_a, b=matrix_b, use_gpu=True)
```

### Quantum MCP

```python
# Create a Gaussian wave packet
psi = create_gaussian_wavepacket(
    grid_size=[256],
    position=[64],
    momentum=[2.0],
    width=5.0
)

# Solve time-dependent Schrodinger equation
simulation = solve_schrodinger(
    potential=barrier_potential,
    initial_state=psi,
    time_steps=1000,
    dt=0.1,
    use_gpu=True
)
```

### Molecular MCP

```python
# Create particle system
system = create_particles(
    n_particles=1000,
    box_size=[20, 20, 20],
    temperature=1.5
)

# Add Lennard-Jones potential
add_potential(system_id=system, potential_type="lennard_jones")

# Run MD simulation
trajectory = run_nvt(system_id=system, n_steps=100000, temperature=1.0)

# Analyze diffusion
msd = compute_msd(trajectory_id=trajectory)
```

### Neural MCP

```python
# Define model
model = define_model(architecture="resnet18", num_classes=10, pretrained=True)

# Load dataset
dataset = load_dataset(dataset_name="CIFAR10", split="train")

# Train
experiment = train_model(
    model_id=model,
    dataset_id=dataset,
    epochs=50,
    batch_size=128,
    use_gpu=True
)

# Export for deployment
export_model(model_id=model, format="onnx", output_path="model.onnx")
```

## Documentation

Full documentation is available at **[andylbrummer.github.io/math-mcp](https://andylbrummer.github.io/math-mcp/)**

- [Installation Guide](https://andylbrummer.github.io/math-mcp/getting-started/installation)
- [Configuration](https://andylbrummer.github.io/math-mcp/getting-started/configuration)
- [Quick Start Tutorial](https://andylbrummer.github.io/math-mcp/getting-started/quick-start)
- [Architecture Overview](https://andylbrummer.github.io/math-mcp/concepts/architecture)
- [GPU Acceleration](https://andylbrummer.github.io/math-mcp/concepts/gpu-acceleration)
- [API Reference](https://andylbrummer.github.io/math-mcp/api/overview)

## Development

```bash
# Clone the repository
git clone https://github.com/andylbrummer/math-mcp.git
cd math-mcp

# Install dependencies
uv sync --all-extras

# Run tests
uv run pytest -m "not gpu"  # CPU only
uv run pytest               # All tests (requires CUDA)

# Run with coverage
uv run pytest --cov=shared --cov=servers
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## Performance

GPU acceleration provides significant speedups for compute-intensive operations:

| MCP | Operation | CPU | GPU | Speedup |
|-----|-----------|-----|-----|---------|
| Math | Matrix multiply (4096x4096) | 2.1s | 35ms | 60x |
| Quantum | 2D Schrodinger (512x512, 1000 steps) | 2h | 2min | 60x |
| Molecular | MD (100k particles, 10k steps) | 1h | 30s | 120x |
| Neural | ResNet18 training (1 epoch) | 45min | 30s | 90x |

## Architecture

For technical details about the system architecture, see [ARCHITECTURE.md](ARCHITECTURE.md).

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
