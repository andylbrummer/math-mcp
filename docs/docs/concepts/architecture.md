---
sidebar_position: 1
---

# Architecture

The Math-Physics-ML MCP System is designed as a modular, GPU-accelerated platform built on the Model Context Protocol (MCP).

## System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────┐
│                   MCP Client                            │
│            (Claude Desktop / Claude Code)               │
└─────────────────┬───────────────────────────────────────┘
                  │
      ┌───────────┴───────────┬─────────────┬─────────────┐
      │                       │             │             │
┌─────▼─────┐        ┌────────▼────┐  ┌────▼────┐  ┌────▼────┐
│  Math MCP │        │ Quantum MCP │  │Molecular│  │ Neural  │
│           │        │             │  │  MCP    │  │  MCP    │
│ 14 tools  │        │  12 tools   │  │15 tools │  │16 tools │
└─────┬─────┘        └────────┬────┘  └────┬────┘  └────┬────┘
      │                       │            │            │
      └───────────┬───────────┴────────────┴────────────┘
                  │
       ┌──────────▼──────────┐
       │  Shared Packages    │
       ├─────────────────────┤
       │  mcp-common         │
       │  compute-core       │
       └─────────────────────┘
                  │
       ┌──────────▼──────────┐
       │  GPU / CPU Backend  │
       │  (CUDA / NumPy)     │
       └─────────────────────┘
```

## Components

### MCP Servers

The system consists of 4 specialized MCP servers:

#### 1. Math MCP (Foundation Layer)
- **Purpose**: Symbolic algebra and numerical computing
- **Tools**: 14 tools across symbolic, numerical, transforms, and optimization
- **Dependencies**: SymPy, NumPy, SciPy, CuPy (optional)
- **Role**: Foundation layer used by other MCPs

#### 2. Quantum MCP
- **Purpose**: Wave mechanics and Schrödinger equation simulations
- **Tools**: 12 tools for quantum simulations
- **Dependencies**: Math MCP (for FFT operations), NumPy, CuPy
- **Key Algorithm**: Split-step Fourier method for time evolution

#### 3. Molecular MCP
- **Purpose**: Classical molecular dynamics simulations
- **Tools**: 15 tools for particle systems and MD simulations
- **Dependencies**: NumPy, CuPy (optional)
- **Ensembles**: NVE, NVT, NPT

#### 4. Neural MCP
- **Purpose**: Neural network training and experimentation
- **Tools**: 16 tools for deep learning workflows
- **Dependencies**: PyTorch, torchvision
- **Features**: Pre-built models, custom architectures, hyperparameter tuning

### Shared Packages

#### mcp-common
Provides shared infrastructure for all MCP servers:

- **GPUManager**: Singleton for GPU detection and memory management
- **TaskManager**: Async task management using MCP Tasks primitive
- **Config**: KDL configuration file parsing
- **Serialization**: Size-based data serialization strategies

#### compute-core
Unified computational backend:

- **Array Interface**: Automatic NumPy/CuPy switching based on GPU availability
- **FFT Operations**: Unified FFT interface for CPU and GPU
- **Linear Algebra**: Matrix operations with automatic backend selection

## MCP Dependency Graph

```
                    Math MCP
                  (Foundation)
                       │
       ┌───────────────┼───────────────┐
       │               │               │
       │               │               │
  Quantum MCP    Molecular MCP    Neural MCP
  (uses FFT)     (independent)   (independent)
```

**Key Insight**: Only Quantum MCP depends on Math MCP (for FFT operations). Molecular and Neural MCPs are independent but can exchange data through client orchestration.

## Data Flow

### Resource URIs

The system uses URI-based references for efficient data sharing:

```
array://abc123     → Large arrays (Math MCP)
potential://def456 → Quantum potentials
system://ghi789    → Particle systems
model://jkl012     → Neural networks
```

### Serialization Strategy

Based on data size:

- **Small (less than 10MB)**: Inline JSON in tool responses
- **Medium (10-100MB)**: In-memory cache with URI references
- **Large (greater than 100MB)**: Disk storage with file path references

### Cross-MCP Workflows

Example: Math MCP → Quantum MCP

```python
# 1. Create potential with Math MCP
potential_array = create_array(
    shape=[256],
    fill_type='function',
    function='10*exp(-(x-128)**2/100)'
)
# Returns: array://abc123

# 2. Use in Quantum MCP
simulation = solve_schrodinger(
    potential='array://abc123',  # Reference Math MCP data
    initial_state=wavepacket,
    time_steps=1000
)
```

## Technology Stack

### Backend
- **Python 3.11+**: Core language
- **FastMCP**: MCP server framework
- **NumPy**: CPU arrays and linear algebra
- **CuPy**: GPU-accelerated arrays (optional)
- **SymPy**: Symbolic mathematics
- **PyTorch**: Deep learning

### Configuration
- **KDL**: Configuration file format
- **uv**: Fast Python package manager

### Testing
- **pytest**: Test framework
- **pre-commit**: Git hooks for code quality

## GPU Acceleration

### GPU Manager Architecture

```python
class GPUManager:
    """Singleton managing GPU resources"""

    def __init__(self):
        self.backend = self._detect_backend()
        self.memory_pool = self._init_memory_pool()

    def _detect_backend(self):
        try:
            import cupy
            if cupy.cuda.is_available():
                return 'cuda'
        except:
            pass
        return 'cpu'

    def get_array_module(self):
        """Returns NumPy or CuPy"""
        return cupy if self.backend == 'cuda' else numpy
```

### Automatic Fallback

All tools support both CPU and GPU:

```python
def matrix_multiply(a, b, use_gpu=True):
    xp = gpu_manager.get_array_module() if use_gpu else numpy
    return xp.matmul(a, b)
```

## Async Task Management

For long-running operations:

```python
# Start async task
task_id = task_manager.create_task(
    func=run_simulation,
    args=(system_id, n_steps)
)

# Poll progress
status = task_manager.get_status(task_id)
# {state: 'running', progress: 45}

# Get result when complete
result = task_manager.get_result(task_id)
```

## Design Patterns

### Progressive Discovery
Each MCP has an `info` tool for capability discovery:

```
info() → List all categories
info(topic='symbolic') → Show symbolic tools
info(topic='solve') → Detailed help for specific tool
```

### Hub-and-Spoke (Math MCP)
Central server providing foundational capabilities to other MCPs.

### Registry Pattern (Neural MCP)
Model and experiment registry for tracking training runs.

### CRUD + Analysis (Molecular MCP)
Create systems, run simulations, analyze trajectories.

## Performance Considerations

### Optimization Strategies

1. **GPU Memory Pooling**: Reuse GPU memory allocations
2. **Lazy Evaluation**: Defer computations until needed
3. **Batch Operations**: Group operations for efficiency
4. **Sparse Storage**: Efficient storage for trajectory data

### Scaling Characteristics

| MCP | Complexity | Memory | GPU Speedup |
|-----|-----------|--------|-------------|
| Math | O(N) to O(N³) | Up to 5GB | 10-100x |
| Quantum | O(N log N) per step | 1-10GB | 6-60x |
| Molecular | O(N) with cutoffs | 5-100GB | 10-100x |
| Neural | O(E·N·B) | 5-30GB | 5-20x |

## Security

### Input Validation
- Symbolic expressions: SymPy with safe evaluation
- Array sizes: Configurable limits
- File paths: Sandboxed to allowed directories

### Resource Limits
- Max array size: 100M elements
- Max particles: 10M
- Timeouts for symbolic operations

## Deployment

### Development
```bash
uv run math-mcp
```

### Production (Docker)
```yaml
services:
  math-mcp:
    image: math-mcp:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
```

## Next Steps

- **[GPU Acceleration](gpu-acceleration)** - Deep dive into GPU features
- **[Cross-MCP Workflows](cross-mcp-workflows)** - Building multi-MCP pipelines
- **[API Reference](../api/overview)** - Complete API documentation
