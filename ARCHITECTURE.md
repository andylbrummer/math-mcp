# Architecture

Technical architecture documentation for the Math-Physics-ML MCP system.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              AI Assistant                                │
│                        (Claude Desktop/Code)                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                         MCP Protocol (stdio)
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────┐         ┌───────────────┐         ┌───────────────┐
│   Math MCP    │         │  Quantum MCP  │         │ Molecular MCP │
│  (14 tools)   │         │  (12 tools)   │         │  (15 tools)   │
└───────┬───────┘         └───────┬───────┘         └───────┬───────┘
        │                         │                         │
        └─────────────────────────┼─────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
            ┌───────────────┐         ┌───────────────┐
            │  mcp-common   │         │ compute-core  │
            │  (shared)     │         │   (shared)    │
            └───────────────┘         └───────────────┘
                    │                         │
                    └─────────────┬───────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
            ┌───────────────┐         ┌───────────────┐
            │    NumPy      │         │     CuPy      │
            │    (CPU)      │         │    (GPU)      │
            └───────────────┘         └───────────────┘
```

## Package Structure

### Workspace Layout

```
math-mcp/
├── pyproject.toml              # Workspace root
├── servers/
│   ├── math-mcp/               # Symbolic & numerical computing
│   │   ├── pyproject.toml
│   │   ├── src/math_mcp/
│   │   │   ├── __init__.py
│   │   │   └── server.py       # MCP server implementation
│   │   └── tests/
│   ├── quantum-mcp/            # Schrodinger equation solver
│   ├── molecular-mcp/          # Molecular dynamics
│   └── neural-mcp/             # Neural network training
├── shared/
│   ├── mcp-common/             # Shared MCP utilities
│   │   ├── src/mcp_common/
│   │   │   ├── gpu_manager.py  # CUDA/CPU backend selection
│   │   │   ├── task_manager.py # Async task handling
│   │   │   ├── config.py       # KDL configuration
│   │   │   └── serialization.py
│   │   └── tests/
│   └── compute-core/           # Unified compute interface
│       ├── src/compute_core/
│       │   ├── arrays.py       # NumPy/CuPy abstraction
│       │   ├── fft.py          # FFT operations
│       │   └── linalg.py       # Linear algebra
│       └── tests/
└── tests/                      # Integration tests
```

### Dependency Graph

```
math-mcp ──────┬──► mcp-common ──► mcp>=1.0.0
               │
               └──► compute-core ──┬──► numpy>=1.26
                                   └──► cupy (optional)

quantum-mcp ───┬──► mcp-common
               └──► compute-core

molecular-mcp ─┬──► mcp-common
               └──► compute-core

neural-mcp ────┬──► mcp-common
               └──► torch>=2.0
```

## Shared Packages

### mcp-common

Core utilities shared across all MCP servers.

#### GPUManager

Handles transparent GPU/CPU backend selection:

```python
from mcp_common import GPUManager

gpu = GPUManager()

# Check availability
if gpu.is_available():
    print(f"GPU: {gpu.device_name}")

# Get appropriate array module
xp = gpu.get_array_module()  # Returns cupy or numpy

# Create array on best device
arr = xp.zeros((1000, 1000))
```

**Features:**
- Automatic CUDA detection
- Graceful CPU fallback
- Memory management
- Device info queries

#### TaskManager

Manages long-running async operations:

```python
from mcp_common import TaskManager

tasks = TaskManager()

# Start async task
task_id = await tasks.start_task(
    name="simulation",
    coroutine=run_simulation(params)
)

# Check status
status = await tasks.get_status(task_id)
# Returns: {"state": "running", "progress": 0.45}

# Get result when complete
result = await tasks.get_result(task_id)
```

**Features:**
- Task lifecycle management
- Progress tracking
- Result caching
- Automatic cleanup

#### Configuration

KDL-based configuration:

```python
from mcp_common import Config

config = Config.load("server.kdl")
gpu_enabled = config.get("gpu.enabled", default=True)
max_memory = config.get("gpu.max_memory_gb", default=8)
```

### compute-core

Unified array interface abstracting NumPy/CuPy differences.

#### Array Operations

```python
from compute_core import create_array, to_numpy

# Create on best available device
arr = create_array(
    shape=(1024, 1024),
    dtype="float64",
    device="auto"  # "cpu", "gpu", or "auto"
)

# Convert to numpy for serialization
np_arr = to_numpy(arr)
```

#### FFT Module

```python
from compute_core import fft, ifft

# Automatic GPU acceleration
spectrum = fft(signal, use_gpu=True)
reconstructed = ifft(spectrum, use_gpu=True)
```

#### Linear Algebra

```python
from compute_core import solve, matmul, eig

# Solve Ax = b
x = solve(A, b, use_gpu=True)

# Matrix multiplication
C = matmul(A, B, use_gpu=True)

# Eigenvalue decomposition
eigenvalues, eigenvectors = eig(matrix, use_gpu=True)
```

## MCP Server Architecture

### Server Structure

Each MCP server follows this pattern:

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server

# Create server instance
mcp = Server("server-name")

# Storage for stateful objects
_arrays: dict[str, np.ndarray] = {}
_simulations: dict[str, SimulationResult] = {}

@mcp.tool()
async def tool_name(param1: str, param2: int = 10) -> dict[str, Any]:
    """Tool description."""
    # Implementation
    result = compute(param1, param2)

    # Store large results by reference
    result_id = f"result://{uuid4()}"
    _results[result_id] = result

    return {"result_id": result_id, "summary": summary}

async def main():
    async with stdio_server() as (read, write):
        await mcp.run(read, write, mcp.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
```

### URI-Based References

Large data objects are stored by reference to minimize token usage:

```
array://uuid        → Numerical arrays
potential://uuid    → Quantum potentials
system://uuid       → Particle systems
trajectory://uuid   → MD trajectories
model://uuid        → Neural network models
dataset://uuid      → Training datasets
experiment://uuid   → Training experiments
simulation://uuid   → Simulation results
```

**Benefits:**
- Reduces context window usage
- Enables cross-tool data passing
- Supports long-running workflows
- Allows garbage collection

### Progressive Discovery

Each server implements an `info` tool for capability exploration:

```python
@mcp.tool()
async def info(topic: str | None = None) -> dict[str, Any]:
    """Progressive discovery of server capabilities."""

    if topic is None or topic == "overview":
        return {
            "name": "Math MCP",
            "version": "0.1.0",
            "categories": {
                "symbolic": {"count": 4, "description": "Symbolic algebra"},
                "numerical": {"count": 3, "description": "Array operations"},
                "transforms": {"count": 2, "description": "FFT"},
                "optimization": {"count": 2, "description": "Optimization"},
            }
        }

    if topic in CATEGORIES:
        return {"category": topic, "tools": CATEGORY_TOOLS[topic]}

    if topic in TOOLS:
        return {"tool": topic, "documentation": TOOL_DOCS[topic]}

    return {"error": f"Unknown topic: {topic}"}
```

## GPU Acceleration

### Backend Selection

```python
class GPUManager:
    def __init__(self):
        self._gpu_available = False
        self._xp = np  # Default to numpy

        try:
            import cupy as cp
            # Test GPU access
            cp.cuda.Device(0).compute_capability
            self._gpu_available = True
            self._xp = cp
        except Exception:
            pass

    def get_array_module(self, use_gpu: bool = True):
        if use_gpu and self._gpu_available:
            return self._xp
        return np

    def to_numpy(self, arr):
        if hasattr(arr, 'get'):  # CuPy array
            return arr.get()
        return arr
```

### Memory Management

```python
def compute_with_memory_limit(func, *args, max_memory_gb=8):
    """Execute computation with memory limit."""
    xp = gpu.get_array_module()

    if xp.__name__ == 'cupy':
        # Check available memory
        free_memory = xp.cuda.Device().mem_info[0]
        if free_memory < max_memory_gb * 1e9:
            # Fall back to CPU
            xp = np

    return func(*args, xp=xp)
```

### Performance Characteristics

| Operation | Complexity | CPU Time | GPU Time | Notes |
|-----------|------------|----------|----------|-------|
| Matrix multiply (N×N) | O(N³) | ~N³/10⁹ s | ~N³/10¹² s | 1000x at N=4096 |
| FFT (N points) | O(N log N) | ~N log N / 10⁸ s | ~N log N / 10¹⁰ s | 100x at N=2²⁰ |
| Eigendecomposition | O(N³) | ~N³/10⁸ s | ~N³/10¹⁰ s | 100x at N=2048 |

## Async Task System

### Task Lifecycle

```
┌─────────┐     ┌─────────┐     ┌───────────┐     ┌───────────┐
│ pending │ ──► │ running │ ──► │ completed │ ──► │  cleaned  │
└─────────┘     └─────────┘     └───────────┘     └───────────┘
                     │                                   ▲
                     ▼                                   │
               ┌──────────┐                              │
               │  failed  │ ─────────────────────────────┘
               └──────────┘
```

### Implementation

```python
class TaskManager:
    def __init__(self):
        self._tasks: dict[str, asyncio.Task] = {}
        self._results: dict[str, Any] = {}
        self._status: dict[str, TaskStatus] = {}

    async def start_task(self, name: str, coro: Coroutine) -> str:
        task_id = f"task://{uuid4()}"
        self._status[task_id] = TaskStatus(state="running", progress=0.0)

        async def wrapper():
            try:
                result = await coro
                self._results[task_id] = result
                self._status[task_id].state = "completed"
            except Exception as e:
                self._status[task_id].state = "failed"
                self._status[task_id].error = str(e)

        self._tasks[task_id] = asyncio.create_task(wrapper())
        return task_id

    async def get_status(self, task_id: str) -> dict:
        return self._status[task_id].to_dict()

    async def get_result(self, task_id: str) -> Any:
        task = self._tasks.get(task_id)
        if task:
            await task
        return self._results.get(task_id)
```

## Security Considerations

### Input Validation

All user inputs are validated before processing:

```python
def validate_expression(expr: str) -> str:
    """Validate symbolic expression for safety."""
    # Whitelist allowed characters
    allowed = set("0123456789+-*/^().xyzabcdefghijklmnopqrstuvw ")
    allowed.update(set("sincostan log exp sqrt pi E I"))

    if not all(c in allowed for c in expr):
        raise ValueError("Expression contains invalid characters")

    # Parse with SymPy's safe parser
    return sympify(expr, evaluate=False)
```

### Sandboxing

- No file system access except designated temp directories
- No network access
- No subprocess execution
- Memory limits enforced

### Resource Limits

```python
MAX_ARRAY_SIZE = 100_000_000  # 100M elements
MAX_SIMULATION_STEPS = 10_000_000
MAX_TRAINING_EPOCHS = 1000
TASK_TIMEOUT_SECONDS = 3600  # 1 hour
```

## Testing Strategy

### Test Categories

1. **Unit Tests**: Individual functions
2. **Integration Tests**: Tool interactions
3. **GPU Tests**: CUDA-specific functionality
4. **Performance Tests**: Benchmarks

### Test Markers

```python
@pytest.mark.gpu          # Requires CUDA
@pytest.mark.slow         # Long-running
@pytest.mark.integration  # Cross-component
```

### Coverage Targets

- Shared packages: 90%+
- Server tools: 80%+
- Error handling: 100%

## Deployment

### Package Publishing

Each server is published as a separate PyPI package:

- `math-mcp`
- `quantum-mcp`
- `molecular-mcp`
- `neural-mcp`

Shared packages are bundled (not published separately) to simplify installation.

### Version Strategy

- Semantic versioning (MAJOR.MINOR.PATCH)
- All packages versioned together
- Breaking changes require major version bump

### Release Process

1. Update version in all `pyproject.toml` files
2. Create git tag: `git tag v0.2.0`
3. Push tag: `git push origin v0.2.0`
4. GitHub Actions builds and publishes to PyPI

## Future Considerations

### Planned Enhancements

1. **Distributed Computing**: Multi-GPU and cluster support
2. **Checkpointing**: Save/restore long simulations
3. **Caching**: Memoize expensive computations
4. **Streaming**: Real-time progress for visualizations

### Extension Points

- Custom potential functions (Quantum MCP)
- Custom force fields (Molecular MCP)
- Custom architectures (Neural MCP)
- Plugin system for new backends
