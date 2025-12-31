# MCP Server Registration Guide

This guide explains how to register the Math-Physics-ML MCP servers with Claude Desktop or Claude Code.

## Available MCP Servers

All 4 MCP servers are now registered and ready to use:

1. **math-mcp** - Symbolic algebra and GPU-accelerated numerical computing
2. **quantum-mcp** - Wave mechanics and Schrödinger equation simulations
3. **molecular-mcp** - Classical molecular dynamics simulations
4. **neural-mcp** - Neural network training and experimentation

## Installation

All MCP servers are already installed in the project's virtual environment with entry points:

```bash
# Verify installation
ls -la .venv/bin/*-mcp

# You should see:
# - .venv/bin/math-mcp
# - .venv/bin/quantum-mcp
# - .venv/bin/molecular-mcp
# - .venv/bin/neural-mcp
```

## Registration Methods

### Method 1: Claude Desktop (Recommended for GUI)

Add the MCPs to your Claude Desktop configuration:

**Location:** `~/.config/Claude/claude_desktop_config.json` (Linux/macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows)

**Configuration:**

```json
{
  "mcpServers": {
    "math-mcp": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/home/beagle/work/math-mcp",
        "math-mcp"
      ]
    },
    "quantum-mcp": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/home/beagle/work/math-mcp",
        "quantum-mcp"
      ]
    },
    "molecular-mcp": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/home/beagle/work/math-mcp",
        "molecular-mcp"
      ]
    },
    "neural-mcp": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/home/beagle/work/math-mcp",
        "neural-mcp"
      ]
    }
  }
}
```

**Important:** Replace `/home/beagle/work/math-mcp` with the absolute path to your math-mcp directory.

### Method 2: Claude Code CLI

Use the provided configuration file:

```bash
# Use the pre-configured file
cat mcp-servers.json

# Or add to your Claude Code settings manually
claude-code config add-mcp-server math-mcp "uv run --directory $(pwd) math-mcp"
claude-code config add-mcp-server quantum-mcp "uv run --directory $(pwd) quantum-mcp"
claude-code config add-mcp-server molecular-mcp "uv run --directory $(pwd) molecular-mcp"
claude-code config add-mcp-server neural-mcp "uv run --directory $(pwd) neural-mcp"
```

### Method 3: Direct Execution (Testing)

Run any MCP server directly for testing:

```bash
# Activate the virtual environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# Run any server
math-mcp
quantum-mcp
molecular-mcp
neural-mcp

# Or with uv
uv run math-mcp
uv run quantum-mcp
uv run molecular-mcp
uv run neural-mcp
```

## Verification

After registration, verify the MCPs are available:

### In Claude Desktop:
1. Restart Claude Desktop
2. Start a new conversation
3. The MCP servers should appear in the tools panel
4. Try using a tool: "Create a 5x5 matrix of random numbers" (uses math-mcp)

### In Claude Code:
1. List available MCPs: `claude-code mcp list`
2. Check server status
3. Try a tool call

## Tool Discovery

Each MCP has an `info` tool for progressive discovery:

```
# Math MCP
info(topic="overview")           # See all categories
info(topic="symbolic")           # Symbolic algebra tools
info(topic="numerical")          # Numerical computing tools
info(topic="matrix_multiply")    # Specific tool help

# Quantum MCP
info(topic="overview")           # All quantum tools
info(topic="solve_schrodinger") # Schrödinger solver help

# Molecular MCP
info()                          # All molecular dynamics tools

# Neural MCP
info()                          # All neural network tools
```

## Example Usage

### Math MCP
```python
# Solve quadratic equation
symbolic_solve(expression="x**2 - 5*x + 6", variables=["x"])

# Create and multiply matrices
arr1 = create_array(shape=[100, 100], fill_type="random")
arr2 = create_array(shape=[100, 100], fill_type="random")
result = matrix_multiply(a=arr1, b=arr2, use_gpu=True)

# FFT analysis
signal = create_array(shape=[1024], fill_type="function", function="sin(2*pi*5*x)")
spectrum = fft(array_id=signal)
```

### Quantum MCP
```python
# Create potential well
potential = create_lattice_potential(
    grid_size=[256],
    lattice_type="square",
    depth=10.0
)

# Create wavepacket
psi = create_gaussian_wavepacket(
    grid_size=[256],
    position=[64],
    momentum=[2.0],
    width=5.0
)

# Solve Schrödinger equation
simulation = solve_schrodinger(
    potential=potential,
    initial_state=psi,
    time_steps=100,
    dt=0.1,
    use_gpu=True
)
```

### Molecular MCP
```python
# Create particle system
system = create_particles(
    n_particles=1000,
    box_size=[50, 50, 50],
    temperature=1.0
)

# Add Lennard-Jones potential
add_potential(
    system_id=system,
    potential_type="lennard_jones",
    epsilon=1.0,
    sigma=1.0
)

# Run MD simulation
trajectory = run_md(
    system_id=system,
    n_steps=10000,
    dt=0.001,
    use_gpu=True
)

# Analyze
rdf = compute_rdf(trajectory_id=trajectory, n_bins=100)
```

### Neural MCP
```python
# Define model
model = define_model(
    architecture="resnet18",
    num_classes=10,
    pretrained=False
)

# Load dataset
dataset = load_dataset(
    dataset_name="CIFAR10",
    split="train"
)

# Train
experiment = train_model(
    model_id=model,
    dataset_id=dataset,
    epochs=10,
    batch_size=64,
    learning_rate=0.001,
    use_gpu=True
)

# Evaluate
metrics = evaluate_model(model_id=model, dataset_id=dataset)
```

## GPU Support

All MCPs support GPU acceleration with automatic CPU fallback:

- **GPU Available:** Set `use_gpu=True` in tool calls for 10-100x speedup
- **No GPU:** Automatically falls back to CPU (NumPy instead of CuPy)
- **Check Status:** GPU Manager logs CUDA availability at startup

## Cross-MCP Workflows

MCPs can work together using resource URIs:

```python
# Math → Quantum workflow
# 1. Create potential array in Math MCP
potential_array = create_array(
    shape=[256],
    fill_type="function",
    function="10*exp(-(x-128)**2/100)"
)
# Returns: array://abc123...

# 2. Use in Quantum MCP
simulation = solve_schrodinger(
    potential="array://abc123...",  # Reference Math MCP array
    initial_state=psi,
    time_steps=100
)
```

## Troubleshooting

### Server won't start
```bash
# Check Python version (requires 3.11+)
python --version

# Reinstall dependencies
uv sync

# Reinstall servers
uv pip install -e servers/math-mcp -e servers/quantum-mcp \
               -e servers/molecular-mcp -e servers/neural-mcp
```

### Entry points not found
```bash
# Verify installation
pip show math-mcp quantum-mcp molecular-mcp neural-mcp

# Check entry points
which math-mcp quantum-mcp molecular-mcp neural-mcp

# Should show paths like:
# /path/to/math-mcp/.venv/bin/math-mcp
```

### GPU not detected
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
python -c "import cupy; print(cupy.cuda.is_available())"

# If False, install CUDA toolkit or use CPU mode (automatic fallback)
```

## Configuration Files

- **mcp-servers.json** - Pre-configured server definitions
- **config.kdl** - GPU memory settings and runtime configuration
- **pyproject.toml** - Package dependencies and entry points

## Support

For issues or questions:
1. Check STATUS.md for implementation details
2. Run tests: `pytest -v`
3. Check logs for error messages
4. Verify all dependencies are installed

## Development

To modify or extend the MCPs:

```bash
# Make changes to server code
vim servers/math-mcp/src/math_mcp/server.py

# No reinstall needed (editable install)
# Changes take effect immediately

# Run tests
pytest servers/math-mcp/tests/ -v

# Add new tools - update list_tools() and call_tool()
```

## Summary

✅ All 4 MCP servers registered with entry points
✅ Configuration file provided (mcp-servers.json)
✅ GPU acceleration with CPU fallback
✅ Progressive discovery via info tools
✅ Cross-MCP resource sharing
✅ 52 total tools across all servers

Ready to use in Claude Desktop, Claude Code, or any MCP-compatible client!
