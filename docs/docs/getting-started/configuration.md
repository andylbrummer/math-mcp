---
sidebar_position: 2
---

# Configuration

This guide explains how to configure the Math-Physics-ML MCP servers for use with Claude Desktop or Claude Code.

## Available MCP Servers

The system provides four specialized MCP servers:

1. **math-mcp** - Symbolic algebra and GPU-accelerated numerical computing
2. **quantum-mcp** - Wave mechanics and Schrödinger equation simulations
3. **molecular-mcp** - Classical molecular dynamics simulations
4. **neural-mcp** - Neural network training and experimentation

## Prerequisites

Before configuring the MCP servers, ensure they are installed:

```bash
# Verify installation
ls -la .venv/bin/*-mcp

# You should see:
# - .venv/bin/math-mcp
# - .venv/bin/quantum-mcp
# - .venv/bin/molecular-mcp
# - .venv/bin/neural-mcp
```

If the servers are not installed, run:

```bash
uv sync --all-extras
uv pip install --python .venv/bin/python3 -e shared/mcp-common -e shared/compute-core
uv pip install --python .venv/bin/python3 -e servers/math-mcp -e servers/quantum-mcp
uv pip install --python .venv/bin/python3 -e servers/molecular-mcp -e servers/neural-mcp
```

## Configuration Methods

### Claude Desktop Configuration

Claude Desktop is the recommended method for GUI-based usage.

**Configuration File Location:**
- **Linux/macOS:** `~/.config/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

**Configuration:**

Add the following to your `claude_desktop_config.json` file:

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

**Important:** Replace `/home/beagle/work/math-mcp` with the absolute path to your math-mcp installation directory.

**Verification:**

1. Restart Claude Desktop
2. Start a new conversation
3. The MCP servers should appear in the tools panel
4. Try using a tool: "Create a 5x5 matrix of random numbers" (uses math-mcp)

### Claude Code CLI Configuration

For command-line usage with Claude Code:

```bash
# Add servers manually using Claude Code CLI
claude-code config add-mcp-server math-mcp "uv run --directory $(pwd) math-mcp"
claude-code config add-mcp-server quantum-mcp "uv run --directory $(pwd) quantum-mcp"
claude-code config add-mcp-server molecular-mcp "uv run --directory $(pwd) molecular-mcp"
claude-code config add-mcp-server neural-mcp "uv run --directory $(pwd) neural-mcp"
```

Alternatively, use the provided `mcp-servers.json` configuration file in the project root.

**Verification:**

```bash
# List available MCPs
claude-code mcp list

# Check server status
claude-code mcp status
```

### Direct Execution (Testing)

For testing and development, you can run any MCP server directly:

```bash
# Activate the virtual environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# Run any server directly
math-mcp
quantum-mcp
molecular-mcp
neural-mcp

# Or with uv (no activation needed)
uv run math-mcp
uv run quantum-mcp
uv run molecular-mcp
uv run neural-mcp
```

## Tool Discovery

Each MCP server provides an `info` tool for progressive discovery of capabilities:

### Math MCP

```python
info(topic="overview")           # See all categories
info(topic="symbolic")           # Symbolic algebra tools
info(topic="numerical")          # Numerical computing tools
info(topic="transforms")         # FFT and transforms
info(topic="optimization")       # Optimization tools
info(topic="matrix_multiply")    # Specific tool help
```

### Quantum MCP

```python
info(topic="overview")           # All quantum tools
info(topic="solve_schrodinger") # Schrödinger solver help
```

### Molecular MCP

```python
info()                          # All molecular dynamics tools
```

### Neural MCP

```python
info()                          # All neural network tools
```

## GPU Support

All MCP servers support GPU acceleration with automatic CPU fallback:

- **GPU Available:** Set `use_gpu=True` in tool calls for 10-100x speedup
- **No GPU:** Automatically falls back to CPU (NumPy instead of CuPy)
- **Check Status:** GPU Manager logs CUDA availability at startup

To verify GPU availability:

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
python -c "import cupy; print(cupy.cuda.is_available())"
```

## Configuration Files

The system uses the following configuration files:

- **mcp-servers.json** - Pre-configured server definitions
- **config.kdl** - GPU memory settings and runtime configuration
- **pyproject.toml** - Package dependencies and entry points

### GPU Memory Configuration

Edit `config.kdl` to customize GPU memory settings:

```kdl
gpu {
    memory_pool_size "500MB"
    memory_growth true
}
```

## Troubleshooting

### Server Won't Start

```bash
# Check Python version (requires 3.11+)
python --version

# Reinstall dependencies
uv sync

# Reinstall servers
uv pip install -e servers/math-mcp -e servers/quantum-mcp \
               -e servers/molecular-mcp -e servers/neural-mcp
```

### Entry Points Not Found

```bash
# Verify installation
pip show math-mcp quantum-mcp molecular-mcp neural-mcp

# Check entry points
which math-mcp quantum-mcp molecular-mcp neural-mcp

# Should show paths like:
# /path/to/math-mcp/.venv/bin/math-mcp
```

### GPU Not Detected

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
python -c "import cupy; print(cupy.cuda.is_available())"

# If False, install CUDA toolkit or use CPU mode (automatic fallback)
```

### Tools Not Appearing in Claude Desktop

1. Ensure the configuration file path is correct
2. Verify the absolute path to math-mcp directory is correct
3. Restart Claude Desktop completely
4. Check Claude Desktop logs for error messages

## Next Steps

Once configured, proceed to the [Quick Start](./quick-start.md) guide to learn how to use the MCP tools.
