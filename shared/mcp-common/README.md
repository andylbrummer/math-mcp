# scicomp-mcp-common

Common utilities and base classes for the Math-Physics-ML MCP system.

## Overview

This package provides shared functionality used across all MCP servers in the Math-Physics-ML system, including:

- MCP server base classes and utilities
- Common error handling patterns
- Shared type definitions and protocols
- Utility functions for tool registration and management

## Installation

```bash
pip install scicomp-mcp-common
```

## Usage

This package is primarily used as a dependency by other MCP servers in the system. Direct usage is typically not needed for end users.

```python
from mcp_common import MCPServer, Tool

# Example: Used by other MCP servers
server = MCPServer()
```

## Part of Math-Physics-ML MCP System

This package is part of a larger system that includes:
- [scicomp-compute-core](https://pypi.org/project/scicomp-compute-core/) - GPU-accelerated computation core
- [scicomp-math-mcp](https://pypi.org/project/scicomp-math-mcp/) - Symbolic math and numerical computing
- [scicomp-quantum-mcp](https://pypi.org/project/scicomp-quantum-mcp/) - Quantum mechanics simulations
- [scicomp-molecular-mcp](https://pypi.org/project/scicomp-molecular-mcp/) - Molecular dynamics
- [scicomp-neural-mcp](https://pypi.org/project/scicomp-neural-mcp/) - Neural network training

See the [full documentation](https://andylbrummer.github.io/math-mcp/) for more details.
