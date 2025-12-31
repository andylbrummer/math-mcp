# Contributing to Math-Physics-ML MCP

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Please be respectful and constructive in all interactions. We're building tools for scientific computing and welcome contributors from all backgrounds.

## Getting Started

### Prerequisites

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/) package manager
- Git
- (Optional) NVIDIA GPU with CUDA for GPU acceleration

### Development Setup

```bash
# Clone the repository
git clone https://github.com/andylbrummer/math-mcp.git
cd math-mcp

# Install all dependencies including dev tools
uv sync --all-extras

# Install pre-commit hooks
uv run pre-commit install

# Verify setup by running tests
uv run pytest -m "not gpu"
```

## Project Structure

```
math-mcp/
├── servers/                    # MCP server implementations
│   ├── math-mcp/              # Symbolic & numerical computing
│   ├── quantum-mcp/           # Quantum mechanics simulations
│   ├── molecular-mcp/         # Molecular dynamics
│   └── neural-mcp/            # Neural network training
├── shared/                     # Shared packages
│   ├── mcp-common/            # GPU manager, tasks, config
│   └── compute-core/          # Unified array interface
├── tests/                      # Integration tests
└── docs/                       # Documentation site
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Changes

- Follow the existing code style (enforced by Ruff)
- Add type hints to all functions
- Write docstrings for public APIs
- Add tests for new functionality

### 3. Run Quality Checks

```bash
# Format and lint
uv run ruff check --fix .
uv run ruff format .

# Type checking
uv run mypy shared/ servers/

# Run tests
uv run pytest -m "not gpu"

# Or run all checks via pre-commit
uv run pre-commit run --all-files
```

### 4. Commit Changes

We use conventional commits:

```bash
git commit -m "feat(math-mcp): add symbolic limit computation"
git commit -m "fix(quantum-mcp): correct wavefunction normalization"
git commit -m "docs: update installation instructions"
```

Commit types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, no logic change)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Build, CI, or tooling changes

### 5. Submit Pull Request

```bash
git push origin feature/your-feature-name
```

Then open a pull request on GitHub with:
- Clear description of changes
- Link to any related issues
- Screenshots/examples if applicable

## Adding New Tools

### To an Existing MCP Server

1. Add the tool function in `servers/<mcp>/src/<mcp>/server.py`
2. Register with the `@mcp.tool()` decorator
3. Add tests in `servers/<mcp>/tests/`
4. Update the info tool's category list

Example:

```python
@mcp.tool()
async def my_new_tool(
    param1: str,
    param2: int = 10,
) -> dict[str, Any]:
    """Brief description of what the tool does.

    Args:
        param1: Description of param1
        param2: Description of param2 (default: 10)

    Returns:
        Dictionary with result fields
    """
    # Implementation
    return {"result": computed_value}
```

### Adding a New MCP Server

1. Create directory structure:
   ```
   servers/new-mcp/
   ├── pyproject.toml
   ├── src/new_mcp/
   │   ├── __init__.py
   │   └── server.py
   └── tests/
       └── test_new_server.py
   ```

2. Add to workspace in root `pyproject.toml`

3. Implement the server following existing patterns

4. Add tests and documentation

## Testing Guidelines

### Writing Tests

```python
import pytest
from new_mcp.server import my_tool

@pytest.mark.asyncio
async def test_my_tool_basic():
    """Test basic functionality."""
    result = await my_tool(param1="test")
    assert "result" in result
    assert result["result"] == expected_value

@pytest.mark.asyncio
async def test_my_tool_edge_case():
    """Test edge case handling."""
    with pytest.raises(ValueError):
        await my_tool(param1="invalid")

@pytest.mark.gpu
@pytest.mark.asyncio
async def test_my_tool_gpu():
    """Test GPU acceleration."""
    result = await my_tool(param1="test", use_gpu=True)
    assert result["used_gpu"] == True
```

### Test Markers

- `@pytest.mark.gpu` - Requires CUDA GPU
- `@pytest.mark.slow` - Long-running tests
- `@pytest.mark.integration` - Integration tests

### Running Specific Tests

```bash
# Run tests for specific server
uv run pytest servers/math-mcp/tests/

# Run specific test file
uv run pytest servers/math-mcp/tests/test_symbolic.py

# Run specific test
uv run pytest servers/math-mcp/tests/test_symbolic.py::test_solve_quadratic

# Run with verbose output
uv run pytest -v

# Run with coverage
uv run pytest --cov=servers/math-mcp
```

## Code Style

### Python Style

- Line length: 100 characters
- Use type hints for all function signatures
- Use `async/await` for all MCP tool functions
- Prefer `dict[str, Any]` over `Dict[str, Any]`

### Imports

```python
# Standard library
import asyncio
from typing import Any

# Third-party
import numpy as np
from mcp.server import Server

# Local
from mcp_common import GPUManager
from compute_core import create_array
```

### Docstrings

Use Google-style docstrings:

```python
def compute_something(data: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Compute something interesting from data.

    This function performs an important computation that does
    something useful with the input data.

    Args:
        data: Input array of shape (N, M)
        scale: Scaling factor (default: 1.0)

    Returns:
        Transformed array of shape (N, M)

    Raises:
        ValueError: If data has wrong dimensions

    Example:
        >>> result = compute_something(np.array([[1, 2], [3, 4]]))
        >>> result.shape
        (2, 2)
    """
```

## Documentation

### Updating Docs

The documentation site is in `docs/` and uses Docusaurus.

```bash
cd docs
pnpm install
pnpm start  # Development server at http://localhost:3000
```

### Documentation Style

- Use clear, concise language
- Include code examples that actually work
- Show expected output where helpful
- Link to related concepts

## Release Process

Releases are automated via GitHub Actions when a version tag is pushed:

```bash
# Update version in pyproject.toml files
# Then tag and push
git tag v0.2.0
git push origin v0.2.0
```

This triggers:
1. Build packages for all servers
2. Run full test suite
3. Publish to PyPI
4. Create GitHub release

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open a GitHub Issue with reproduction steps
- **Features**: Open a GitHub Issue describing the use case

## Recognition

Contributors are recognized in:
- GitHub Contributors list
- Release notes for significant contributions

Thank you for contributing!
