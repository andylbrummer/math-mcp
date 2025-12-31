---
sidebar_position: 4
---

# Progressive Discovery

All MCP servers implement progressive discovery through the `info` tool, allowing users to explore capabilities efficiently without overwhelming context.

## The Problem

Large MCP servers with many tools create challenges:

1. **Context Overload**: Sending all tool documentation uses excessive tokens
2. **Discovery Friction**: Users don't know what's available
3. **Learning Curve**: New users struggle to find relevant tools

## The Solution: Progressive Discovery

Each MCP server provides an `info` tool that reveals capabilities layer by layer:

```
Level 1: Overview     → Categories and tool counts
Level 2: Category     → Tools in a category with brief descriptions
Level 3: Tool Detail  → Full documentation for a specific tool
```

## How It Works

### Level 1: Overview

Start with `info()` or `info(topic="overview")` to see what's available:

```python
result = math_mcp.info()
```

**Response:**
```json
{
  "name": "Math MCP",
  "version": "1.0.0",
  "description": "Symbolic and numerical computing with GPU acceleration",
  "categories": {
    "symbolic": {"count": 4, "description": "Symbolic algebra operations"},
    "numerical": {"count": 3, "description": "GPU-accelerated numerical computing"},
    "transforms": {"count": 2, "description": "FFT and signal processing"},
    "optimization": {"count": 2, "description": "Function optimization and root finding"}
  },
  "total_tools": 11,
  "gpu_available": true
}
```

### Level 2: Category Details

Drill into a specific category:

```python
result = math_mcp.info(topic="symbolic")
```

**Response:**
```json
{
  "category": "symbolic",
  "description": "Symbolic algebra operations using SymPy",
  "tools": [
    {"name": "symbolic_solve", "brief": "Solve algebraic equations"},
    {"name": "symbolic_diff", "brief": "Compute derivatives"},
    {"name": "symbolic_integrate", "brief": "Compute integrals"},
    {"name": "symbolic_simplify", "brief": "Simplify expressions"}
  ]
}
```

### Level 3: Tool Documentation

Get full details for a specific tool:

```python
result = math_mcp.info(topic="symbolic_solve")
```

**Response:**
```json
{
  "tool": "symbolic_solve",
  "description": "Solve algebraic and transcendental equations symbolically",
  "parameters": {
    "equations": {
      "type": "string or array",
      "description": "Equation(s) to solve",
      "required": true,
      "examples": ["x**2 - 4", ["x + y - 3", "x - y - 1"]]
    },
    "variables": {
      "type": "string or array",
      "description": "Variable(s) to solve for",
      "required": false
    },
    "domain": {
      "type": "string",
      "options": ["real", "complex", "positive", "integer"],
      "default": "complex"
    }
  },
  "returns": {
    "solutions": "List of solutions",
    "variables": "Variables solved for"
  },
  "examples": [
    {
      "input": {"equations": "x**2 - 4"},
      "output": {"solutions": [-2, 2]}
    }
  ]
}
```

## Implementation Pattern

Each MCP server implements this pattern:

```python
@mcp.tool()
async def info(topic: str = "overview") -> dict:
    """Progressive discovery of server capabilities."""

    if topic == "overview":
        return {
            "name": SERVER_NAME,
            "categories": get_categories(),
            "total_tools": count_tools()
        }

    if topic in CATEGORIES:
        return {
            "category": topic,
            "tools": get_tools_in_category(topic)
        }

    if topic in TOOLS:
        return get_tool_documentation(topic)

    return {"error": f"Unknown topic: {topic}"}
```

## Benefits

### For Users

1. **Efficient Exploration**: Learn what's available without reading everything
2. **Context Preservation**: Only load documentation when needed
3. **Natural Flow**: Discover → Explore → Use

### For AI Assistants

1. **Token Efficiency**: Minimal context for tool discovery
2. **Accurate Tool Selection**: Detailed docs when actually using tools
3. **Better Recommendations**: Can suggest relevant tools based on categories

### For Developers

1. **Self-Documenting**: Documentation lives with the code
2. **Consistent UX**: Same pattern across all servers
3. **Easy Updates**: Add new tools without breaking discovery

## Usage Examples

### Exploring Math MCP

```python
# 1. What can Math MCP do?
info = math_mcp.info()
# See: symbolic (4), numerical (3), transforms (2), optimization (2)

# 2. What symbolic tools are available?
info = math_mcp.info(topic="symbolic")
# See: solve, diff, integrate, simplify

# 3. How do I use solve?
info = math_mcp.info(topic="symbolic_solve")
# See: full parameter documentation and examples

# 4. Now use it!
result = math_mcp.symbolic_solve(equations="x**3 - 6*x**2 + 11*x - 6")
# Returns: solutions [1, 2, 3]
```

### Exploring Quantum MCP

```python
# 1. Overview
info = quantum_mcp.info()
# Categories: potentials (2), wavepackets (2), simulations (3),
#            analysis (2), visualization (2)

# 2. What simulation tools exist?
info = quantum_mcp.info(topic="simulations")
# Tools: solve_schrodinger, solve_schrodinger_2d, get_task_status

# 3. How do I run a 2D simulation?
info = quantum_mcp.info(topic="solve_schrodinger_2d")
# Full documentation with parameters and examples
```

### Exploring Molecular MCP

```python
# 1. Overview
info = molecular_mcp.info()
# Categories: system (2), simulation (3), analysis (5), visualization (2)

# 2. What analysis can I do?
info = molecular_mcp.info(topic="analysis")
# Tools: get_trajectory, compute_rdf, compute_msd,
#        analyze_temperature, detect_phase_transition

# 3. How do I compute RDF?
info = molecular_mcp.info(topic="compute_rdf")
# Full documentation
```

### Exploring Neural MCP

```python
# 1. Overview
info = neural_mcp.info()
# Categories: models (3), data (2), training (2), evaluation (2),
#            tuning (1), visualization (3), deployment (1)

# 2. What training tools exist?
info = neural_mcp.info(topic="training")
# Tools: train_model, get_experiment_status

# 3. How do I train a model?
info = neural_mcp.info(topic="train_model")
# Full documentation with all parameters
```

## Design Principles

### 1. Overview Should Be Scannable

The overview response should fit in a single screen and give immediate value:

```json
{
  "name": "Server Name",
  "description": "One-line description",
  "categories": {"cat1": {"count": N}, ...},
  "total_tools": N
}
```

### 2. Categories Should Be Logical

Group tools by function, not implementation:

- Good: "symbolic", "numerical", "visualization"
- Bad: "sympy_tools", "numpy_tools", "plotting"

### 3. Tool Docs Should Be Complete

Include everything needed to use the tool:

- All parameters with types and defaults
- Return value structure
- At least one example with expected output
- Related tools for common workflows

### 4. Examples Should Be Runnable

Every example in the documentation should work exactly as shown:

```python
# This should actually work
result = symbolic_solve(equations="x**2 - 4")
assert result["solutions"] == [-2, 2]
```

## Comparison with Alternatives

| Approach | Pros | Cons |
|----------|------|------|
| Full documentation in system prompt | Always available | Token-heavy, overwhelming |
| No documentation | Token-efficient | Users can't discover features |
| Progressive discovery | Balanced, efficient | Requires extra info calls |

Progressive discovery provides the best balance of discoverability and efficiency, especially for complex MCP servers with many tools.
