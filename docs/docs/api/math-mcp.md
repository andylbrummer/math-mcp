---
sidebar_position: 2
---

# Math MCP

Foundation layer for symbolic algebra and GPU-accelerated numerical computing.

## Overview

Math MCP provides 14 tools across 5 categories:
- **Discovery**: Progressive capability exploration
- **Symbolic**: Equation solving, calculus, simplification
- **Numerical**: Array operations, linear algebra
- **Transforms**: Fast Fourier Transform
- **Optimization**: Function minimization, root finding

## Tools

### Discovery

#### `info`

Progressive discovery of Math MCP capabilities.

**Parameters:**
- `topic` (string, optional): Category or tool name
  - Omit for overview
  - `"symbolic"`, `"numerical"`, `"transforms"`, `"optimization"` for categories
  - Specific tool name for detailed help

**Example:**
```python
# Get overview
info()

# Get symbolic tools
info(topic="symbolic")

# Get help for specific tool
info(topic="matrix_multiply")
```

### Symbolic Tools

#### `symbolic_solve`

Solve symbolic equations.

**Parameters:**
- `equations` (string | array): Equation(s) to solve
- `variables` (string | array, optional): Variables to solve for
- `domain` (string, optional): Solution domain
  - `"complex"` (default)
  - `"real"`
  - `"positive"`
  - `"integer"`

**Example:**
```python
# Solve quadratic
symbolic_solve(
    equations="x**2 - 4",
    variables="x"
)
# Returns: `{solutions: ["-2", "2"]}`

# System of equations
symbolic_solve(
    equations=["x + y = 5", "x - y = 1"],
    variables=["x", "y"]
)
# Returns: `{solutions: [{x: "3", y: "2"}]}`
```

#### `symbolic_diff`

Compute symbolic derivatives.

**Parameters:**
- `expression` (string): Expression to differentiate
- `variable` (string): Differentiation variable
- `order` (int, optional): Derivative order (default: 1)

**Example:**
```python
symbolic_diff(
    expression="x**3 + 2*x**2 + x",
    variable="x",
    order=2
)
# Returns: `{result: "6*x + 4"}`
```

#### `symbolic_integrate`

Compute symbolic integrals.

**Parameters:**
- `expression` (string): Expression to integrate
- `variable` (string): Integration variable
- `limits` (array, optional): `[lower, upper]` for definite integral

**Example:**
```python
# Indefinite integral
symbolic_integrate(
    expression="x**2",
    variable="x"
)
# Returns: `{result: "x**3/3"}`

# Definite integral
symbolic_integrate(
    expression="x**2",
    variable="x",
    limits=[0, 1]
)
# Returns: `{result: "1/3"}`
```

#### `symbolic_simplify`

Simplify symbolic expressions.

**Parameters:**
- `expression` (string): Expression to simplify
- `method` (string, optional): Simplification method
  - `"auto"` (default)
  - `"trigsimp"` - Trigonometric simplification
  - `"expand"` - Algebraic expansion
  - `"factor"` - Factorization

**Example:**
```python
symbolic_simplify(
    expression="(x**2 - 1)/(x - 1)",
    method="auto"
)
# Returns: `{result: "x + 1"}`
```

### Numerical Tools

#### `create_array`

Create arrays with various initialization patterns.

**Parameters:**
- `shape` (array): Array dimensions `[n]` or `[m, n]`
- `fill_type` (string, optional): Initialization pattern
  - `"zeros"` (default)
  - `"ones"`
  - `"random"` - Uniform random [0, 1)
  - `"linspace"` - Evenly spaced values
  - `"function"` - Custom function
- `function` (string, optional): Function string (for `fill_type="function"`)
- `linspace_range` (array, optional): `[start, stop]` for linspace
- `dtype` (string, optional): Data type (default: `"float64"`)
- `use_gpu` (bool, optional): Use GPU (default: `true`)

**Example:**
```python
# Random matrix
create_array(
    shape=[100, 100],
    fill_type="random",
    use_gpu=True
)
# Returns: `{array_id: "array://abc123", shape: [100, 100], dtype: "float64"}`

# Custom function
create_array(
    shape=[256],
    fill_type="function",
    function="sin(2*pi*x/256)"
)
```

#### `matrix_multiply`

GPU-accelerated matrix multiplication.

**Parameters:**
- `a` (array | string): First matrix or array URI
- `b` (array | string): Second matrix or array URI
- `use_gpu` (bool, optional): Use GPU (default: `true`)

**Example:**
```python
matrix_multiply(
    a=[[1, 2], [3, 4]],
    b=[[5, 6], [7, 8]],
    use_gpu=True
)
# Returns: `{result: [[19, 22], [43, 50]]}`
```

#### `solve_linear_system`

Solve linear system Ax = b.

**Parameters:**
- `a` (array): Coefficient matrix
- `b` (array): Right-hand side vector/matrix
- `use_gpu` (bool, optional): Use GPU (default: `true`)

**Example:**
```python
solve_linear_system(
    a=[[2, 1], [1, 3]],
    b=[5, 10],
    use_gpu=True
)
# Returns: `{solution: [1, 3]}`
```

### Transform Tools

#### `fft`

Fast Fourier Transform.

**Parameters:**
- `array` (array | string): Input array or array URI
- `norm` (string, optional): Normalization mode
  - `"backward"` (default)
  - `"ortho"` - Orthonormal
  - `"forward"` - Forward normalized
- `use_gpu` (bool, optional): Use GPU (default: `true`)

**Example:**
```python
# Create signal
signal = create_array(
    shape=[1024],
    fill_type="function",
    function="sin(2*pi*5*x/1024) + sin(2*pi*10*x/1024)"
)

# Compute FFT
fft(
    array=signal['array_id'],
    use_gpu=True
)
# Returns: `{array_id: "array://def456", shape: [1024], dtype: "complex128"}`
```

#### `ifft`

Inverse Fast Fourier Transform.

**Parameters:**
- Same as `fft`

**Example:**
```python
# Round-trip
original = create_array(shape=[256], fill_type="random")
spectrum = fft(array=original['array_id'])
recovered = ifft(array=spectrum['array_id'])
# recovered ≈ original
```

### Optimization Tools

#### `optimize_function`

Minimize a function.

**Parameters:**
- `function` (string): Function to minimize
- `variables` (array): Variable names
- `initial_guess` (array): Starting point
- `method` (string, optional): Optimization method
  - `"BFGS"` (default) - Quasi-Newton
  - `"Nelder-Mead"` - Simplex
  - `"Powell"` - Powell's method

**Example:**
```python
optimize_function(
    function="(x - 2)**2 + (y - 3)**2",
    variables=["x", "y"],
    initial_guess=[0, 0],
    method="BFGS"
)
# Returns: `{minimum: [2.0, 3.0], function_value: 0.0}`
```

#### `find_roots`

Find roots of equations.

**Parameters:**
- `function` (string | array): Equation(s)
- `variables` (array): Variable names
- `initial_guess` (array): Starting point
- `method` (string, optional): Root-finding method
  - `"fsolve"` (default)
  - `"root"`

**Example:**
```python
find_roots(
    function="x**2 - 4",
    variables=["x"],
    initial_guess=[1.0]
)
# Returns: `{roots: [2.0]}`

# System of equations
find_roots(
    function=["x + y - 5", "x - y - 1"],
    variables=["x", "y"],
    initial_guess=[0, 0]
)
# Returns: `{roots: [3.0, 2.0]}`
```

## Resources

Math MCP provides 3 resource types:

### `constants://math/{name}`

Mathematical constants:
- `pi` - π ≈ 3.14159...
- `e` - Euler's number ≈ 2.71828...
- `golden_ratio` - φ ≈ 1.61803...
- `inf` - Infinity
- `nan` - Not a Number

### `array://{array_id}`

Large numerical arrays created by Math MCP tools.

### `expr://{expression_id}`

Symbolic expressions for reuse across tool calls.

## Performance

GPU vs CPU speedups:

| Operation | Size | CPU Time | GPU Time | Speedup |
|-----------|------|----------|----------|---------|
| Matrix multiply | 100×100 | 5ms | 2ms | 2.5x |
| Matrix multiply | 1000×1000 | 100ms | 1ms | 100x |
| Matrix multiply | 5000×5000 | 25s | 150ms | 167x |
| FFT | 1M points | 500ms | 5ms | 100x |

## Error Handling

Common errors:

| Error | Cause | Solution |
|-------|-------|----------|
| `SympifyError` | Invalid symbolic expression | Check syntax, use `**` for powers |
| `ValueError` | Invalid array shape | Ensure dimensions are compatible |
| `GPUMemoryError` | Array too large for GPU | Reduce size or use CPU |
| `TimeoutError` | Symbolic operation timeout | Simplify expression or increase timeout |

## Next Steps

- **[Quantum MCP](./quantum-mcp)** - Use Math arrays in quantum simulations
- **[Molecular MCP](./molecular-mcp)** - Classical molecular dynamics
- **[Neural MCP](./neural-mcp)** - Neural network training
- **[Cross-MCP Workflows](../concepts/cross-mcp-workflows)** - Combine multiple MCPs
