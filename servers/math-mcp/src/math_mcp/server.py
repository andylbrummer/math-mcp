"""Math MCP server implementation."""

import logging
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import sympy
from compute_core import fft as compute_fft
from compute_core import ifft as compute_ifft
from compute_core.arrays import ensure_array, to_numpy
from compute_core.linalg import matmul, solve
from mcp.server import Server
from mcp.types import Resource, Tool
from mcp_common import GPUManager, load_config, serialize_array

logger = logging.getLogger(__name__)

# Initialize server
app = Server("math-mcp")

# Storage for arrays and expressions
_array_cache: dict[str, Any] = {}
_expression_cache: dict[str, sympy.Expr] = {}

# Load config
_config_path = Path(__file__).parent.parent / "config.kdl"
_config = load_config(_config_path) if _config_path.exists() else None

# Initialize GPU
_gpu = GPUManager.get_instance()


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List all available tools."""
    return [
        Tool(
            name="info",
            description="Progressive discovery of Math MCP capabilities",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Topic: overview, symbolic, numerical, etc.",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["table", "json"],
                        "default": "table",
                    },
                },
            },
        ),
        Tool(
            name="symbolic_solve",
            description="Solve symbolic equations using SymPy",
            inputSchema={
                "type": "object",
                "properties": {
                    "equations": {
                        "type": ["string", "array"],
                        "description": "Equation(s) to solve",
                    },
                    "variables": {
                        "type": ["string", "array"],
                        "description": "Variable(s) to solve for",
                    },
                    "domain": {
                        "type": "string",
                        "enum": ["real", "complex", "positive", "integer"],
                        "default": "complex",
                    },
                    "simplify": {"type": "boolean", "default": True},
                },
                "required": ["equations"],
            },
        ),
        Tool(
            name="symbolic_diff",
            description="Compute symbolic derivatives",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Expression to differentiate",
                    },
                    "variable": {
                        "type": "string",
                        "description": "Variable to differentiate with respect to",
                    },
                    "order": {"type": "integer", "default": 1, "minimum": 1},
                },
                "required": ["expression", "variable"],
            },
        ),
        Tool(
            name="symbolic_integrate",
            description="Compute symbolic integrals",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                    "variable": {"type": "string"},
                    "limits": {
                        "type": "array",
                        "description": "Integration limits [lower, upper] for definite integral",
                    },
                },
                "required": ["expression", "variable"],
            },
        ),
        Tool(
            name="symbolic_simplify",
            description="Simplify symbolic expressions",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                    "method": {
                        "type": "string",
                        "enum": ["auto", "trigsimp", "expand", "factor"],
                        "default": "auto",
                    },
                },
                "required": ["expression"],
            },
        ),
        Tool(
            name="create_array",
            description="Create arrays with various initialization patterns",
            inputSchema={
                "type": "object",
                "properties": {
                    "shape": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Array shape",
                    },
                    "fill_type": {
                        "type": "string",
                        "enum": ["zeros", "ones", "random", "linspace", "function"],
                        "default": "zeros",
                    },
                    "dtype": {"type": "string", "default": "float64"},
                    "use_gpu": {"type": "boolean", "default": False},
                    "function": {
                        "type": "string",
                        "description": "Function for fill_type='function' (e.g., 'x**2 + y')",
                    },
                    "linspace_range": {
                        "type": "array",
                        "description": "Range [start, stop] for linspace",
                    },
                },
                "required": ["shape"],
            },
        ),
        Tool(
            name="matrix_multiply",
            description="GPU-accelerated matrix multiplication",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "array", "description": "First matrix"},
                    "b": {"type": "array", "description": "Second matrix"},
                    "use_gpu": {"type": "boolean", "default": False},
                },
                "required": ["a", "b"],
            },
        ),
        Tool(
            name="solve_linear_system",
            description="Solve linear system Ax = b",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "array", "description": "Coefficient matrix"},
                    "b": {"type": "array", "description": "Right-hand side"},
                    "use_gpu": {"type": "boolean", "default": False},
                },
                "required": ["a", "b"],
            },
        ),
        Tool(
            name="fft",
            description="Fast Fourier Transform",
            inputSchema={
                "type": "object",
                "properties": {
                    "array": {
                        "type": ["array", "string"],
                        "description": "Input array or array_id",
                    },
                    "use_gpu": {"type": "boolean", "default": False},
                    "norm": {"type": "string", "enum": ["backward", "ortho", "forward"]},
                },
                "required": ["array"],
            },
        ),
        Tool(
            name="ifft",
            description="Inverse Fast Fourier Transform",
            inputSchema={
                "type": "object",
                "properties": {
                    "array": {"type": ["array", "string"]},
                    "use_gpu": {"type": "boolean", "default": False},
                    "norm": {"type": "string", "enum": ["backward", "ortho", "forward"]},
                },
                "required": ["array"],
            },
        ),
        Tool(
            name="optimize_function",
            description="Minimize a function",
            inputSchema={
                "type": "object",
                "properties": {
                    "function": {"type": "string", "description": "Function to minimize"},
                    "variables": {"type": "array", "description": "Variable names"},
                    "initial_guess": {"type": "array", "description": "Initial guess"},
                    "method": {
                        "type": "string",
                        "enum": ["BFGS", "Nelder-Mead", "Powell"],
                        "default": "BFGS",
                    },
                },
                "required": ["function", "variables", "initial_guess"],
            },
        ),
        Tool(
            name="find_roots",
            description="Find roots of equations",
            inputSchema={
                "type": "object",
                "properties": {
                    "function": {"type": "string"},
                    "variables": {"type": "array"},
                    "initial_guess": {"type": "array"},
                    "method": {"type": "string", "enum": ["fsolve", "root"], "default": "fsolve"},
                },
                "required": ["function", "variables", "initial_guess"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[Any]:
    """Handle tool calls."""
    handlers = {
        "info": _tool_info,
        "symbolic_solve": _tool_symbolic_solve,
        "symbolic_diff": _tool_symbolic_diff,
        "symbolic_integrate": _tool_symbolic_integrate,
        "symbolic_simplify": _tool_symbolic_simplify,
        "create_array": _tool_create_array,
        "matrix_multiply": _tool_matrix_multiply,
        "solve_linear_system": _tool_solve_linear_system,
        "fft": _tool_fft,
        "ifft": _tool_ifft,
        "optimize_function": _tool_optimize_function,
        "find_roots": _tool_find_roots,
    }
    handler = handlers.get(name)
    if handler is None:
        msg = f"Unknown tool: {name}"
        raise ValueError(msg)
    return await handler(arguments)


@app.list_resources()
async def list_resources() -> list[Resource]:
    """List available resources."""
    resources = [
        Resource(
            uri="constants://math/pi",  # type: ignore[arg-type]
            name="Mathematical constant π",
            mimeType="application/json",
        ),
        Resource(
            uri="constants://math/e",  # type: ignore[arg-type]
            name="Mathematical constant e",
            mimeType="application/json",
        ),
        Resource(
            uri="constants://math/golden_ratio",  # type: ignore[arg-type]
            name="Golden ratio φ",
            mimeType="application/json",
        ),
    ]

    # Add cached arrays
    for array_id in _array_cache:
        resources.append(
            Resource(
                uri=f"array://{array_id}",  # type: ignore[arg-type]
                name=f"Cached array {array_id}",
                mimeType="application/json",
            )
        )

    # Add cached expressions
    for expr_id in _expression_cache:
        resources.append(
            Resource(
                uri=f"expr://{expr_id}",  # type: ignore[arg-type]
                name=f"Symbolic expression {expr_id}",
                mimeType="application/json",
            )
        )

    return resources


@app.read_resource()
async def read_resource(uri: str) -> str:
    """Read resource by URI."""
    if uri.startswith("constants://math/"):
        constant = uri.split("/")[-1]
        constants_map = {
            "pi": float(np.pi),
            "e": float(np.e),
            "golden_ratio": (1 + np.sqrt(5)) / 2,
        }
        if constant in constants_map:
            return str({"value": constants_map[constant], "name": constant})
        msg = f"Unknown constant: {constant}"
        raise ValueError(msg)

    if uri.startswith("array://"):
        array_id = uri.replace("array://", "")
        if array_id in _array_cache:
            arr = _array_cache[array_id]
            metadata = serialize_array(arr, array_id, force_inline=True)
            return str(metadata)
        msg = f"Array not found: {array_id}"
        raise ValueError(msg)

    if uri.startswith("expr://"):
        expr_id = uri.replace("expr://", "")
        if expr_id in _expression_cache:
            expr = _expression_cache[expr_id]
            return str({"expression": str(expr), "latex": sympy.latex(expr)})
        msg = f"Expression not found: {expr_id}"
        raise ValueError(msg)

    msg = f"Unknown resource URI: {uri}"
    raise ValueError(msg)


# Tool implementations


async def _tool_info(args: dict[str, Any]) -> list[Any]:
    """Info tool implementation."""
    topic = args.get("topic", "overview")

    info_data = {
        "overview": {
            "categories": [
                {"name": "symbolic", "count": 4, "description": "SymPy symbolic math"},
                {"name": "numerical", "count": 3, "description": "Array operations"},
                {"name": "transforms", "count": 2, "description": "FFT operations"},
                {"name": "optimization", "count": 2, "description": "Minimization & roots"},
            ]
        },
        "symbolic": {
            "tools": [
                {"name": "symbolic_solve", "description": "Solve equations", "gpu": False},
                {"name": "symbolic_diff", "description": "Derivatives", "gpu": False},
                {"name": "symbolic_integrate", "description": "Integrals", "gpu": False},
                {"name": "symbolic_simplify", "description": "Simplify expressions", "gpu": False},
            ]
        },
        "numerical": {
            "tools": [
                {"name": "create_array", "description": "Create arrays", "gpu": True},
                {"name": "matrix_multiply", "description": "Matrix multiplication", "gpu": True},
                {"name": "solve_linear_system", "description": "Solve Ax=b", "gpu": True},
            ]
        },
        "transforms": {
            "tools": [
                {"name": "fft", "description": "Fast Fourier Transform", "gpu": True},
                {"name": "ifft", "description": "Inverse FFT", "gpu": True},
            ]
        },
        "optimization": {
            "tools": [
                {"name": "optimize_function", "description": "Function minimization", "gpu": False},
                {"name": "find_roots", "description": "Root finding", "gpu": False},
            ]
        },
    }

    if topic == "overview":
        return [{"type": "text", "text": str(info_data["overview"])}]
    if topic in info_data:
        return [{"type": "text", "text": str(info_data[topic])}]
    available = "overview, symbolic, numerical, transforms, optimization"
    return [{"type": "text", "text": f"Topic '{topic}' not found. Available: {available}"}]


async def _tool_symbolic_solve(args: dict[str, Any]) -> list[Any]:
    """Solve symbolic equations."""
    equations = args["equations"]
    if isinstance(equations, str):
        equations = [equations]

    variables = args.get("variables")

    try:
        # Parse equations
        eqs = []
        for eq in equations:
            if "=" in eq:
                lhs, rhs = eq.split("=")
                eqs.append(sympy.sympify(lhs, evaluate=False) - sympy.sympify(rhs, evaluate=False))
            else:
                eqs.append(sympy.sympify(eq, evaluate=False))

        # Detect variables if not provided
        if not variables:
            all_symbols = set()
            for eq in eqs:
                all_symbols.update(eq.free_symbols)
            variables = [str(s) for s in all_symbols]
        elif isinstance(variables, str):
            variables = [variables]

        # Create symbols
        syms = [sympy.Symbol(v) for v in variables]

        # Solve
        solutions = sympy.solve(eqs, syms, dict=True)

        # Format results
        result = {
            "solutions": [{str(k): str(v) for k, v in sol.items()} for sol in solutions],
            "latex": sympy.latex(solutions),
            "count": len(solutions),
        }

        # Store expression if requested
        if solutions:
            expr_id = str(uuid.uuid4())
            _expression_cache[expr_id] = solutions[0] if len(solutions) == 1 else solutions
            result["expression_id"] = f"expr://{expr_id}"

        return [{"type": "text", "text": str(result)}]

    except Exception:
        logger.exception("Error solving equations")
        return [{"type": "text", "text": "Error solving equations"}]


async def _tool_symbolic_diff(args: dict[str, Any]) -> list[Any]:
    """Compute symbolic derivative."""
    expression = args["expression"]
    variable = args["variable"]
    order = args.get("order", 1)

    try:
        expr = sympy.sympify(expression, evaluate=False)
        var = sympy.Symbol(variable)

        result = sympy.diff(expr, var, order)

        return [
            {
                "type": "text",
                "text": str(
                    {
                        "derivative": str(result),
                        "latex": sympy.latex(result),
                        "order": order,
                    }
                ),
            }
        ]
    except Exception as e:
        return [{"type": "text", "text": f"Error: {e}"}]


async def _tool_symbolic_integrate(args: dict[str, Any]) -> list[Any]:
    """Compute symbolic integral."""
    expression = args["expression"]
    variable = args["variable"]
    limits = args.get("limits")

    try:
        expr = sympy.sympify(expression, evaluate=False)
        var = sympy.Symbol(variable)

        if limits:
            result = sympy.integrate(expr, (var, limits[0], limits[1]))
            integral_type = "definite"
        else:
            result = sympy.integrate(expr, var)
            integral_type = "indefinite"

        return [
            {
                "type": "text",
                "text": str(
                    {
                        "integral": str(result),
                        "latex": sympy.latex(result),
                        "type": integral_type,
                    }
                ),
            }
        ]
    except Exception as e:
        return [{"type": "text", "text": f"Error: {e}"}]


async def _tool_symbolic_simplify(args: dict[str, Any]) -> list[Any]:
    """Simplify symbolic expression."""
    expression = args["expression"]
    method = args.get("method", "auto")

    try:
        expr = sympy.sympify(expression, evaluate=False)

        if method == "auto":
            result = sympy.simplify(expr)
        elif method == "trigsimp":
            result = sympy.trigsimp(expr)
        elif method == "expand":
            result = sympy.expand(expr)
        elif method == "factor":
            result = sympy.factor(expr)
        else:
            result = sympy.simplify(expr)

        return [
            {
                "type": "text",
                "text": str(
                    {
                        "simplified": str(result),
                        "latex": sympy.latex(result),
                        "method": method,
                    }
                ),
            }
        ]
    except Exception as e:
        return [{"type": "text", "text": f"Error: {e}"}]


async def _tool_create_array(args: dict[str, Any]) -> list[Any]:
    """Create array with specified pattern."""
    shape = tuple(args["shape"])
    fill_type = args.get("fill_type", "zeros")
    dtype = args.get("dtype", "float64")
    use_gpu = args.get("use_gpu", False) and _gpu.cuda_available

    try:
        # Check size limits
        total_size = int(np.prod(shape))
        if _config and total_size > _config.limits.max_array_size:
            max_size = _config.limits.max_array_size
            return [{"type": "text", "text": f"Error: Array size {total_size} exceeds {max_size}"}]

        # Create array
        if fill_type == "zeros":
            arr = ensure_array(np.zeros(shape, dtype=dtype), use_gpu=use_gpu)
        elif fill_type == "ones":
            arr = ensure_array(np.ones(shape, dtype=dtype), use_gpu=use_gpu)
        elif fill_type == "random":
            rng = np.random.default_rng()
            arr = ensure_array(rng.random(shape).astype(dtype), use_gpu=use_gpu)
        elif fill_type == "linspace":
            linspace_range = args.get("linspace_range", [0, 1])
            arr = ensure_array(
                np.linspace(linspace_range[0], linspace_range[1], total_size)
                .reshape(shape)
                .astype(dtype),
                use_gpu=use_gpu,
            )
        elif fill_type == "function":
            # Create coordinate grids and evaluate function
            func_str = args.get("function", "0")
            coords = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
            namespace: dict[str, Any] = {"x": coords[0] if len(coords) > 0 else 0}
            if len(coords) > 1:
                namespace["y"] = coords[1]
            if len(coords) > 2:
                namespace["z"] = coords[2]
            namespace.update(
                {"np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp, "sqrt": np.sqrt}
            )

            arr = ensure_array(eval(func_str, namespace).astype(dtype), use_gpu=use_gpu)
        else:
            return [{"type": "text", "text": f"Error: Unknown fill_type '{fill_type}'"}]

        # Store array
        array_id = str(uuid.uuid4())
        _array_cache[array_id] = arr

        # Serialize metadata
        metadata = serialize_array(arr, array_id)

        return [
            {"type": "text", "text": str({"array_id": f"array://{array_id}", "metadata": metadata})}
        ]

    except Exception:
        logger.exception("Error creating array")
        return [{"type": "text", "text": "Error creating array"}]


async def _tool_matrix_multiply(args: dict[str, Any]) -> list[Any]:
    """Matrix multiplication."""
    a = ensure_array(args["a"], use_gpu=args.get("use_gpu", False) and _gpu.cuda_available)
    b = ensure_array(args["b"], use_gpu=args.get("use_gpu", False) and _gpu.cuda_available)

    try:
        result = matmul(a, b)

        # Store result
        array_id = str(uuid.uuid4())
        _array_cache[array_id] = result

        metadata = serialize_array(result, array_id)

        return [
            {"type": "text", "text": str({"array_id": f"array://{array_id}", "metadata": metadata})}
        ]

    except Exception as e:
        return [{"type": "text", "text": f"Error: {e}"}]


async def _tool_solve_linear_system(args: dict[str, Any]) -> list[Any]:
    """Solve linear system."""
    a = ensure_array(args["a"], use_gpu=args.get("use_gpu", False) and _gpu.cuda_available)
    b = ensure_array(args["b"], use_gpu=args.get("use_gpu", False) and _gpu.cuda_available)

    try:
        result = solve(a, b)

        # Store result
        array_id = str(uuid.uuid4())
        _array_cache[array_id] = result

        metadata = serialize_array(result, array_id)

        return [
            {
                "type": "text",
                "text": str(
                    {
                        "array_id": f"array://{array_id}",
                        "metadata": metadata,
                        "solution": to_numpy(result).tolist(),
                    }
                ),
            }
        ]

    except Exception as e:
        return [{"type": "text", "text": f"Error: {e}"}]


async def _tool_fft(args: dict[str, Any]) -> list[Any]:
    """Fast Fourier Transform."""
    array_input = args["array"]

    # Handle array_id or direct array
    if isinstance(array_input, str) and array_input.startswith("array://"):
        array_id = array_input.replace("array://", "")
        if array_id not in _array_cache:
            return [{"type": "text", "text": f"Error: Array {array_id} not found"}]
        arr = _array_cache[array_id]
    else:
        arr = ensure_array(array_input, use_gpu=args.get("use_gpu", False) and _gpu.cuda_available)

    try:
        result = compute_fft(arr, norm=args.get("norm"))

        # Store result
        array_id = str(uuid.uuid4())
        _array_cache[array_id] = result

        metadata = serialize_array(result, array_id)

        return [
            {"type": "text", "text": str({"array_id": f"array://{array_id}", "metadata": metadata})}
        ]

    except Exception as e:
        return [{"type": "text", "text": f"Error: {e}"}]


async def _tool_ifft(args: dict[str, Any]) -> list[Any]:
    """Inverse Fast Fourier Transform."""
    array_input = args["array"]

    if isinstance(array_input, str) and array_input.startswith("array://"):
        array_id = array_input.replace("array://", "")
        if array_id not in _array_cache:
            return [{"type": "text", "text": f"Error: Array {array_id} not found"}]
        arr = _array_cache[array_id]
    else:
        arr = ensure_array(array_input, use_gpu=args.get("use_gpu", False) and _gpu.cuda_available)

    try:
        result = compute_ifft(arr, norm=args.get("norm"))

        # Store result
        array_id = str(uuid.uuid4())
        _array_cache[array_id] = result

        metadata = serialize_array(result, array_id)

        return [
            {"type": "text", "text": str({"array_id": f"array://{array_id}", "metadata": metadata})}
        ]

    except Exception as e:
        return [{"type": "text", "text": f"Error: {e}"}]


async def _tool_optimize_function(args: dict[str, Any]) -> list[Any]:
    """Optimize (minimize) a function."""
    from scipy.optimize import minimize  # noqa: PLC0415

    func_str = args["function"]
    variables = args["variables"]
    initial_guess = args["initial_guess"]
    method = args.get("method", "BFGS")

    try:
        # Create objective function
        def objective(x: np.ndarray) -> float:
            namespace = dict(zip(variables, x, strict=False))
            namespace.update(
                {"np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp, "sqrt": np.sqrt}
            )
            return float(eval(func_str, namespace))

        # Optimize
        result = minimize(objective, initial_guess, method=method)

        return [
            {
                "type": "text",
                "text": str(
                    {
                        "success": bool(result.success),
                        "x": result.x.tolist(),
                        "fun": float(result.fun),
                        "message": result.message,
                    }
                ),
            }
        ]

    except Exception as e:
        return [{"type": "text", "text": f"Error: {e}"}]


async def _tool_find_roots(args: dict[str, Any]) -> list[Any]:
    """Find roots of equations."""
    from scipy.optimize import fsolve  # noqa: PLC0415

    func_str = args["function"]
    variables = args["variables"]
    initial_guess = args["initial_guess"]

    try:
        # Create function
        def func(x: np.ndarray) -> np.ndarray:
            namespace = dict(zip(variables, x, strict=False))
            namespace.update(
                {"np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp, "sqrt": np.sqrt}
            )
            result = eval(func_str, namespace)
            return np.atleast_1d(result)

        # Find roots
        roots = fsolve(func, initial_guess)

        return [
            {
                "type": "text",
                "text": str(
                    {
                        "roots": roots.tolist(),
                        "variables": variables,
                    }
                ),
            }
        ]

    except Exception as e:
        return [{"type": "text", "text": f"Error: {e}"}]


async def run() -> None:
    """Run the Math MCP server."""
    from mcp.server.stdio import stdio_server  # noqa: PLC0415

    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


def main() -> None:
    """Entry point for the math-mcp command."""
    import asyncio  # noqa: PLC0415

    asyncio.run(run())


if __name__ == "__main__":
    main()
