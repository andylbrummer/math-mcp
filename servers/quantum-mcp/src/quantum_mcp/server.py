"""Quantum MCP server implementation."""

import logging
import uuid
from typing import Any

import numpy as np
from compute_core import fft, ifft
from compute_core.arrays import ensure_array, to_numpy
from mcp.server import Server
from mcp.types import Tool
from mcp_common import GPUManager, TaskManager

logger = logging.getLogger(__name__)

app = Server("quantum-mcp")

# Storage
_potentials: dict[str, Any] = {}
_simulations: dict[str, dict[str, Any]] = {}

# Initialize GPU and task manager
_gpu = GPUManager.get_instance()
_task_manager = TaskManager.get_instance()


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="info",
            description="Progressive discovery of Quantum MCP capabilities",
            inputSchema={"type": "object", "properties": {"topic": {"type": "string"}}},
        ),
        Tool(
            name="create_lattice_potential",
            description="Create crystalline lattice potential",
            inputSchema={
                "type": "object",
                "properties": {
                    "lattice_type": {
                        "type": "string",
                        "enum": ["square", "hexagonal", "triangular"],
                    },
                    "grid_size": {"type": "array", "items": {"type": "integer"}},
                    "depth": {"type": "number", "description": "Potential well depth"},
                    "spacing": {"type": "number", "description": "Lattice spacing"},
                },
                "required": ["lattice_type", "grid_size", "depth"],
            },
        ),
        Tool(
            name="create_custom_potential",
            description="Create custom potential from function or array",
            inputSchema={
                "type": "object",
                "properties": {
                    "grid_size": {"type": "array"},
                    "function": {
                        "type": "string",
                        "description": "Potential function V(x) or V(x,y)",
                    },
                    "array_uri": {"type": "string", "description": "Array URI from Math MCP"},
                },
                "required": ["grid_size"],
            },
        ),
        Tool(
            name="create_gaussian_wavepacket",
            description="Create localized Gaussian wave packet",
            inputSchema={
                "type": "object",
                "properties": {
                    "grid_size": {"type": "array"},
                    "position": {"type": "array", "description": "Center position"},
                    "momentum": {"type": "array", "description": "Initial momentum"},
                    "width": {"type": "number", "default": 5.0},
                },
                "required": ["grid_size", "position", "momentum"],
            },
        ),
        Tool(
            name="create_plane_wave",
            description="Create plane wave state",
            inputSchema={
                "type": "object",
                "properties": {
                    "grid_size": {"type": "array"},
                    "momentum": {"type": "array"},
                },
                "required": ["grid_size", "momentum"],
            },
        ),
        Tool(
            name="solve_schrodinger",
            description="Solve 1D time-dependent Schrödinger equation",
            inputSchema={
                "type": "object",
                "properties": {
                    "potential": {"type": "string", "description": "Potential ID"},
                    "initial_state": {"type": "array", "description": "Initial wavefunction"},
                    "time_steps": {"type": "integer"},
                    "dt": {"type": "number", "description": "Time step"},
                    "use_gpu": {"type": "boolean", "default": False},
                },
                "required": ["potential", "initial_state", "time_steps", "dt"],
            },
        ),
        Tool(
            name="solve_schrodinger_2d",
            description="Solve 2D time-dependent Schrödinger equation",
            inputSchema={
                "type": "object",
                "properties": {
                    "potential": {"type": "string"},
                    "initial_state": {"type": "array"},
                    "time_steps": {"type": "integer"},
                    "dt": {"type": "number"},
                    "use_gpu": {"type": "boolean", "default": False},
                },
                "required": ["potential", "initial_state", "time_steps", "dt"],
            },
        ),
        Tool(
            name="get_task_status",
            description="Monitor async simulation status",
            inputSchema={
                "type": "object",
                "properties": {"task_id": {"type": "string"}},
                "required": ["task_id"],
            },
        ),
        Tool(
            name="get_simulation_result",
            description="Retrieve completed simulation data",
            inputSchema={
                "type": "object",
                "properties": {"simulation_id": {"type": "string"}},
                "required": ["simulation_id"],
            },
        ),
        Tool(
            name="analyze_wavefunction",
            description="Compute observables from wavefunction",
            inputSchema={
                "type": "object",
                "properties": {
                    "wavefunction": {"type": "array"},
                    "dx": {"type": "number", "default": 1.0},
                },
                "required": ["wavefunction"],
            },
        ),
        Tool(
            name="render_video",
            description="Animate probability density evolution",
            inputSchema={
                "type": "object",
                "properties": {
                    "simulation_id": {"type": "string"},
                    "output_path": {"type": "string", "description": "Output video path"},
                    "fps": {"type": "integer", "default": 30},
                },
                "required": ["simulation_id"],
            },
        ),
        Tool(
            name="visualize_potential",
            description="Plot potential energy landscape",
            inputSchema={
                "type": "object",
                "properties": {
                    "potential_id": {"type": "string"},
                    "output_path": {"type": "string"},
                },
                "required": ["potential_id"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[Any]:
    """Handle tool calls."""
    if name == "info":
        return await _tool_info(arguments)
    elif name == "create_lattice_potential":
        return await _tool_create_lattice_potential(arguments)
    elif name == "create_custom_potential":
        return await _tool_create_custom_potential(arguments)
    elif name == "create_gaussian_wavepacket":
        return await _tool_create_gaussian_wavepacket(arguments)
    elif name == "create_plane_wave":
        return await _tool_create_plane_wave(arguments)
    elif name == "solve_schrodinger":
        return await _tool_solve_schrodinger(arguments)
    elif name == "solve_schrodinger_2d":
        return await _tool_solve_schrodinger_2d(arguments)
    elif name == "get_task_status":
        return await _tool_get_task_status(arguments)
    elif name == "get_simulation_result":
        return await _tool_get_simulation_result(arguments)
    elif name == "analyze_wavefunction":
        return await _tool_analyze_wavefunction(arguments)
    elif name == "render_video":
        return await _tool_render_video(arguments)
    elif name == "visualize_potential":
        return await _tool_visualize_potential(arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")


async def _tool_info(args: dict[str, Any]) -> list[Any]:
    """Info tool."""
    topic = args.get("topic", "overview")
    if topic == "overview":
        return [
            {
                "type": "text",
                "text": str(
                    {
                        "categories": [
                            {"name": "potentials", "count": 2},
                            {"name": "wavepackets", "count": 2},
                            {"name": "simulations", "count": 3},
                            {"name": "analysis", "count": 2},
                            {"name": "visualization", "count": 2},
                        ]
                    }
                ),
            }
        ]
    return [{"type": "text", "text": f"Topic: {topic}"}]


async def _tool_create_lattice_potential(args: dict[str, Any]) -> list[Any]:
    """Create lattice potential."""
    lattice_type = args["lattice_type"]
    grid_size = tuple(args["grid_size"])
    depth = args["depth"]
    spacing = args.get("spacing", 1.0)

    # Create potential grid
    if len(grid_size) == 1:
        x = np.arange(grid_size[0])
        V = depth * np.cos(2 * np.pi * x / spacing) ** 2
    else:  # 2D
        x = np.arange(grid_size[0])
        y = np.arange(grid_size[1])
        X, Y = np.meshgrid(x, y, indexing="ij")
        if lattice_type == "square":
            V = depth * (
                np.cos(2 * np.pi * X / spacing) ** 2 + np.cos(2 * np.pi * Y / spacing) ** 2
            )
        elif lattice_type == "hexagonal":
            V = depth * (
                np.cos(2 * np.pi * X / spacing) ** 2
                + np.cos(2 * np.pi * (X / 2 + np.sqrt(3) * Y / 2) / spacing) ** 2
            )
        else:
            V = depth * np.ones_like(X)

    potential_id = str(uuid.uuid4())
    _potentials[potential_id] = V

    return [
        {
            "type": "text",
            "text": str({"potential_id": f"potential://{potential_id}", "shape": V.shape}),
        }
    ]


async def _tool_create_custom_potential(args: dict[str, Any]) -> list[Any]:
    """Create custom potential."""
    grid_size = tuple(args["grid_size"])
    function = args.get("function")

    if function:
        # Evaluate function
        if len(grid_size) == 1:
            x = np.arange(grid_size[0])
            namespace = {"x": x, "np": np, "exp": np.exp, "sin": np.sin, "cos": np.cos}
            V = eval(function, namespace)
        else:
            x = np.arange(grid_size[0])
            y = np.arange(grid_size[1])
            X, Y = np.meshgrid(x, y, indexing="ij")
            namespace = {"x": X, "y": Y, "np": np, "exp": np.exp, "sin": np.sin, "cos": np.cos}
            V = eval(function, namespace)
    else:
        V = np.zeros(grid_size)

    potential_id = str(uuid.uuid4())
    _potentials[potential_id] = V

    return [{"type": "text", "text": str({"potential_id": f"potential://{potential_id}"})}]


async def _tool_create_gaussian_wavepacket(args: dict[str, Any]) -> list[Any]:
    """Create Gaussian wavepacket."""
    grid_size = tuple(args["grid_size"])
    position = np.array(args["position"])
    momentum = np.array(args["momentum"])
    width = args.get("width", 5.0)

    if len(grid_size) == 1:
        x = np.arange(grid_size[0])
        psi = np.exp(-((x - position[0]) ** 2) / (2 * width**2) + 1j * momentum[0] * x)
    else:
        x = np.arange(grid_size[0])
        y = np.arange(grid_size[1])
        X, Y = np.meshgrid(x, y, indexing="ij")
        psi = np.exp(
            -((X - position[0]) ** 2 + (Y - position[1]) ** 2) / (2 * width**2)
            + 1j * (momentum[0] * X + momentum[1] * Y)
        )

    # Normalize
    psi = psi / np.sqrt(np.sum(np.abs(psi) ** 2))

    return [{"type": "text", "text": str({"wavefunction": psi.tolist()})}]


async def _tool_create_plane_wave(args: dict[str, Any]) -> list[Any]:
    """Create plane wave."""
    grid_size = tuple(args["grid_size"])
    momentum = np.array(args["momentum"])

    if len(grid_size) == 1:
        x = np.arange(grid_size[0])
        psi = np.exp(1j * momentum[0] * x)
    else:
        x = np.arange(grid_size[0])
        y = np.arange(grid_size[1])
        X, Y = np.meshgrid(x, y, indexing="ij")
        psi = np.exp(1j * (momentum[0] * X + momentum[1] * Y))

    psi = psi / np.sqrt(np.sum(np.abs(psi) ** 2))

    return [{"type": "text", "text": str({"wavefunction": psi.tolist()})}]


async def _tool_solve_schrodinger(args: dict[str, Any]) -> list[Any]:
    """Solve 1D Schrödinger equation using split-step Fourier method."""
    potential_id = args["potential"].replace("potential://", "")
    if potential_id not in _potentials:
        return [{"type": "text", "text": f"Error: Potential {potential_id} not found"}]

    V = _potentials[potential_id]
    psi0 = np.array(args["initial_state"], dtype=complex)
    time_steps = args["time_steps"]
    dt = args["dt"]
    use_gpu = args.get("use_gpu", False) and _gpu.cuda_available

    # Run simulation asynchronously if time_steps > 100
    if time_steps > 100:

        async def run_simulation() -> dict[str, Any]:
            return _split_step_1d(psi0, V, time_steps, dt, use_gpu)

        task_id = _task_manager.create_task("schrodinger_1d", run_simulation())
        simulation_id = str(uuid.uuid4())

        return [
            {
                "type": "text",
                "text": str(
                    {
                        "task_id": task_id,
                        "simulation_id": f"simulation://{simulation_id}",
                        "status": "running",
                    }
                ),
            }
        ]
    else:
        # Run synchronously
        result = _split_step_1d(psi0, V, time_steps, dt, use_gpu)
        simulation_id = str(uuid.uuid4())
        _simulations[simulation_id] = result

        return [
            {
                "type": "text",
                "text": str(
                    {
                        "simulation_id": f"simulation://{simulation_id}",
                        "status": "completed",
                        "frames": len(result["trajectory"]),
                    }
                ),
            }
        ]


def _split_step_1d(
    psi0: np.ndarray, V: np.ndarray, time_steps: int, dt: float, use_gpu: bool
) -> dict[str, Any]:
    """Split-step Fourier method for 1D Schrödinger equation."""
    psi = ensure_array(psi0, use_gpu=use_gpu)
    V_arr = ensure_array(V, use_gpu=use_gpu)

    N = len(psi)
    dx = 1.0
    k = 2 * np.pi * np.fft.fftfreq(N, dx)
    k_arr = ensure_array(k, use_gpu=use_gpu)

    # Store trajectory
    trajectory = [to_numpy(psi)]
    store_every = max(1, time_steps // 100)  # Store max 100 frames

    # Propagators
    U_V = ensure_array(np.exp(-1j * V_arr * dt / 2), use_gpu=use_gpu)
    U_K = ensure_array(np.exp(-1j * k_arr**2 * dt / 2), use_gpu=use_gpu)

    for step in range(time_steps):
        # Half step in position space
        psi = psi * U_V

        # Full step in momentum space
        psi = fft(psi)
        psi = psi * U_K
        psi = ifft(psi)

        # Half step in position space
        psi = psi * U_V

        if step % store_every == 0:
            trajectory.append(to_numpy(psi))

    return {"trajectory": trajectory, "time_steps": time_steps, "dt": dt}


async def _tool_solve_schrodinger_2d(args: dict[str, Any]) -> list[Any]:
    """Solve 2D Schrödinger equation."""
    # Similar to 1D but with 2D FFT
    return [{"type": "text", "text": "2D solver - implementation similar to 1D"}]


async def _tool_get_task_status(args: dict[str, Any]) -> list[Any]:
    """Get task status."""
    task_id = args["task_id"]
    task = _task_manager.get_task(task_id)

    if task is None:
        return [{"type": "text", "text": "Task not found"}]

    return [
        {
            "type": "text",
            "text": str(
                {
                    "task_id": task_id,
                    "status": task.status.value,
                    "progress": task.progress,
                }
            ),
        }
    ]


async def _tool_get_simulation_result(args: dict[str, Any]) -> list[Any]:
    """Get simulation result."""
    simulation_id = args["simulation_id"].replace("simulation://", "")

    if simulation_id not in _simulations:
        return [{"type": "text", "text": "Simulation not found"}]

    result = _simulations[simulation_id]
    return [
        {
            "type": "text",
            "text": str({"frames": len(result["trajectory"]), "time_steps": result["time_steps"]}),
        }
    ]


async def _tool_analyze_wavefunction(args: dict[str, Any]) -> list[Any]:
    """Analyze wavefunction to compute observables."""
    psi = np.array(args["wavefunction"], dtype=complex)
    dx = args.get("dx", 1.0)

    # Probability density
    prob = np.abs(psi) ** 2

    # Position expectation
    x = np.arange(len(psi)) * dx
    x_avg = np.sum(x * prob) * dx

    # Momentum (via derivative)
    k = 2 * np.pi * np.fft.fftfreq(len(psi), dx)
    psi_k = np.fft.fft(psi)
    p_avg = np.sum(k * np.abs(psi_k) ** 2)

    # Energy
    E = p_avg**2 / 2  # Kinetic only for now

    return [
        {
            "type": "text",
            "text": str(
                {
                    "position": float(x_avg),
                    "momentum": float(p_avg),
                    "energy": float(E),
                    "norm": float(np.sum(prob) * dx),
                }
            ),
        }
    ]


async def _tool_render_video(args: dict[str, Any]) -> list[Any]:
    """Render simulation video."""
    simulation_id = args["simulation_id"].replace("simulation://", "")
    output_path = args.get("output_path", f"/tmp/quantum-sim-{simulation_id}.mp4")

    return [
        {
            "type": "text",
            "text": str(
                {"output_path": output_path, "status": "Video rendering not fully implemented"}
            ),
        }
    ]


async def _tool_visualize_potential(args: dict[str, Any]) -> list[Any]:
    """Visualize potential."""
    potential_id = args["potential_id"].replace("potential://", "")
    output_path = args.get("output_path", f"/tmp/potential-{potential_id}.png")

    return [{"type": "text", "text": str({"output_path": output_path})}]


async def run() -> None:
    """Run the Quantum MCP server."""
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


def main() -> None:
    """Entry point for the quantum-mcp command."""
    import asyncio

    asyncio.run(run())


if __name__ == "__main__":
    main()
