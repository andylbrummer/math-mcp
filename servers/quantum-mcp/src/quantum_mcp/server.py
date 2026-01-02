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
_wavefunctions: dict[str, np.ndarray] = {}
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
                    "width": {"type": "number", "description": "Point center width (default 2.0)"},
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
                    "width": {
                        "description": "Number (isotropic) or [width_x, width_y] (elliptical)",
                        "oneOf": [{"type": "number"}, {"type": "array"}],
                        "default": 5.0,
                    },
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
                    "potential": {
                        "type": "string",
                        "description": "Potential ID (potential://...)",
                    },
                    "initial_state": {
                        "description": "Wavefunction ID (wavefunction://...) or array",
                        "oneOf": [{"type": "string"}, {"type": "array"}],
                    },
                    "time_steps": {"type": "integer"},
                    "dt": {"type": "number"},
                    "use_gpu": {"type": "boolean", "default": False},
                    "boundary": {
                        "type": "object",
                        "description": "Boundary conditions",
                        "properties": {
                            "left": {
                                "type": "string",
                                "enum": ["absorb", "reflect", "periodic"],
                                "default": "periodic",
                            },
                            "right": {
                                "type": "string",
                                "enum": ["absorb", "reflect", "periodic"],
                                "default": "periodic",
                            },
                            "absorb_width": {"type": "number", "default": 20},
                            "absorb_strength": {"type": "number", "default": 0.05},
                        },
                    },
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
                    "sensor_line": {
                        "type": "integer",
                        "description": "X position of detector screen to show intensity pattern",
                    },
                    "show_potential": {
                        "type": "boolean",
                        "default": False,
                        "description": "Overlay potential contours on animation",
                    },
                    "accumulate_sensor": {
                        "type": "boolean",
                        "default": True,
                        "description": "Accumulate intensity at sensor over time",
                    },
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
    handlers = {
        "info": _tool_info,
        "create_lattice_potential": _tool_create_lattice_potential,
        "create_custom_potential": _tool_create_custom_potential,
        "create_gaussian_wavepacket": _tool_create_gaussian_wavepacket,
        "create_plane_wave": _tool_create_plane_wave,
        "solve_schrodinger": _tool_solve_schrodinger,
        "solve_schrodinger_2d": _tool_solve_schrodinger_2d,
        "get_task_status": _tool_get_task_status,
        "get_simulation_result": _tool_get_simulation_result,
        "analyze_wavefunction": _tool_analyze_wavefunction,
        "render_video": _tool_render_video,
        "visualize_potential": _tool_visualize_potential,
    }
    handler = handlers.get(name)
    if handler is None:
        msg = f"Unknown tool: {name}"
        raise ValueError(msg)
    return await handler(arguments)


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


def _create_2d_lattice(
    lattice_type: str, nx: int, ny: int, spacing: float, depth: float, width: float
) -> np.ndarray:
    """Create 2D lattice potential with Gaussian point centers."""
    x = np.arange(nx)
    y = np.arange(ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    potential = np.zeros((nx, ny))

    if lattice_type == "square":
        for cx in np.arange(spacing / 2, nx, spacing):
            for cy in np.arange(spacing / 2, ny, spacing):
                r2 = (xx - cx) ** 2 + (yy - cy) ** 2
                potential += depth * np.exp(-r2 / (2 * width**2))
    elif lattice_type in ("hexagonal", "triangular"):
        # Hexagonal/triangular: offset every other row
        cx_values = list(np.arange(spacing / 2, nx, spacing * np.sqrt(3) / 2))
        for row, cx in enumerate(cx_values):
            offset = (spacing / 2) if row % 2 else 0
            for cy in np.arange(spacing / 2 + offset, ny, spacing):
                r2 = (xx - cx) ** 2 + (yy - cy) ** 2
                potential += depth * np.exp(-r2 / (2 * width**2))
    return potential


async def _tool_create_lattice_potential(args: dict[str, Any]) -> list[Any]:
    """Create lattice potential with tight point-like scattering centers."""
    lattice_type = args["lattice_type"]
    grid_size = tuple(args["grid_size"])
    depth = args["depth"]
    spacing = args.get("spacing", 20.0)
    width = args.get("width", 2.0)  # Gaussian width for point centers

    # Create potential grid with Gaussian point centers
    if len(grid_size) == 1:
        x = np.arange(grid_size[0])
        potential = np.zeros(grid_size[0])
        for cx in np.arange(0, grid_size[0], spacing):
            potential += depth * np.exp(-((x - cx) ** 2) / (2 * width**2))
    else:
        nx, ny = grid_size[0], grid_size[1]
        potential = _create_2d_lattice(lattice_type, nx, ny, spacing, depth, width)

    potential_id = str(uuid.uuid4())
    _potentials[potential_id] = potential

    return [
        {
            "type": "text",
            "text": str({"potential_id": f"potential://{potential_id}", "shape": potential.shape}),
        }
    ]


async def _tool_create_custom_potential(args: dict[str, Any]) -> list[Any]:
    """Create custom potential."""
    grid_size = tuple(args["grid_size"])
    function = args.get("function")

    if function:
        # Evaluate function (user-provided mathematical expression)
        if len(grid_size) == 1:
            x = np.arange(grid_size[0])
            namespace = {"x": x, "np": np, "exp": np.exp, "sin": np.sin, "cos": np.cos}
            potential = eval(function, namespace)
        else:
            x = np.arange(grid_size[0])
            y = np.arange(grid_size[1])
            xx, yy = np.meshgrid(x, y, indexing="ij")
            namespace = {"x": xx, "y": yy, "np": np, "exp": np.exp, "sin": np.sin, "cos": np.cos}
            potential = eval(function, namespace)
    else:
        potential = np.zeros(grid_size)

    potential_id = str(uuid.uuid4())
    _potentials[potential_id] = potential

    return [{"type": "text", "text": str({"potential_id": f"potential://{potential_id}"})}]


async def _tool_create_gaussian_wavepacket(args: dict[str, Any]) -> list[Any]:
    """Create Gaussian wavepacket and store it."""
    grid_size = tuple(args["grid_size"])
    position = np.array(args["position"])
    momentum = np.array(args["momentum"])
    width_arg = args.get("width", 5.0)

    # Parse width - can be number (isotropic) or [width_x, width_y] (elliptical)
    if isinstance(width_arg, (list, tuple)):
        width_x, width_y = width_arg[0], width_arg[1]
    else:
        width_x = width_y = width_arg

    if len(grid_size) == 1:
        x = np.arange(grid_size[0])
        psi = np.exp(-((x - position[0]) ** 2) / (2 * width_x**2) + 1j * momentum[0] * x)
    else:
        x = np.arange(grid_size[0])
        y = np.arange(grid_size[1])
        xx, yy = np.meshgrid(x, y, indexing="ij")
        # Elliptical Gaussian with separate widths for x and y
        psi = np.exp(
            -((xx - position[0]) ** 2) / (2 * width_x**2)
            - ((yy - position[1]) ** 2) / (2 * width_y**2)
            + 1j * (momentum[0] * xx + momentum[1] * yy)
        )

    # Normalize
    psi = psi / np.sqrt(np.sum(np.abs(psi) ** 2))

    # Store and return ID
    wavefunction_id = str(uuid.uuid4())
    _wavefunctions[wavefunction_id] = psi

    return [
        {
            "type": "text",
            "text": str(
                {
                    "wavefunction_id": f"wavefunction://{wavefunction_id}",
                    "shape": list(psi.shape),
                    "norm": float(np.sum(np.abs(psi) ** 2)),
                }
            ),
        }
    ]


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
        xx, yy = np.meshgrid(x, y, indexing="ij")
        psi = np.exp(1j * (momentum[0] * xx + momentum[1] * yy))

    psi = psi / np.sqrt(np.sum(np.abs(psi) ** 2))

    # Store and return ID
    wavefunction_id = str(uuid.uuid4())
    _wavefunctions[wavefunction_id] = psi

    return [
        {
            "type": "text",
            "text": str(
                {
                    "wavefunction_id": f"wavefunction://{wavefunction_id}",
                    "shape": list(psi.shape),
                    "norm": float(np.sum(np.abs(psi) ** 2)),
                }
            ),
        }
    ]


async def _tool_solve_schrodinger(args: dict[str, Any]) -> list[Any]:
    """Solve 1D Schrödinger equation using split-step Fourier method."""
    potential_id = args["potential"].replace("potential://", "")
    if potential_id not in _potentials:
        return [{"type": "text", "text": f"Error: Potential {potential_id} not found"}]

    v = _potentials[potential_id]

    # Handle wavefunction_id or raw array
    initial_state = args["initial_state"]
    if isinstance(initial_state, str) and initial_state.startswith("wavefunction://"):
        wf_id = initial_state.replace("wavefunction://", "")
        if wf_id not in _wavefunctions:
            return [{"type": "text", "text": f"Error: Wavefunction {wf_id} not found"}]
        psi0 = _wavefunctions[wf_id]
    else:
        psi0 = np.array(initial_state, dtype=complex)
    time_steps = args["time_steps"]
    dt = args["dt"]
    use_gpu = args.get("use_gpu", False) and _gpu.cuda_available

    # Run simulation asynchronously if time_steps > 100
    if time_steps > 100:

        async def run_simulation() -> dict[str, Any]:
            return _split_step_1d(psi0, v, time_steps, dt, use_gpu)

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
    # Run synchronously
    result = _split_step_1d(psi0, v, time_steps, dt, use_gpu)
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
    psi0: np.ndarray, potential: np.ndarray, time_steps: int, dt: float, use_gpu: bool
) -> dict[str, Any]:
    """Split-step Fourier method for 1D Schrödinger equation."""
    psi = ensure_array(psi0, use_gpu=use_gpu)
    v_arr = ensure_array(potential, use_gpu=use_gpu)

    n_points = len(psi)
    dx = 1.0
    k = 2 * np.pi * np.fft.fftfreq(n_points, dx)
    k_arr = ensure_array(k, use_gpu=use_gpu)

    # Store trajectory
    trajectory = [to_numpy(psi)]
    store_every = max(1, time_steps // 100)  # Store max 100 frames

    # Propagators
    u_v = ensure_array(np.exp(-1j * v_arr * dt / 2), use_gpu=use_gpu)
    u_k = ensure_array(np.exp(-1j * k_arr**2 * dt / 2), use_gpu=use_gpu)

    for step in range(time_steps):
        # Half step in position space
        psi = psi * u_v

        # Full step in momentum space
        psi = fft(psi)
        psi = psi * u_k
        psi = ifft(psi)

        # Half step in position space
        psi = psi * u_v

        if step % store_every == 0:
            trajectory.append(to_numpy(psi))

    return {"trajectory": trajectory, "time_steps": time_steps, "dt": dt}


async def _tool_solve_schrodinger_2d(args: dict[str, Any]) -> list[Any]:
    """Solve 2D Schrödinger equation using split-step Fourier method."""
    potential_id = args["potential"].replace("potential://", "")
    if potential_id not in _potentials:
        return [{"type": "text", "text": f"Error: Potential {potential_id} not found"}]

    potential = _potentials[potential_id]

    # Get initial state - can be wavefunction ID or raw array
    initial_state = args["initial_state"]
    if isinstance(initial_state, str) and initial_state.startswith("wavefunction://"):
        wf_id = initial_state.replace("wavefunction://", "")
        if wf_id not in _wavefunctions:
            return [{"type": "text", "text": f"Error: Wavefunction {wf_id} not found"}]
        psi0 = _wavefunctions[wf_id]
    else:
        psi0 = np.array(initial_state, dtype=complex)

    time_steps = args["time_steps"]
    dt = args["dt"]
    use_gpu = args.get("use_gpu", False) and _gpu.cuda_available
    boundary = args.get("boundary", {})

    # Run simulation
    result = _split_step_2d(psi0, potential, time_steps, dt, use_gpu, boundary)
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
                    "grid_size": list(potential.shape),
                }
            ),
        }
    ]


def _create_absorbing_mask(nx: int, ny: int, boundary: dict, dt: float) -> np.ndarray:
    """Create absorbing boundary mask (imaginary potential)."""
    left_bc = boundary.get("left", "periodic")
    right_bc = boundary.get("right", "periodic")
    width = int(boundary.get("absorb_width", 20))
    strength = boundary.get("absorb_strength", 0.05)

    # Create smooth absorbing potential at boundaries
    absorb = np.zeros((nx, ny))
    x = np.arange(nx)

    if left_bc == "absorb":
        # Smooth ramp from edge
        left_ramp = np.clip((width - x) / width, 0, 1) ** 2
        absorb += strength * left_ramp[:, np.newaxis]

    if right_bc == "absorb":
        right_ramp = np.clip((x - (nx - width)) / width, 0, 1) ** 2
        absorb += strength * right_ramp[:, np.newaxis]

    # Convert to decay factor per time step
    return np.exp(-absorb * dt)


def _split_step_2d(
    psi0: np.ndarray,
    potential: np.ndarray,
    time_steps: int,
    dt: float,
    _use_gpu: bool,
    boundary: dict | None = None,
) -> dict[str, Any]:
    """Split-step Fourier method for 2D Schrödinger equation."""
    # Note: _use_gpu reserved for future GPU acceleration
    if boundary is None:
        boundary = {}

    psi = psi0.copy().astype(complex)
    nx, ny = potential.shape
    dx = 1.0

    # Momentum space grids
    kx = 2 * np.pi * np.fft.fftfreq(nx, dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, dx)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    k_squared = kx_grid**2 + ky_grid**2

    # Propagators
    u_v = np.exp(-1j * potential * dt / 2)
    u_k = np.exp(-1j * k_squared * dt / 2)

    # Absorbing boundary mask
    absorb_mask = _create_absorbing_mask(nx, ny, boundary, dt)

    # Store trajectory (probability density only to save memory)
    store_every = max(1, time_steps // 100)
    trajectory = [np.abs(psi) ** 2]

    for step in range(time_steps):
        # Split-step Fourier
        psi = psi * u_v
        psi = np.fft.ifft2(np.fft.fft2(psi) * u_k)
        psi = psi * u_v

        # Apply absorbing boundaries
        psi = psi * absorb_mask

        if (step + 1) % store_every == 0:
            trajectory.append(np.abs(psi) ** 2)

    return {
        "trajectory": trajectory,
        "time_steps": time_steps,
        "dt": dt,
        "potential": potential,
        "final_state": psi,
    }


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

    # Energy (kinetic only for now)
    energy = p_avg**2 / 2

    return [
        {
            "type": "text",
            "text": str(
                {
                    "position": float(x_avg),
                    "momentum": float(p_avg),
                    "energy": float(energy),
                    "norm": float(np.sum(prob) * dx),
                }
            ),
        }
    ]


async def _tool_render_video(args: dict[str, Any]) -> list[Any]:  # noqa: PLR0915
    """Render simulation video as animated GIF or MP4."""
    from pathlib import Path  # noqa: PLC0415

    import matplotlib as mpl  # noqa: PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib import animation  # noqa: PLC0415

    mpl.use("Agg")

    simulation_id = args["simulation_id"].replace("simulation://", "")
    output_path = args.get("output_path", f"/tmp/quantum-sim-{simulation_id}.gif")
    fps = args.get("fps", 30)
    sensor_line = args.get("sensor_line")
    show_potential = args.get("show_potential", False)
    accumulate_sensor = args.get("accumulate_sensor", True)

    if simulation_id not in _simulations:
        return [{"type": "text", "text": f"Error: Simulation {simulation_id} not found"}]

    result = _simulations[simulation_id]
    trajectory = result["trajectory"]
    potential = result.get("potential")

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Determine if 1D or 2D
    if trajectory[0].ndim == 1:
        # 1D animation - simple plot
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.patch.set_facecolor("#0a0a1a")
        ax.set_facecolor("#0a0a1a")
        (line,) = ax.plot([], [], color="cyan", linewidth=2)
        ax.set_xlim(0, len(trajectory[0]))
        ax.set_ylim(0, np.max([np.max(t) for t in trajectory]) * 1.1)
        ax.set_xlabel("Position", color="white")
        ax.set_ylabel("|ψ|²", color="white")
        ax.tick_params(colors="white")

        def animate(frame: int) -> tuple:
            line.set_data(np.arange(len(trajectory[frame])), trajectory[frame])
            return (line,)

    else:
        # 2D animation with optional sensor line
        if sensor_line is not None:
            fig, (ax, ax_sensor) = plt.subplots(
                1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [2, 1]}
            )
            fig.patch.set_facecolor("#0a0a1a")
            ax_sensor.set_facecolor("#0a0a1a")
        else:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax_sensor = None

        fig.patch.set_facecolor("#0a0a1a")
        ax.set_facecolor("#0a0a1a")

        vmax = np.percentile([np.max(t) for t in trajectory], 95)
        im = ax.imshow(
            trajectory[0].T,
            origin="lower",
            cmap="viridis",
            vmin=0,
            vmax=vmax,
            aspect="equal",
        )

        # Show potential overlay
        if show_potential and potential is not None:
            pot_max = np.max(potential)
            if pot_max > 0:
                ax.contour(
                    potential.T,
                    levels=[pot_max * 0.5],
                    colors=["red"],
                    alpha=0.7,
                    linewidths=1.5,
                )

        # Sensor line visualization
        sensor_accum = None
        sensor_line_plot = None
        sensor_max = None
        if sensor_line is not None and ax_sensor is not None:
            # Draw sensor line on main plot
            ax.axvline(x=sensor_line, color="yellow", linestyle="--", linewidth=1.5, alpha=0.7)
            ny = trajectory[0].shape[1]
            sensor_accum = np.zeros(ny)
            # Pre-compute max accumulated intensity for fixed scale
            temp_accum = np.zeros(ny)
            for frame in trajectory:
                temp_accum = temp_accum + frame[sensor_line, :]
            sensor_max = np.max(temp_accum) * 1.1
            (sensor_line_plot,) = ax_sensor.plot(
                np.zeros(ny), np.arange(ny), color="cyan", linewidth=2
            )
            ax_sensor.set_xlim(0, sensor_max)
            ax_sensor.set_ylim(0, ny)
            ax_sensor.set_xlabel("Accumulated Intensity", color="white")
            ax_sensor.set_ylabel("y", color="white")
            ax_sensor.set_title("Detector", color="white")
            ax_sensor.tick_params(colors="white")

        ax.set_xlabel("x", color="white")
        ax.set_ylabel("y", color="white")
        ax.tick_params(colors="white")
        ax.set_title("Probability Density |ψ|²", color="white")

        def animate(frame: int) -> tuple:
            nonlocal sensor_accum
            im.set_array(trajectory[frame].T)
            elements = [im]
            if sensor_line is not None and sensor_line_plot is not None:
                current_intensity = trajectory[frame][sensor_line, :]
                if accumulate_sensor and sensor_accum is not None:
                    sensor_accum = sensor_accum + current_intensity
                    sensor_line_plot.set_xdata(sensor_accum)
                else:
                    sensor_line_plot.set_xdata(current_intensity)
                # Fixed scale - no auto-scaling
                elements.append(sensor_line_plot)
            return tuple(elements)

    anim = animation.FuncAnimation(
        fig, animate, frames=len(trajectory), interval=1000 / fps, blit=True
    )

    # Save as GIF (always works) or try MP4
    try:
        if output_path.endswith(".mp4"):
            writer = animation.FFMpegWriter(fps=fps, bitrate=3000)
            anim.save(output_path, writer=writer, dpi=100)
        else:
            anim.save(output_path, writer="pillow", fps=fps, dpi=100)
        status = "completed"
    except Exception:
        # Fallback to GIF
        gif_path = output_path.replace(".mp4", ".gif")
        anim.save(gif_path, writer="pillow", fps=min(fps, 20), dpi=80)
        output_path = gif_path
        status = "completed (as GIF, FFmpeg unavailable)"

    plt.close(fig)

    return [
        {
            "type": "text",
            "text": str(
                {
                    "output_path": output_path,
                    "status": status,
                    "frames": len(trajectory),
                    "fps": fps,
                }
            ),
        }
    ]


async def _tool_visualize_potential(args: dict[str, Any]) -> list[Any]:
    """Visualize potential energy landscape."""
    from pathlib import Path  # noqa: PLC0415

    import matplotlib as mpl  # noqa: PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415

    mpl.use("Agg")

    potential_id = args["potential_id"].replace("potential://", "")
    output_path = args.get("output_path", f"/tmp/potential-{potential_id}.png")

    if potential_id not in _potentials:
        return [{"type": "text", "text": f"Error: Potential {potential_id} not found"}]

    potential = _potentials[potential_id]

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("#0a0a1a")
    ax.set_facecolor("#0a0a1a")

    if potential.ndim == 1:
        ax.plot(potential, color="cyan", linewidth=2)
        ax.fill_between(np.arange(len(potential)), 0, potential, alpha=0.3, color="cyan")
        ax.set_xlabel("Position", color="white")
        ax.set_ylabel("V(x)", color="white")
    else:
        im = ax.imshow(potential.T, origin="lower", cmap="hot", aspect="equal")
        plt.colorbar(im, ax=ax, label="V(x,y)")
        ax.set_xlabel("x", color="white")
        ax.set_ylabel("y", color="white")

    ax.set_title("Potential Energy", color="white", fontsize=14)
    ax.tick_params(colors="white")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor="#0a0a1a", edgecolor="none")
    plt.close(fig)

    return [
        {
            "type": "text",
            "text": str(
                {
                    "output_path": output_path,
                    "status": "completed",
                    "shape": list(potential.shape),
                }
            ),
        }
    ]


async def run() -> None:
    """Run the Quantum MCP server."""
    from mcp.server.stdio import stdio_server  # noqa: PLC0415

    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


def main() -> None:
    """Entry point for the quantum-mcp command."""
    import asyncio  # noqa: PLC0415

    asyncio.run(run())


if __name__ == "__main__":
    main()
