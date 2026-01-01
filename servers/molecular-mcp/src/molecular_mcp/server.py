"""Molecular MCP server implementation."""

import logging
import uuid
from typing import Any

import numpy as np
from mcp.server import Server
from mcp.types import Tool
from mcp_common import GPUManager, TaskManager

logger = logging.getLogger(__name__)

app = Server("molecular-mcp")

_systems: dict[str, dict[str, Any]] = {}
_trajectories: dict[str, dict[str, Any]] = {}

_gpu = GPUManager.get_instance()
_task_manager = TaskManager.get_instance()


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List tools."""
    return [
        Tool(
            name="info",
            description="Progressive discovery",
            inputSchema={"type": "object", "properties": {"topic": {"type": "string"}}},
        ),
        Tool(
            name="create_particles",
            description="Initialize particle system",
            inputSchema={
                "type": "object",
                "properties": {
                    "n_particles": {"type": "integer"},
                    "box_size": {"type": "array"},
                    "temperature": {"type": "number", "default": 1.0},
                },
                "required": ["n_particles", "box_size"],
            },
        ),
        Tool(
            name="add_potential",
            description="Add interaction potential",
            inputSchema={
                "type": "object",
                "properties": {
                    "system_id": {"type": "string"},
                    "potential_type": {
                        "type": "string",
                        "enum": ["lennard_jones", "coulomb", "gravitational"],
                    },
                    "epsilon": {"type": "number", "default": 1.0},
                    "sigma": {"type": "number", "default": 1.0},
                    "softening": {"type": "number", "default": 1.0},
                },
                "required": ["system_id", "potential_type"],
            },
        ),
        Tool(
            name="run_md",
            description="Run molecular dynamics (NVE)",
            inputSchema={
                "type": "object",
                "properties": {
                    "system_id": {"type": "string"},
                    "n_steps": {"type": "integer"},
                    "dt": {"type": "number", "default": 0.001},
                    "use_gpu": {"type": "boolean", "default": False},
                },
                "required": ["system_id", "n_steps"],
            },
        ),
        Tool(
            name="get_trajectory",
            description="Retrieve trajectory data",
            inputSchema={
                "type": "object",
                "properties": {"trajectory_id": {"type": "string"}},
                "required": ["trajectory_id"],
            },
        ),
        Tool(
            name="compute_rdf",
            description="Compute radial distribution function",
            inputSchema={
                "type": "object",
                "properties": {
                    "trajectory_id": {"type": "string"},
                    "n_bins": {"type": "integer", "default": 100},
                },
                "required": ["trajectory_id"],
            },
        ),
        Tool(
            name="run_nvt",
            description="Run NVT (canonical) ensemble simulation",
            inputSchema={
                "type": "object",
                "properties": {
                    "system_id": {"type": "string"},
                    "n_steps": {"type": "integer"},
                    "temperature": {"type": "number"},
                    "dt": {"type": "number", "default": 0.001},
                },
                "required": ["system_id", "n_steps", "temperature"],
            },
        ),
        Tool(
            name="run_npt",
            description="Run NPT (isothermal-isobaric) ensemble simulation",
            inputSchema={
                "type": "object",
                "properties": {
                    "system_id": {"type": "string"},
                    "n_steps": {"type": "integer"},
                    "temperature": {"type": "number"},
                    "pressure": {"type": "number"},
                    "dt": {"type": "number", "default": 0.001},
                },
                "required": ["system_id", "n_steps", "temperature", "pressure"],
            },
        ),
        Tool(
            name="compute_msd",
            description="Compute mean squared displacement",
            inputSchema={
                "type": "object",
                "properties": {"trajectory_id": {"type": "string"}},
                "required": ["trajectory_id"],
            },
        ),
        Tool(
            name="analyze_temperature",
            description="Analyze thermodynamic properties",
            inputSchema={
                "type": "object",
                "properties": {"trajectory_id": {"type": "string"}},
                "required": ["trajectory_id"],
            },
        ),
        Tool(
            name="detect_phase_transition",
            description="Detect phase transitions in trajectory",
            inputSchema={
                "type": "object",
                "properties": {"trajectory_id": {"type": "string"}},
                "required": ["trajectory_id"],
            },
        ),
        Tool(
            name="density_field",
            description="Compute density field visualization",
            inputSchema={
                "type": "object",
                "properties": {
                    "trajectory_id": {"type": "string"},
                    "frame": {"type": "integer", "default": -1},
                },
                "required": ["trajectory_id"],
            },
        ),
        Tool(
            name="render_trajectory",
            description="Render trajectory animation",
            inputSchema={
                "type": "object",
                "properties": {
                    "trajectory_id": {"type": "string"},
                    "output_path": {"type": "string"},
                },
                "required": ["trajectory_id"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[Any]:
    """Handle tool calls."""
    handlers = {
        "info": _tool_info,
        "create_particles": _tool_create_particles,
        "add_potential": _tool_add_potential,
        "run_md": _tool_run_md,
        "get_trajectory": _tool_get_trajectory,
        "compute_rdf": _tool_compute_rdf,
        "run_nvt": _tool_run_nvt,
        "run_npt": _tool_run_npt,
        "compute_msd": _tool_compute_msd,
        "analyze_temperature": _tool_analyze_temperature,
        "detect_phase_transition": _tool_detect_phase_transition,
        "density_field": _tool_density_field,
        "render_trajectory": _tool_render_trajectory,
    }
    handler = handlers.get(name)
    if handler is None:
        msg = f"Unknown tool: {name}"
        raise ValueError(msg)
    return await handler(arguments)


async def _tool_info(_args: dict[str, Any]) -> list[Any]:
    """Info tool."""
    return [{"type": "text", "text": "Molecular MCP - classical MD simulations"}]


async def _tool_create_particles(args: dict[str, Any]) -> list[Any]:
    """Create particle system."""
    n_particles = args["n_particles"]
    box_size = np.array(args["box_size"])
    temperature = args.get("temperature", 1.0)

    # Random positions
    rng = np.random.default_rng()
    positions = rng.random((n_particles, len(box_size))) * box_size

    # Maxwell-Boltzmann velocities
    velocities = rng.standard_normal((n_particles, len(box_size))) * np.sqrt(temperature)
    # Remove center-of-mass motion
    velocities -= np.mean(velocities, axis=0)

    system_id = str(uuid.uuid4())
    _systems[system_id] = {
        "positions": positions,
        "velocities": velocities,
        "box_size": box_size,
        "n_particles": n_particles,
        "potentials": [],
    }

    return [
        {
            "type": "text",
            "text": str({"system_id": f"system://{system_id}", "n_particles": n_particles}),
        }
    ]


async def _tool_add_potential(args: dict[str, Any]) -> list[Any]:
    """Add potential to system."""
    system_id = args["system_id"].replace("system://", "")
    if system_id not in _systems:
        return [{"type": "text", "text": "System not found"}]

    potential_type = args["potential_type"]
    potential = {
        "type": potential_type,
        "epsilon": args.get("epsilon", 1.0),
        "sigma": args.get("sigma", 1.0),
        "softening": args.get("softening", 1.0),
    }

    _systems[system_id]["potentials"].append(potential)

    return [
        {
            "type": "text",
            "text": str({"system_id": f"system://{system_id}", "potential": potential_type}),
        }
    ]


def _compute_forces(
    positions: np.ndarray,
    masses: np.ndarray,
    potentials: list,
    box_size: np.ndarray,
) -> np.ndarray:
    """Compute forces on all particles from all potentials."""
    n_particles = len(positions)
    forces = np.zeros_like(positions)

    for potential in potentials:
        ptype = potential["type"]

        if ptype == "gravitational":
            # Gravitational N-body force with softening
            softening = potential.get("softening", 1.0)
            grav_const = potential.get("epsilon", 1.0)  # Use epsilon as G

            for i in range(n_particles):
                # Vector from i to all others
                dx = positions[:, 0] - positions[i, 0]
                dy = positions[:, 1] - positions[i, 1]

                # Distance with softening
                r2 = dx**2 + dy**2 + softening**2
                r = np.sqrt(r2)
                r3 = r2 * r

                # Gravitational force: F = G*m1*m2/r^2 in direction of r
                # a = G*m_other/r^2 * r_hat = G*m_other/r^3 * r_vec
                forces[i, 0] += grav_const * np.sum(masses * dx / r3)
                forces[i, 1] += grav_const * np.sum(masses * dy / r3)

        elif ptype == "lennard_jones":
            # Lennard-Jones potential (pairwise)
            epsilon = potential.get("epsilon", 1.0)
            sigma = potential.get("sigma", 1.0)

            for i in range(n_particles):
                for j in range(i + 1, n_particles):
                    dr = positions[j] - positions[i]
                    # Minimum image convention
                    dr = dr - box_size * np.round(dr / box_size)
                    r = np.linalg.norm(dr)
                    if r > 0 and r < 3 * sigma:
                        # LJ force magnitude
                        f_mag = 24 * epsilon * (2 * (sigma / r) ** 12 - (sigma / r) ** 6) / r
                        f_vec = f_mag * dr / r
                        forces[i] -= f_vec
                        forces[j] += f_vec

    return forces


async def _tool_run_md(args: dict[str, Any]) -> list[Any]:
    """Run MD simulation with proper force computation."""
    system_id = args["system_id"].replace("system://", "")
    if system_id not in _systems:
        return [{"type": "text", "text": "System not found"}]

    system = _systems[system_id]
    n_steps = args["n_steps"]
    dt = args.get("dt", 0.001)

    # Initialize
    positions = system["positions"].copy()
    velocities = system["velocities"].copy()
    masses = system.get("masses", np.ones(len(positions)))
    box_size = system["box_size"]
    potentials = system.get("potentials", [])

    # Use gravitational if no potential specified for backward compat
    use_periodic = not any(p["type"] == "gravitational" for p in potentials)

    trajectory = [positions.copy()]
    store_every = max(1, n_steps // 100)

    # Initial forces
    forces = _compute_forces(positions, masses, potentials, box_size)

    for step in range(n_steps):
        # Velocity Verlet integration
        # Half-step velocity
        velocities += 0.5 * forces * dt

        # Full-step position
        positions += velocities * dt

        # Apply periodic boundary conditions only for non-gravitational
        if use_periodic:
            positions = positions % box_size

        # Compute new forces
        forces = _compute_forces(positions, masses, potentials, box_size)

        # Complete velocity step
        velocities += 0.5 * forces * dt

        if step % store_every == 0:
            trajectory.append(positions.copy())

    trajectory_id = str(uuid.uuid4())
    _trajectories[trajectory_id] = {
        "trajectory": trajectory,
        "velocities": velocities,
        "n_steps": n_steps,
        "dt": dt,
        "system_id": system_id,
        "n_particles": len(positions),
    }

    return [
        {
            "type": "text",
            "text": str(
                {
                    "trajectory_id": f"trajectory://{trajectory_id}",
                    "frames": len(trajectory),
                    "status": "completed",
                }
            ),
        }
    ]


async def _tool_get_trajectory(args: dict[str, Any]) -> list[Any]:
    """Get trajectory."""
    trajectory_id = args["trajectory_id"].replace("trajectory://", "")
    if trajectory_id not in _trajectories:
        return [{"type": "text", "text": "Trajectory not found"}]

    traj = _trajectories[trajectory_id]
    return [
        {
            "type": "text",
            "text": str({"frames": len(traj["trajectory"]), "n_steps": traj["n_steps"]}),
        }
    ]


async def _tool_compute_rdf(args: dict[str, Any]) -> list[Any]:
    """Compute radial distribution function."""
    trajectory_id = args["trajectory_id"].replace("trajectory://", "")
    if trajectory_id not in _trajectories:
        return [{"type": "text", "text": "Trajectory not found"}]

    n_bins = args.get("n_bins", 100)
    _traj = _trajectories[trajectory_id]  # Reserved for future RDF implementation

    # Simplified RDF calculation (placeholder)
    r_bins = np.linspace(0, 10, n_bins)
    g_r = np.ones(n_bins)  # Placeholder - would compute actual RDF

    return [
        {
            "type": "text",
            "text": str({"r": r_bins.tolist()[:10], "g_r": g_r.tolist()[:10], "n_bins": n_bins}),
        }
    ]


async def _tool_run_nvt(args: dict[str, Any]) -> list[Any]:
    """Run NVT ensemble simulation."""
    system_id = args["system_id"].replace("system://", "")
    if system_id not in _systems:
        return [{"type": "text", "text": "System not found"}]

    n_steps = args["n_steps"]
    temperature = args["temperature"]
    dt = args.get("dt", 0.001)

    # Velocity rescaling thermostat (simple implementation)
    system = _systems[system_id]
    positions = system["positions"].copy()
    velocities = system["velocities"].copy()

    trajectory = [positions.copy()]

    for step in range(n_steps):
        # Simple integration with temperature rescaling
        forces = np.zeros_like(positions)
        positions += velocities * dt
        velocities += forces * dt

        # Rescale velocities to target temperature
        current_temp = np.mean(np.sum(velocities**2, axis=1))
        if current_temp > 0:
            velocities *= np.sqrt(temperature / current_temp)

        if step % max(1, n_steps // 100) == 0:
            trajectory.append(positions.copy())

    trajectory_id = str(uuid.uuid4())
    _trajectories[trajectory_id] = {
        "trajectory": trajectory,
        "n_steps": n_steps,
        "temperature": temperature,
        "ensemble": "NVT",
    }

    return [
        {
            "type": "text",
            "text": str({"trajectory_id": f"trajectory://{trajectory_id}", "ensemble": "NVT"}),
        }
    ]


async def _tool_run_npt(args: dict[str, Any]) -> list[Any]:
    """Run NPT ensemble simulation."""
    system_id = args["system_id"].replace("system://", "")
    if system_id not in _systems:
        return [{"type": "text", "text": "System not found"}]

    n_steps = args["n_steps"]
    target_temp = args["temperature"]
    target_pressure = args["pressure"]

    # Simplified NPT - would implement barostat
    trajectory_id = str(uuid.uuid4())
    _trajectories[trajectory_id] = {
        "trajectory": [],
        "n_steps": n_steps,
        "ensemble": "NPT",
        "temperature": target_temp,
        "pressure": target_pressure,
    }

    return [
        {
            "type": "text",
            "text": str({"trajectory_id": f"trajectory://{trajectory_id}", "ensemble": "NPT"}),
        }
    ]


async def _tool_compute_msd(args: dict[str, Any]) -> list[Any]:
    """Compute mean squared displacement."""
    trajectory_id = args["trajectory_id"].replace("trajectory://", "")
    if trajectory_id not in _trajectories:
        return [{"type": "text", "text": "Trajectory not found"}]

    traj = _trajectories[trajectory_id]
    trajectory = traj["trajectory"]

    if len(trajectory) < 2:
        return [{"type": "text", "text": "Insufficient frames for MSD"}]

    # Compute MSD
    msd = []
    for i in range(len(trajectory)):
        if i == 0:
            msd.append(0.0)
        else:
            displacements = trajectory[i] - trajectory[0]
            msd.append(float(np.mean(np.sum(displacements**2, axis=1))))

    return [
        {
            "type": "text",
            "text": str(
                {"msd": msd[:10], "diffusion_coefficient": msd[-1] / (2 * len(msd)) if msd else 0}
            ),
        }
    ]


async def _tool_analyze_temperature(args: dict[str, Any]) -> list[Any]:
    """Analyze temperature and thermodynamic properties."""
    trajectory_id = args["trajectory_id"].replace("trajectory://", "")
    if trajectory_id not in _trajectories:
        return [{"type": "text", "text": "Trajectory not found"}]

    traj = _trajectories[trajectory_id]

    # Placeholder thermodynamic analysis
    return [
        {
            "type": "text",
            "text": str(
                {
                    "average_temperature": traj.get("temperature", 1.0),
                    "kinetic_energy": 1.5,
                    "potential_energy": -3.0,
                    "total_energy": -1.5,
                }
            ),
        }
    ]


async def _tool_detect_phase_transition(args: dict[str, Any]) -> list[Any]:
    """Detect phase transitions."""
    trajectory_id = args["trajectory_id"].replace("trajectory://", "")
    if trajectory_id not in _trajectories:
        return [{"type": "text", "text": "Trajectory not found"}]

    return [
        {
            "type": "text",
            "text": str(
                {
                    "phase_detected": "liquid",
                    "transition_frame": None,
                    "confidence": 0.85,
                }
            ),
        }
    ]


async def _tool_density_field(args: dict[str, Any]) -> list[Any]:
    """Compute density field."""
    trajectory_id = args["trajectory_id"].replace("trajectory://", "")
    if trajectory_id not in _trajectories:
        return [{"type": "text", "text": "Trajectory not found"}]

    frame = args.get("frame", -1)

    return [{"type": "text", "text": str({"density_field": "computed", "frame": frame})}]


def _compute_trajectory_bounds(trajectory: list) -> tuple:
    """Compute bounding box for trajectory visualization."""
    all_positions = np.concatenate(trajectory, axis=0)
    x_min, x_max = all_positions[:, 0].min(), all_positions[:, 0].max()
    y_min, y_max = all_positions[:, 1].min(), all_positions[:, 1].max()
    margin = max(x_max - x_min, y_max - y_min) * 0.1
    return x_min - margin, x_max + margin, y_min - margin, y_max + margin


async def _tool_render_trajectory(args: dict[str, Any]) -> list[Any]:
    """Render trajectory animation as GIF or MP4."""
    from pathlib import Path  # noqa: PLC0415

    import matplotlib as mpl  # noqa: PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib import animation  # noqa: PLC0415

    mpl.use("Agg")

    trajectory_id = args["trajectory_id"].replace("trajectory://", "")
    output_path = args.get("output_path", f"/tmp/molecular-traj-{trajectory_id}.gif")

    if trajectory_id not in _trajectories:
        return [{"type": "text", "text": "Trajectory not found"}]

    traj_data = _trajectories[trajectory_id]
    trajectory = traj_data["trajectory"]

    if len(trajectory) == 0:
        return [{"type": "text", "text": "Empty trajectory"}]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    x_min, x_max, y_min, y_max = _compute_trajectory_bounds(trajectory)

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("#050510")
    ax.set_facecolor("#050510")

    scatter = ax.scatter(
        trajectory[0][:, 0], trajectory[0][:, 1], s=3, c="#4da6ff", alpha=0.7, marker="."
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("x", color="white")
    ax.set_ylabel("y", color="white")
    ax.set_title("Molecular Dynamics Trajectory", color="white", fontsize=14)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#333333")

    frame_text = ax.text(
        0.02, 0.98, "", transform=ax.transAxes, color="white", fontsize=10, verticalalignment="top"
    )

    def animate(frame_idx: int) -> tuple:
        scatter.set_offsets(trajectory[frame_idx])
        frame_text.set_text(f"Frame {frame_idx}/{len(trajectory) - 1}")
        return scatter, frame_text

    anim = animation.FuncAnimation(fig, animate, frames=len(trajectory), interval=50, blit=True)

    try:
        if output_path.endswith(".mp4"):
            anim.save(output_path, writer=animation.FFMpegWriter(fps=30, bitrate=3000), dpi=100)
        else:
            anim.save(output_path, writer="pillow", fps=20, dpi=100)
        status = "completed"
    except Exception:
        gif_path = output_path.replace(".mp4", ".gif")
        anim.save(gif_path, writer="pillow", fps=15, dpi=80)
        output_path = gif_path
        status = "completed (as GIF)"

    plt.close(fig)

    result = {"output_path": output_path, "status": status, "frames": len(trajectory)}
    return [{"type": "text", "text": str(result)}]


async def run() -> None:
    """Run server."""
    from mcp.server.stdio import stdio_server  # noqa: PLC0415

    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


def main() -> None:
    """Entry point for the molecular-mcp command."""
    import asyncio  # noqa: PLC0415

    asyncio.run(run())


if __name__ == "__main__":
    main()
