# Math MCP Visual Demonstrations

Dramatic physics simulations showcasing the capabilities of the Math MCP ecosystem.
These demos are designed to be run through **Claude with MCP tools enabled**.

## How to Run Demos

### Option 1: Interactive Chat with Claude Code

Start Claude Code in this directory (it will automatically load the MCP servers):

```bash
cd /path/to/math-mcp
claude
```

Then give prompts like:
- "Simulate double-slit interference and save an animation to /tmp/demo.gif"
- "Create a Bragg scattering demo with a hexagonal lattice"
- "Simulate two galaxies colliding with gravitational N-body dynamics"

### Option 2: One-shot with `claude -p`

```bash
claude -p "Simulate quantum double-slit interference: create a barrier with two slits at x=64, create a wavepacket at x=25 moving right, evolve for 300 steps, and render to /tmp/double_slit.gif"
```

## Demo Categories

### Quantum Mechanics (`demos/quantum/`)

| Demo | Prompt | Description |
|------|--------|-------------|
| [Single-Slit](quantum/single-slit-diffraction.md) | "Simulate single-slit diffraction" | Wave diffraction through one opening |
| [Double-Slit](quantum/double-slit-interference.md) | "Simulate double-slit interference" | Classic quantum interference |
| [Triple-Slit](quantum/triple-slit-interference.md) | "Simulate triple-slit interference" | Complex multi-slit patterns |
| [Bragg Scattering](quantum/bragg-scattering-lattices.md) | "Scatter a wavepacket off a crystal lattice" | Diffraction from periodic structures |

### Molecular Dynamics (`demos/molecular/`)

| Demo | Prompt | Description |
|------|--------|-------------|
| [Galaxy Collision](molecular/galaxy-collision.md) | "Simulate two galaxies colliding" | N-body gravitational dynamics |

## MCP Tools Used

### Quantum MCP (`mcp__quantum-mcp__*`)

| Tool | Purpose |
|------|---------|
| `create_custom_potential` | Define barrier/slit geometry |
| `create_lattice_potential` | Create crystal lattice (square/hex/triangular) |
| `create_gaussian_wavepacket` | Initialize quantum state |
| `solve_schrodinger_2d` | Evolve the wavefunction |
| `visualize_potential` | Save potential as image |
| `render_video` | Create animation GIF/MP4 |

### Molecular MCP (`mcp__molecular-mcp__*`)

| Tool | Purpose |
|------|---------|
| `create_particles` | Initialize N-body system |
| `add_potential` | Add gravitational/LJ forces |
| `run_md` | Velocity Verlet integration |
| `render_trajectory` | Create animation |

## Example Session

```
User: Simulate double-slit interference

Claude: I'll create a double-slit interference simulation.

[Calls mcp__quantum-mcp__create_custom_potential with barrier function]
Created potential with two slits at y=54 and y=74

[Calls mcp__quantum-mcp__create_gaussian_wavepacket]
Created wavepacket at position [25, 64] moving right

[Calls mcp__quantum-mcp__solve_schrodinger_2d]
Simulation complete: 100 frames

[Calls mcp__quantum-mcp__render_video]
Saved animation to /tmp/double_slit.gif

The animation shows the classic double-slit interference pattern emerging
as the quantum wavepacket passes through both slits and the probability
waves interfere constructively and destructively.
```

## Output Files

Demos typically save to `/tmp/`:
- `*.gif` - Animated visualizations
- `*.png` - Static images of potentials

## Physics Notes

### Quantum Simulations
- **Method**: Split-step Fourier (spectral method)
- **Equation**: Time-dependent Schrödinger equation
- Wavefunction IDs avoid passing large arrays between tools

### Molecular Dynamics
- **Method**: Velocity Verlet integration
- **Forces**: Gravitational with softening, Lennard-Jones
- **Complexity**: O(N²) direct summation

## Directory Structure

```
demos/
├── README.md                           # This file
├── quantum/
│   ├── single-slit-diffraction.md     # Prompt + tool sequence
│   ├── double-slit-interference.md
│   ├── triple-slit-interference.md
│   └── bragg-scattering-lattices.md
└── molecular/
    └── galaxy-collision.md
```

## Contributing

To add a new demo:

1. Create a markdown file documenting:
   - The natural language prompt
   - The MCP tool sequence
   - Expected results
   - Variations and physics background
2. Test it interactively with Claude
3. Add to this README
