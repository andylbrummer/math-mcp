# Bragg Scattering from Crystal Lattices

## The Prompt

```
Demonstrate Bragg scattering from crystal lattices using the quantum MCP tools.
Create a square lattice potential, fire a wavepacket at it, and show the
diffraction pattern. Then compare with hexagonal and triangular lattices.
```

## MCP Tool Sequence

### Step 1: Create Square Lattice Potential

```
mcp__quantum-mcp__create_lattice_potential
  lattice_type: "square"
  grid_size: [128, 128]
  depth: 15
  spacing: 10
```

**Response:**
```json
{"potential_id": "potential://square-lattice", "shape": [128, 128]}
```

### Step 2: Create Incoming Wavepacket

```
mcp__quantum-mcp__create_gaussian_wavepacket
  grid_size: [128, 128]
  position: [20, 64]
  momentum: [0.8, 0]
  width: 12
```

**Response:**
```json
{"wavefunction_id": "wavefunction://incoming", "shape": [128, 128], "norm": 1.0}
```

### Step 3: Visualize the Lattice

```
mcp__quantum-mcp__visualize_potential
  potential_id: "potential://square-lattice"
  output_path: "/tmp/square_lattice.png"
```

### Step 4: Run Scattering Simulation

```
mcp__quantum-mcp__solve_schrodinger_2d
  potential: "potential://square-lattice"
  initial_state: "wavefunction://incoming"
  time_steps: 400
  dt: 0.03
```

**Response:**
```json
{"simulation_id": "simulation://bragg-square", "status": "completed", "frames": 100}
```

### Step 5: Render Animation

```
mcp__quantum-mcp__render_video
  simulation_id: "simulation://bragg-square"
  output_path: "/tmp/bragg_square.gif"
  fps: 20
```

## Hexagonal Lattice (Graphene-like)

```
mcp__quantum-mcp__create_lattice_potential
  lattice_type: "hexagonal"
  grid_size: [128, 128]
  depth: 15
  spacing: 10
```

The hexagonal lattice has honeycomb structure like graphene:
- Two atoms per unit cell
- Sixfold rotational symmetry
- Different diffraction pattern than square

## Triangular Lattice (Close-packed)

```
mcp__quantum-mcp__create_lattice_potential
  lattice_type: "triangular"
  grid_size: [128, 128]
  depth: 15
  spacing: 10
```

The triangular lattice is the densest 2D packing:
- One atom per unit cell
- Each atom has 6 nearest neighbors
- Also shows sixfold symmetric diffraction

## Expected Diffraction Patterns

| Lattice | Symmetry | Pattern |
|---------|----------|---------|
| Square | 4-fold | Square grid of spots |
| Hexagonal | 6-fold | Hexagonal arrangement, some spots missing |
| Triangular | 6-fold | Complete hexagonal pattern |

## The Physics

### Bragg's Law
Constructive interference occurs when:
$$2d\sin\theta = n\lambda$$

### Reciprocal Lattice
The diffraction pattern reveals the **reciprocal lattice** - the Fourier transform of the real-space crystal structure.

## Comparison Prompt

```
Create all three lattice types (square, hexagonal, triangular) with the same
spacing and depth. Run identical wavepacket scattering simulations on each.
Compare the resulting diffraction patterns to show how crystal structure
determines the diffraction signature.
```
