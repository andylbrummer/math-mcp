# Double-Slit Interference Demo

## The Prompt

```
Simulate the famous double-slit experiment using the quantum MCP tools.
Create a barrier with two slits, send a quantum wavepacket through,
and render an animation of the interference pattern forming.
```

## MCP Tool Sequence

### Step 1: Create Double-Slit Barrier

```
mcp__quantum-mcp__create_custom_potential
  grid_size: [128, 128]
  function: "100 * ((np.abs(x - 64) < 3) & (np.abs(y - 54) > 5) & (np.abs(y - 74) > 5))"
```

Creates a barrier at x=64 with two slits centered at y=54 and y=74.

**Response:**
```json
{"potential_id": "potential://abc123", "shape": [128, 128]}
```

### Step 2: Create Incoming Wavepacket

```
mcp__quantum-mcp__create_gaussian_wavepacket
  grid_size: [128, 128]
  position: [25, 64]
  momentum: [0.5, 0]
  width: 10
```

**Response:**
```json
{"wavefunction_id": "wavefunction://xyz789", "shape": [128, 128], "norm": 1.0}
```

### Step 3: Visualize the Potential

```
mcp__quantum-mcp__visualize_potential
  potential_id: "potential://abc123"
  output_path: "/tmp/double_slit_barrier.png"
```

**Response:**
```json
{"output_path": "/tmp/double_slit_barrier.png", "status": "completed", "shape": [128, 128]}
```

### Step 4: Solve the 2D Schrödinger Equation

```
mcp__quantum-mcp__solve_schrodinger_2d
  potential: "potential://abc123"
  initial_state: "wavefunction://xyz789"
  time_steps: 300
  dt: 0.05
```

**Response:**
```json
{"simulation_id": "simulation://sim123", "status": "completed", "frames": 100, "grid_size": [128, 128]}
```

### Step 5: Render Animation

```
mcp__quantum-mcp__render_video
  simulation_id: "simulation://sim123"
  output_path: "/tmp/double_slit_interference.gif"
  fps: 20
```

**Response:**
```json
{"output_path": "/tmp/double_slit_interference.gif", "status": "completed", "frames": 100, "fps": 20}
```

## Expected Results

### The Barrier
A vertical barrier with two narrow slits. The wavepacket approaches from the left.

### The Interference Pattern
After passing through both slits:
- Waves from each slit spread and overlap
- Constructive interference creates bright bands
- Destructive interference creates dark bands
- Classic alternating fringe pattern emerges

## Variations

### Single Slit
```
Change the potential function to have only one slit:
"100 * ((np.abs(x - 64) < 3) & (np.abs(y - 64) > 5))"
```

### Triple Slit
```
Add a third slit:
"100 * ((np.abs(x - 64) < 3) & (np.abs(y - 44) > 4) & (np.abs(y - 64) > 4) & (np.abs(y - 84) > 4))"
```

### Vary Slit Separation
Adjust the y-positions (54, 74) to change the fringe spacing.

### Higher Momentum
Increase momentum from [0.5, 0] to [1.0, 0] for a narrower diffraction pattern.
