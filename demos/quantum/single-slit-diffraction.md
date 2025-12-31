# Single-Slit Diffraction Demo

## The Prompt

```
Simulate single-slit diffraction: create a barrier with one narrow slit,
send a Gaussian wavepacket through it, and show me the diffraction pattern.
Use a 256x256 grid for a 2D simulation.
```

## MCP Tool Sequence

### Step 1: Create the Barrier Potential

```tool
mcp__quantum-mcp__create_custom_potential
  grid_size: [256, 256]
  function: "1000 * ((np.abs(x - 128) < 3) & (np.abs(y - 128) > 8))"
```

This creates a barrier at x=128 with a single slit of width 16 units centered at y=128.

**Response:**
```json
{
  "potential_id": "potential://abc123",
  "shape": [256, 256]
}
```

### Step 2: Create the Incoming Wavepacket

```tool
mcp__quantum-mcp__create_gaussian_wavepacket
  grid_size: [256, 256]
  position: [50, 128]
  momentum: [2.0, 0]
  width: 15
```

A Gaussian wavepacket starting on the left, moving right toward the slit.

**Response:**
```json
{
  "wavefunction": [[...], [...], ...]
}
```

### Step 3: Solve the Schrödinger Equation

```tool
mcp__quantum-mcp__solve_schrodinger_2d
  potential: "potential://abc123"
  initial_state: <wavefunction from step 2>
  time_steps: 500
  dt: 0.02
```

**Response:**
```json
{
  "simulation_id": "simulation://xyz789",
  "status": "completed",
  "frames": 50
}
```

### Step 4: Visualize the Potential

```tool
mcp__quantum-mcp__visualize_potential
  potential_id: "potential://abc123"
  output_path: "/tmp/single_slit_potential.png"
```

### Step 5: Render the Animation

```tool
mcp__quantum-mcp__render_video
  simulation_id: "simulation://xyz789"
  output_path: "/tmp/single_slit_diffraction.mp4"
  fps: 30
```

## Expected Results

### The Barrier Potential
A vertical barrier with a single narrow opening. High potential (1000) blocks the wave; zero potential at the slit allows passage.

### The Diffraction Pattern
After passing through the slit, the wave spreads out in the characteristic single-slit pattern:
- Central bright maximum directly behind the slit
- Decreasing intensity side lobes
- Dark minima where destructive interference occurs

### The Physics
The narrow slit localizes the particle's position (Δy small), which by the uncertainty principle increases momentum spread (Δpy large), causing the wave to fan out.

## Variations

### Narrower Slit
```
Make the slit half as wide and show how the diffraction pattern changes.
```
Result: Wider angular spread (more diffraction)

### Higher Momentum
```
Double the incoming momentum and compare the patterns.
```
Result: Narrower diffraction pattern (shorter effective wavelength)
