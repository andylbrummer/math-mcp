---
sidebar_position: 3
---

# Quantum MCP

Wave mechanics and Schrödinger equation simulations with GPU acceleration.

## Overview

Quantum MCP provides 12 tools for quantum mechanical simulations:
- **Discovery**: Progressive capability exploration
- **Potentials**: Create quantum potential energy landscapes
- **Wavepackets**: Initialize quantum states
- **Simulations**: Time-evolve quantum systems via split-step Fourier method
- **Analysis**: Extract physical observables
- **Visualization**: Generate plots and animations

## Live Examples

These examples were generated using the actual Quantum MCP tools:

### Creating a Lattice Potential

```python
# Create a square lattice potential for quantum tunneling studies
result = create_lattice_potential(
    lattice_type="square",
    grid_size=[256],
    depth=15,
    spacing=20
)
```

**Actual Result:**
```json
{
  "potential_id": "potential://953d8a0c-f857-4f73-a330-1e81488d6c6c",
  "shape": [256]
}
```

### Creating a Gaussian Wave Packet

```python
# Create a localized wave packet with initial momentum
result = create_gaussian_wavepacket(
    grid_size=[256],
    position=[50],      # Initial position
    momentum=[0.5],     # Initial momentum (rightward)
    width=8             # Spatial width
)
```

The wavefunction is returned as a complex array representing the quantum state amplitude at each grid point.

## Tools Reference

### Discovery

#### info

Progressive discovery of Quantum MCP capabilities.

**Parameters:**
- `topic` (string, optional): Category or tool name

**Example:**
```python
info(topic="overview")
# Returns categories: potentials (2), wavepackets (2),
#                     simulations (3), analysis (2), visualization (2)
```

### Potential Tools

#### create_lattice_potential

Create crystalline lattice potentials for periodic structures.

**Parameters:**
- `lattice_type` (string): Type of lattice
  - `"square"` - Square/cubic lattice
  - `"hexagonal"` - Hexagonal lattice
  - `"triangular"` - Triangular lattice
- `grid_size` (array): Grid dimensions, e.g., `[256]` for 1D, `[128, 128]` for 2D
- `depth` (number): Potential well depth (determines barrier height)
- `spacing` (number): Distance between lattice sites

**Returns:**
- `potential_id`: URI reference to the created potential
- `shape`: Grid dimensions

#### create_custom_potential

Create custom potentials from mathematical functions or arrays.

**Parameters:**
- `grid_size` (array): Grid dimensions
- `function` (string, optional): Mathematical expression, e.g., `"10*exp(-(x-128)**2/100)"`
- `array_uri` (string, optional): Reference to a Math MCP array

**Example - Gaussian Barrier:**
```python
# Create a Gaussian potential barrier for tunneling
create_custom_potential(
    grid_size=[256],
    function="20*exp(-(x-128)**2/200)"  # Barrier at center
)
```

**Example - Double Well:**
```python
# Create a double-well potential for quantum oscillation
create_custom_potential(
    grid_size=[256],
    function="0.1*(x-64)**2*(x-192)**2/10000"
)
```

### Wavepacket Tools

#### create_gaussian_wavepacket

Create a localized Gaussian wave packet - the most common initial state for scattering simulations.

**Parameters:**
- `grid_size` (array): Grid dimensions
- `position` (array): Center position of the wave packet
- `momentum` (array): Initial momentum (determines direction and speed)
- `width` (number, optional): Spatial width (default: 5)

**Physics:**
The Gaussian wave packet is:

`ψ(x) = (1/2πσ²)^(1/4) exp(-(x-x₀)²/4σ²) exp(ik₀x)`

Where σ is the width, x₀ is the position, and k₀ is the momentum.

**Example:**
```python
# Wave packet moving right toward a barrier
create_gaussian_wavepacket(
    grid_size=[256],
    position=[64],     # Start on the left
    momentum=[2.0],    # Move rightward
    width=10.0
)
```

#### create_plane_wave

Create a plane wave state with definite momentum.

**Parameters:**
- `grid_size` (array): Grid dimensions
- `momentum` (array): Wave momentum k

**Physics:**
A plane wave has the form ψ(x) = e^(ikx), representing a state with definite momentum ℏk.

### Simulation Tools

#### solve_schrodinger

Solve the 1D time-dependent Schrödinger equation using the split-step Fourier method.

**Parameters:**
- `potential` (string): Potential URI from create_*_potential
- `initial_state` (array): Initial wavefunction
- `time_steps` (integer): Number of time steps
- `dt` (number): Time step size
- `use_gpu` (boolean, optional): Use GPU acceleration (default: true)

**Algorithm:**
The split-step Fourier method alternates between position and momentum space:
1. Apply half kinetic evolution in k-space: e^(-iℏk²dt/4m)
2. Apply potential evolution in x-space: e^(-iV(x)dt/ℏ)
3. Apply half kinetic evolution in k-space: e^(-iℏk²dt/4m)

**Returns:**
- `simulation_id`: URI to access simulation results
- `task_id`: For async progress monitoring (long simulations)

#### solve_schrodinger_2d

Solve the 2D time-dependent Schrödinger equation.

**Parameters:**
Same as `solve_schrodinger`, but with 2D grids.

**Performance:**
| Grid Size | Time Steps | CPU Time | GPU Time |
|-----------|------------|----------|----------|
| 256×256   | 1000       | ~30 min  | ~30 sec  |
| 512×512   | 1000       | ~2 hours | ~2 min   |

#### get_task_status

Monitor async simulation progress.

**Parameters:**
- `task_id` (string): Task ID from simulation

**Returns:**
- `state`: "running", "completed", or "failed"
- `progress`: Completion percentage
- `simulation_id`: Available when completed

### Analysis Tools

#### get_simulation_result

Retrieve completed simulation data.

**Parameters:**
- `simulation_id` (string): Simulation URI

**Returns:**
- `frames`: List of wavefunctions at stored time points
- `times`: Corresponding time values
- `metadata`: Grid info, parameters, etc.

#### analyze_wavefunction

Compute physical observables from a wavefunction.

**Parameters:**
- `wavefunction` (array): Complex wavefunction array
- `dx` (number, optional): Grid spacing (default: 1)

**Returns:**
- `norm`: Total probability (should be 1.0)
- `position_expectation`: Expected position ⟨x⟩
- `momentum_expectation`: Expected momentum ⟨p⟩
- `position_uncertainty`: Δx
- `momentum_uncertainty`: Δp
- `uncertainty_product`: Δx·Δp (must be ≥ ℏ/2)
- `kinetic_energy`: ⟨T⟩
- `potential_energy`: ⟨V⟩
- `total_energy`: ⟨H⟩

### Visualization Tools

#### visualize_potential

Create a static plot of the potential energy landscape.

**Parameters:**
- `potential_id` (string): Potential URI
- `output_path` (string, optional): Save location

#### render_video

Animate the probability density evolution over time.

**Parameters:**
- `simulation_id` (string): Simulation URI
- `output_path` (string, optional): Output video path
- `fps` (integer, optional): Frames per second (default: 30)

## Physics Applications

### Quantum Tunneling

Study wave packet transmission through potential barriers:

```python
# 1. Create barrier
barrier = create_custom_potential(
    grid_size=[512],
    function="30*exp(-(x-256)**2/100)"  # Gaussian barrier
)

# 2. Create incident wave packet
psi = create_gaussian_wavepacket(
    grid_size=[512],
    position=[100],
    momentum=[3.0],    # Energy > barrier for partial transmission
    width=15
)

# 3. Simulate
sim = solve_schrodinger(
    potential=barrier["potential_id"],
    initial_state=psi,
    time_steps=2000,
    dt=0.05
)

# 4. Analyze transmission coefficient
# Compare probability on either side of barrier
```

### Double-Well Oscillation

Observe quantum oscillation between potential wells:

```python
# Symmetric double well
double_well = create_custom_potential(
    grid_size=[256],
    function="0.01*(x-64)**2*(x-192)**2 - 50"
)

# Localized initial state in left well
psi = create_gaussian_wavepacket(
    grid_size=[256],
    position=[64],
    momentum=[0],  # No initial momentum
    width=10
)

# Long-time evolution shows oscillation between wells
```

### Band Structure in Periodic Potentials

Study energy bands in crystalline potentials:

```python
# Periodic potential (Kronig-Penney model)
lattice = create_lattice_potential(
    lattice_type="square",
    grid_size=[256],
    depth=20,
    spacing=32
)

# Plane wave analysis reveals band gaps
```

## Cross-MCP Integration

### Math MCP → Quantum MCP

Create custom potentials using Math MCP arrays:

```python
# Step 1: Create potential array with Math MCP
potential_array = math_mcp.create_array(
    shape=[256],
    fill_type="function",
    function="10*(1/(1+exp(-0.5*(x-100))) - 1/(1+exp(-0.5*(x-156))))"
)
# Returns: array://abc123...

# Step 2: Use in Quantum MCP simulation
simulation = quantum_mcp.create_custom_potential(
    array_uri="array://abc123...",
    grid_size=[256]
)
```

## Performance Tips

1. **Grid Size**: Use powers of 2 (256, 512, 1024) for optimal FFT performance
2. **Time Step**: Smaller dt = more accurate but slower. Start with dt=0.1, reduce if unstable
3. **GPU**: Essential for 2D simulations; provides 60x speedup on 256×256 grids
4. **Memory**: 2D simulations can use significant memory; 512×512×1000 frames ≈ 2GB

## Error Handling

| Error | Cause | Solution |
|-------|-------|----------|
| `UnstableSimulation` | Time step too large | Reduce `dt` parameter |
| `NormNotConserved` | Numerical instability | Reduce dt or increase grid resolution |
| `GPUMemoryError` | Grid too large | Reduce dimensions or use CPU |
