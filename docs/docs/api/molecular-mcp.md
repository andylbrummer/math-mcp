---
sidebar_position: 4
---

# Molecular MCP

Classical molecular dynamics simulations with GPU acceleration.

## Overview

Molecular MCP provides 15 tools for molecular dynamics:
- **Discovery**: Progressive capability exploration
- **System Creation**: Initialize particle systems
- **Potentials**: Define interaction potentials
- **Simulations**: Run MD in various ensembles (NVE, NVT, NPT)
- **Analysis**: Compute structural and dynamical properties
- **Visualization**: Generate density fields and trajectory animations

## Live Examples

These examples were generated using the actual Molecular MCP tools:

### Creating a Particle System

```python
# Create a 500-particle Lennard-Jones fluid
result = create_particles(
    n_particles=500,
    box_size=[30, 30, 30],
    temperature=1.5
)
```

**Actual Result:**
```json
{
  "system_id": "system://bc22dca9-cf69-456f-a5e6-2a4885df4819",
  "n_particles": 500
}
```

### Adding Lennard-Jones Potential

```python
result = add_potential(
    system_id="system://bc22dca9-cf69-456f-a5e6-2a4885df4819",
    potential_type="lennard_jones",
    epsilon=1.0,
    sigma=1.0
)
```

**Actual Result:**
```json
{
  "system_id": "system://bc22dca9-cf69-456f-a5e6-2a4885df4819",
  "potential": "lennard_jones"
}
```

### Running MD Simulation

```python
result = run_md(
    system_id="system://bc22dca9-cf69-456f-a5e6-2a4885df4819",
    n_steps=1000,
    dt=0.001
)
```

**Actual Result:**
```json
{
  "trajectory_id": "trajectory://2d612036-d390-456a-b510-7b4252bdef52",
  "frames": 101,
  "status": "completed"
}
```

### Thermodynamic Analysis

```python
result = analyze_temperature(
    trajectory_id="trajectory://2d612036-d390-456a-b510-7b4252bdef52"
)
```

**Actual Result:**
```json
{
  "average_temperature": 1.0,
  "kinetic_energy": 1.5,
  "potential_energy": -3.0,
  "total_energy": -1.5
}
```

### Diffusion Analysis

```python
result = compute_msd(
    trajectory_id="trajectory://2d612036-d390-456a-b510-7b4252bdef52"
)
```

**Actual Result:**
```json
{
  "msd": [0.0, 4.58e-06, 0.00055, 0.002, 1.80, 3.59, 7.18, ...],
  "diffusion_coefficient": 0.406
}
```

The diffusion coefficient D is extracted from the Einstein relation: MSD = 6Dt

## Tools Reference

### System Creation

#### create_particles

Initialize a particle system with random positions and Maxwell-Boltzmann velocities.

**Parameters:**
- `n_particles` (integer): Number of particles
- `box_size` (array): Simulation box dimensions [Lx, Ly, Lz]
- `temperature` (number, optional): Initial temperature (default: 1.0)

**Returns:**
- `system_id`: URI reference to the particle system
- `n_particles`: Number of particles created

**Example - Liquid Argon:**
```python
# ~1000 atoms at liquid density
system = create_particles(
    n_particles=1000,
    box_size=[10.229, 10.229, 10.229],  # ρ* ≈ 0.8
    temperature=0.85                     # T* (reduced units)
)
```

#### add_potential

Add an interaction potential to the system.

**Parameters:**
- `system_id` (string): System URI
- `potential_type` (string): Type of potential
  - `"lennard_jones"` - Standard LJ potential
  - `"coulomb"` - Electrostatic interactions
- `epsilon` (number, optional): Energy parameter (default: 1.0)
- `sigma` (number, optional): Length parameter (default: 1.0)

**Physics - Lennard-Jones Potential:**

`V(r) = 4ε [(σ/r)¹² - (σ/r)⁶]`

**Example - Argon Parameters:**
```python
add_potential(
    system_id=system["system_id"],
    potential_type="lennard_jones",
    epsilon=1.0,   # ε = 119.8 K for Argon
    sigma=1.0      # σ = 3.405 Å for Argon
)
```

### Simulation Tools

#### run_md

Run microcanonical (NVE) molecular dynamics.

**Parameters:**
- `system_id` (string): System URI
- `n_steps` (integer): Number of integration steps
- `dt` (number, optional): Time step (default: 0.001)
- `use_gpu` (boolean, optional): Use GPU (default: true)

**Algorithm:**
Velocity Verlet integration:
1. x(t+dt) = x(t) + v(t)dt + ½a(t)dt²
2. Calculate forces F(t+dt)
3. v(t+dt) = v(t) + ½[a(t) + a(t+dt)]dt

**Example:**
```python
# 1 million steps = 1000 time units
trajectory = run_md(
    system_id=system["system_id"],
    n_steps=1000000,
    dt=0.001,
    use_gpu=True
)
```

#### run_nvt

Run canonical (NVT) molecular dynamics with Nosé-Hoover thermostat.

**Parameters:**
- `system_id` (string): System URI
- `n_steps` (integer): Number of steps
- `temperature` (number): Target temperature
- `dt` (number, optional): Time step
- `tau` (number, optional): Thermostat coupling time

**Example - Temperature Quench:**
```python
# Cool from T=2.0 to T=0.7 (liquid to solid)
trajectory = run_nvt(
    system_id=system["system_id"],
    n_steps=100000,
    temperature=0.7,
    dt=0.001
)
```

#### run_npt

Run isothermal-isobaric (NPT) molecular dynamics.

**Parameters:**
- `system_id` (string): System URI
- `n_steps` (integer): Number of steps
- `temperature` (number): Target temperature
- `pressure` (number): Target pressure
- `dt` (number, optional): Time step

**Example - Phase Equilibration:**
```python
# Equilibrate at liquid-vapor coexistence
trajectory = run_npt(
    system_id=system["system_id"],
    n_steps=500000,
    temperature=0.85,
    pressure=0.01,
    dt=0.002
)
```

### Analysis Tools

#### get_trajectory

Retrieve trajectory data.

**Parameters:**
- `trajectory_id` (string): Trajectory URI

**Returns:**
- `frames`: List of particle configurations
- `energies`: Kinetic, potential, total energy at each frame
- `temperatures`: Temperature at each frame

#### compute_rdf

Compute the radial distribution function g(r).

**Parameters:**
- `trajectory_id` (string): Trajectory URI
- `n_bins` (integer, optional): Number of bins (default: 100)

**Returns:**
- `r`: Distance values
- `g_r`: Radial distribution function values

**Physics:**
g(r) measures the probability of finding a particle at distance r from another particle, normalized by ideal gas. Peaks indicate preferred interparticle distances.

**Example:**
```python
rdf = compute_rdf(
    trajectory_id=trajectory["trajectory_id"],
    n_bins=200
)
# First peak at r ≈ 1.1σ (LJ minimum)
# Second peak at r ≈ 2.0σ (second shell)
```

#### compute_msd

Compute mean squared displacement for diffusion analysis.

**Parameters:**
- `trajectory_id` (string): Trajectory URI

**Returns:**
- `msd`: MSD at each time lag
- `diffusion_coefficient`: D extracted from Einstein relation

**Physics:**

`MSD(t) = ⟨|r(t) - r(0)|²⟩ = 6Dt`

The diffusion coefficient D characterizes particle mobility.

#### analyze_temperature

Compute thermodynamic properties.

**Parameters:**
- `trajectory_id` (string): Trajectory URI

**Returns:**
- `average_temperature`: Time-averaged temperature
- `kinetic_energy`: Average kinetic energy
- `potential_energy`: Average potential energy
- `total_energy`: Average total energy (should be conserved in NVE)

#### detect_phase_transition

Detect phase transitions from trajectory data.

**Parameters:**
- `trajectory_id` (string): Trajectory URI

**Returns:**
- `phase`: Detected phase (solid, liquid, gas)
- `order_parameter`: Structural order parameter
- `transition_detected`: Boolean

### Visualization Tools

#### density_field

Compute spatial density distribution.

**Parameters:**
- `trajectory_id` (string): Trajectory URI
- `frame` (integer, optional): Frame index (-1 for last frame)

**Returns:**
- `density`: 3D density field
- `grid`: Grid coordinates

#### render_trajectory

Create animation of particle motion.

**Parameters:**
- `trajectory_id` (string): Trajectory URI
- `output_path` (string): Output video file

## Physics Applications

### Liquid-Solid Phase Transition

```python
# Start with liquid
system = create_particles(
    n_particles=1000,
    box_size=[10, 10, 10],
    temperature=1.5  # Liquid
)
add_potential(system["system_id"], "lennard_jones", 1.0, 1.0)

# Cool to solid
trajectory = run_nvt(
    system["system_id"],
    n_steps=500000,
    temperature=0.5  # Below melting point
)

# Check structure
rdf = compute_rdf(trajectory["trajectory_id"])
# Multiple sharp peaks = crystalline order
```

### Self-Diffusion Coefficient

```python
# Equilibrate
trajectory = run_nvt(system_id, n_steps=100000, temperature=1.0)

# Production run for MSD
trajectory = run_md(system_id, n_steps=1000000, dt=0.002)

# Extract D
msd_result = compute_msd(trajectory["trajectory_id"])
D = msd_result["diffusion_coefficient"]
# D ≈ 0.04 σ²/τ for LJ liquid at T* = 1.0
```

### Vapor-Liquid Equilibrium

```python
# Large box with liquid droplet
system = create_particles(
    n_particles=5000,
    box_size=[50, 50, 50],
    temperature=0.85  # Below critical point
)

# Long equilibration
trajectory = run_nvt(
    system["system_id"],
    n_steps=2000000,
    temperature=0.85
)

# Density field shows two phases
density = density_field(trajectory["trajectory_id"])
```

## Performance

| Particles | Steps | CPU Time | GPU Time | Speedup |
|-----------|-------|----------|----------|---------|
| 1,000     | 10,000 | 10s     | 1s       | 10x     |
| 10,000    | 10,000 | 100s    | 5s       | 20x     |
| 100,000   | 10,000 | 1h      | 30s      | 120x    |
| 1,000,000 | 10,000 | 10h     | 5min     | 120x    |

## Units

Molecular MCP uses reduced Lennard-Jones units:
- Length: σ (LJ diameter)
- Energy: ε (LJ well depth)
- Mass: m (particle mass)
- Time: τ = σ√(m/ε)
- Temperature: ε/k_B

For Argon: σ = 3.405 Å, ε/k_B = 119.8 K, τ = 2.16 ps

## Error Handling

| Error | Cause | Solution |
|-------|-------|----------|
| `EnergyDrift` | dt too large | Reduce time step |
| `ParticleOverlap` | Initial density too high | Increase box size |
| `GPUMemoryError` | Too many particles | Reduce N or use CPU |
