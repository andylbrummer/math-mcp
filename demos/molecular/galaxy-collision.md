# Galaxy Collision Demo

## The Prompt

```
Simulate two galaxies colliding using the molecular MCP gravitational N-body tools.
Create two clusters of particles representing disk galaxies, set them on a collision
course, and render the merger animation showing tidal tail formation.
```

## MCP Tool Sequence

### Step 1: Create First Galaxy

```
mcp__molecular-mcp__create_particles
  n_particles: 500
  box_size: [200, 200]
  temperature: 0.3
```

**Response:**
```json
{"system_id": "system://galaxy1", "n_particles": 500}
```

Note: For a proper disk galaxy structure, you'd want to customize the initial
positions and velocities to create rotation. The basic `create_particles` gives
random positions with Maxwell-Boltzmann velocities.

### Step 2: Add Gravitational Potential

```
mcp__molecular-mcp__add_potential
  system_id: "system://galaxy1"
  potential_type: "gravitational"
  epsilon: 1.0
  softening: 2.0
```

**Response:**
```json
{"system_id": "system://galaxy1", "potential": "gravitational"}
```

### Step 3: Run N-body Simulation

```
mcp__molecular-mcp__run_md
  system_id: "system://galaxy1"
  n_steps: 2000
  dt: 0.01
```

**Response:**
```json
{"trajectory_id": "trajectory://traj123", "frames": 100, "status": "completed"}
```

### Step 4: Render Animation

```
mcp__molecular-mcp__render_trajectory
  trajectory_id: "trajectory://traj123"
  output_path: "/tmp/galaxy_collision.gif"
```

**Response:**
```json
{"output_path": "/tmp/galaxy_collision.gif", "status": "completed", "frames": 100}
```

## Two-Galaxy Collision Setup

For a proper collision, you need to:

1. Create two separate particle systems
2. Position them apart with opposite velocities
3. Combine into one system
4. Add gravitational potential
5. Run the simulation

### Example Prompt for Full Collision

```
Create a dramatic galaxy collision:
1. Generate 400 particles for galaxy A centered at (-50, 0) moving right
2. Generate 400 particles for galaxy B centered at (50, 0) moving left
3. Combine them into one gravitational system
4. Simulate for 3000 steps to see tidal tails form
5. Render the animation
```

## The Physics

### Gravitational Force with Softening

The force between particles uses softening to prevent singularities:

$$F = \frac{G m_1 m_2}{(r^2 + \epsilon^2)^{3/2}} \hat{r}$$

Where:
- $G$ is the gravitational constant (epsilon parameter)
- $\epsilon$ is the softening length
- This prevents infinite forces at r=0

### Tidal Forces

As galaxies approach:
- Differential gravity stretches the outer regions
- Stars on the near side accelerate toward the other galaxy
- Stars on the far side lag behind
- This creates the characteristic tidal tails

## Analysis Tools

### Mean Squared Displacement

```
mcp__molecular-mcp__compute_msd
  trajectory_id: "trajectory://traj123"
```

Shows how particles spread during the collision.

### Temperature Analysis

```
mcp__molecular-mcp__analyze_temperature
  trajectory_id: "trajectory://traj123"
```

Monitors kinetic energy (velocity dispersion) evolution.

## Expected Results

1. **Initial Approach**: Galaxies move toward each other
2. **First Passage**: Strong tidal distortion, some mass exchange
3. **Tidal Tails**: Long streamers of stars pulled out
4. **Return & Merger**: Cores spiral together
5. **Relaxation**: System settles into new equilibrium
