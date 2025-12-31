---
sidebar_position: 8
title: Galaxy Collision
---

# Galaxy Collision

<div style={{textAlign: 'center', margin: '2rem 0'}}>

![Galaxy Collision](/img/demos/galaxy_collision.webp)

</div>

## Cosmic Ballet

When galaxies collide, they create some of the most spectacular events in the universe. Gravitational tidal forces rip stars from their orbits, creating sweeping **tidal tails** and **bridges** that span hundreds of thousands of light-years.

This simulation uses **N-body gravitational dynamics** to model the merger of two disk galaxies, similar to the famous Antennae Galaxies or the predicted future collision of the Milky Way and Andromeda.

## The Physics

### N-Body Gravitation

Each star particle experiences gravitational attraction from all others:

$$
\mathbf{a}_i = \sum_{j \neq i} \frac{G m_j (\mathbf{r}_j - \mathbf{r}_i)}{(|\mathbf{r}_j - \mathbf{r}_i|^2 + \epsilon^2)^{3/2}}
$$

The softening parameter $\epsilon$ prevents:
- Numerical instabilities from close encounters
- Unrealistically high velocities
- Effectively models the extended mass of real star systems

### Initial Conditions

Each galaxy is created with:
1. **Central bulge**: Pressure-supported (random velocities)
2. **Disk component**: Rotation-supported (circular orbits)
3. **Massive core particles**: Simulate dark matter halo concentration

```
Galaxy Parameters:
- Stars per galaxy: 1500
- Disk radius: ~12 kpc
- Bulge fraction: 20%
- Initial separation: 70 kpc
- Impact parameter: 20 kpc
```

## Simulation Stages

### 1. Approach
<div style={{background: '#111', padding: '1rem', borderRadius: '8px', margin: '1rem 0'}}>
Two spiral galaxies approach each other on a hyperbolic orbit. Tidal forces begin to distort the outer regions.
</div>

### 2. First Passage
<div style={{background: '#111', padding: '1rem', borderRadius: '8px', margin: '1rem 0'}}>
The galaxies pass through each other (stars rarely collide directly). Strong tidal forces pull out long streamers of stars.
</div>

### 3. Tidal Tail Formation
<div style={{background: '#111', padding: '1rem', borderRadius: '8px', margin: '1rem 0'}}>
Material stripped from the outer disks forms spectacular tidal tails extending far from the merger site.
</div>

### 4. Second Passage & Merger
<div style={{background: '#111', padding: '1rem', borderRadius: '8px', margin: '1rem 0'}}>
Gravitational friction slows the galaxies. They fall back together and eventually merge into a single, larger galaxy.
</div>

## Animation

The animation above shows the gravitational N-body simulation in real-time.

## Run It Yourself

```bash
claude -p "Simulate a galaxy collision with 300 particles and gravitational interactions, run for 2000 steps, and save to /tmp/galaxy_collision.gif" \
  --allowedTools "mcp__molecular-mcp__*"
```

## Real Galaxy Collisions

### The Antennae Galaxies (NGC 4038/4039)

<div style={{background: '#1a1a2e', padding: '1rem', borderRadius: '8px', margin: '1rem 0'}}>

The Antennae are the closest example of a galaxy merger:
- **Distance**: 45 million light-years
- **Status**: Mid-merger
- **Features**: Spectacular tidal tails, intense star formation
- **Future**: Will become a single elliptical galaxy

Our simulation captures the essential physics of this interaction.

</div>

### Milky Way - Andromeda Collision

In about **4.5 billion years**, our Milky Way will collide with the Andromeda Galaxy (M31):
- Current separation: 2.5 million light-years
- Approach velocity: ~110 km/s
- Result: A new elliptical galaxy ("Milkomeda")

## The Science

### Why Stars Don't Collide

Even though galaxies "collide," individual stars almost never hit each other:
- Stars are tiny compared to the space between them
- If the Sun were a grain of sand, the nearest star would be 4 miles away
- Galaxies are mostly empty space

### Tidal Forces

The gravitational gradient across a galaxy creates differential forces:

$$
\Delta F \sim \frac{GM \Delta r}{r^3}
$$

This stretches the galaxy along the line connecting the two centers and compresses it perpendicular to this line.

### Dynamical Friction

Galaxies lose orbital energy through:
1. **Gravitational wake**: Each galaxy creates a wake of particles behind it
2. **Momentum transfer**: Energy goes from bulk motion to random stellar motion
3. **Result**: Galaxies spiral inward and eventually merge

## Collision Scenarios

| Type | Impact Parameter | Features |
|------|-----------------|----------|
| **Head-on** | ~0 | Violent disruption, rapid merger |
| **Moderate** | 0.3 × R | Tidal tails, bridges, eventual merger |
| **Glancing** | > R | Long tidal streamers, may not merge |

## Technical Notes

### Integration Method

**Velocity Verlet** algorithm:
1. Half-step velocities: $v_{n+1/2} = v_n + \frac{1}{2}a_n \Delta t$
2. Full-step positions: $x_{n+1} = x_n + v_{n+1/2} \Delta t$
3. Compute new accelerations: $a_{n+1}$
4. Complete velocity step: $v_{n+1} = v_{n+1/2} + \frac{1}{2}a_{n+1} \Delta t$

### Computational Complexity

Direct N-body is O(N²) per timestep. For larger simulations, use:
- **Barnes-Hut tree code**: O(N log N)
- **Fast Multipole Method**: O(N)
- **Particle-Mesh**: O(N log N) with FFT

## Related Demos

- [Bragg Scattering](./bragg-square) - Quantum N-body analog
- [Double-Slit](./double-slit) - Wave interference

## Further Reading

- Toomre & Toomre (1972): Classic paper on tidal tails
- Barnes & Hernquist (1992): Merger dynamics and remnant properties
- Hopkins et al. (2006): Star formation in galaxy mergers
