---
sidebar_position: 2
title: Single-Slit Diffraction
---

# Single-Slit Diffraction

<div style={{textAlign: 'center', margin: '2rem 0'}}>

<video width="100%" autoPlay loop muted playsInline>
  <source src="/img/demos/single_slit.webm" type="video/webm" />
  <img src="/img/demos/single_slit.webp" alt="Single-Slit Diffraction" />
</video>

</div>

## The Physics

When a wave passes through a narrow opening, it **diffracts** - spreading out in a characteristic pattern. This is a fundamental property of waves, whether they're water waves, sound waves, or quantum probability waves.

The single-slit diffraction pattern shows:
- A **central maximum** directly behind the slit
- **Secondary maxima** of decreasing intensity on either side
- **Dark minima** where destructive interference occurs

### The Diffraction Equation

The angular positions of the dark fringes follow: `a sin(θ) = nλ` (n = 1, 2, 3, ...)

Where:
- **a** is the slit width
- **θ** is the angle from the central axis
- **λ** is the wavelength
- **n** is the order of the minimum

## What You'll See

<div className="demo-stages">

### 1. Initial State
A Gaussian wavepacket approaches the barrier from the left. The wavepacket has a well-defined momentum (direction of travel) but is localized in space.

### 2. Interaction with Slit
When the wavepacket reaches the barrier, most of it is reflected. Only the portion passing through the slit continues.

### 3. Diffraction Pattern Emerges
After passing through, the wave spreads out. The probability density forms the characteristic single-slit pattern with a bright central band.

### 4. Far-Field Pattern
At the detector (right side of simulation), we see the intensity pattern - brightest at center, falling off with characteristic oscillations.

</div>

## The Simulation

```
Grid: 512 × 512 points
Slit width: 8.0 units
Barrier height: 1000 (effectively infinite)
Wavepacket momentum: kx = 3.0
```

## Animation

The video above shows the wavepacket evolution in real-time, demonstrating how the probability density evolves as the quantum particle passes through the slit.

## Run It Yourself

```bash
claude -p "Simulate single-slit diffraction and save to /tmp/single_slit.gif" \
  --allowedTools "mcp__quantum-mcp__*"
```

## Key Observations

1. **Wave Nature**: The spreading after the slit proves the wave nature of the quantum particle
2. **Uncertainty Principle**: Confining the particle's position (narrow slit) spreads its momentum (diffraction)
3. **Intensity Distribution**: Follows the sinc² function: `I(θ) ∝ (sin(β)/β)²` where `β = πa sin(θ)/λ`

## Related Demos

- [Double-Slit Interference](./double-slit) - Add a second slit for interference
- [Triple-Slit Interference](./triple-slit) - Even more complex patterns
