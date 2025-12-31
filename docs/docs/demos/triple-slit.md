---
sidebar_position: 4
title: Triple-Slit Interference
---

# Triple-Slit Interference

<div style={{textAlign: 'center', margin: '2rem 0'}}>

![Triple-Slit Interference](/img/demos/triple_slit.webp)

</div>

## Beyond Double-Slit: Three-Way Interference

Adding a third slit creates a richer interference pattern with **subsidiary maxima** between the principal peaks. This demonstrates how multiple coherent sources combine to create complex wave patterns.

## The Physics

With three slits, the interference pattern becomes more structured:

- **Principal maxima**: Occur where all three waves constructively interfere
- **Subsidiary maxima**: Smaller peaks between principals where two waves reinforce
- **Minima**: More complex pattern of destructive interference

### Intensity Pattern

For three equally-spaced slits, the intensity distribution is:

$$
I(\theta) = I_0 \left( \frac{\sin(3\phi/2)}{\sin(\phi/2)} \right)^2
$$

Where $\phi = \frac{2\pi d \sin\theta}{\lambda}$ is the phase difference between adjacent slits.

## Pattern Characteristics

```
Triple-Slit vs Double-Slit Pattern:

Double-Slit:
████    ████    ████    ████    ████    ████
 (simple, evenly-spaced peaks)

Triple-Slit:
██████   ██   ██████   ██   ██████   ██   ██████
 (main)  (sub) (main)  (sub) (main)  (sub) (main)

 └─────────────────────────────────────────────┘
         More structure, sharper peaks
```

Key differences:
1. **Sharper principal maxima** - peaks are narrower
2. **Subsidiary maxima** - small peaks between main peaks
3. **Better resolution** - useful for spectroscopy

## The Simulation

```
Grid: 512 × 512 points
Slit width: 5.0 units
Slit separation: 25.0 units
Number of slits: 3
Initial momentum: kx = 3.0
```

## Video

The animation above shows the wavepacket evolution in real-time.

## Run It Yourself

```bash
claude -p "Simulate triple-slit interference and save to /tmp/triple_slit.gif" \
  --allowedTools "mcp__quantum-mcp__*"
```

## From Slits to Gratings

The triple-slit experiment is a step toward the **diffraction grating**:

| N Slits | Pattern Characteristics |
|---------|------------------------|
| 1 | Broad central maximum, weak side lobes |
| 2 | Regular interference fringes |
| 3 | Sharper peaks with 1 subsidiary maximum |
| N | Very sharp peaks with N-2 subsidiary maxima |
| ∞ | Delta-function peaks (perfect diffraction grating) |

### The Grating Equation

For N slits, principal maxima occur at:

$$
d \sin\theta = m\lambda \quad (m = 0, \pm 1, \pm 2, ...)
$$

And the peak width decreases as $1/N$, making diffraction gratings excellent for spectroscopy.

## Applications

- **Spectroscopy**: Gratings with thousands of lines separate wavelengths
- **X-ray crystallography**: Crystal lattices act as 3D gratings
- **Holography**: Interference patterns encode 3D information
- **Quantum computing**: Multi-path interference is fundamental

## Side-by-Side Comparison

<div style={{textAlign: 'center', margin: '2rem 0'}}>

![Triple-Slit Potential](/img/demos/triple_slit_potential.png)

</div>

## Related Demos

- [Single-Slit Diffraction](./single-slit) - Start with one slit
- [Double-Slit Interference](./double-slit) - The classic experiment
- [Bragg Scattering](./bragg-square) - Interference from crystal lattices
