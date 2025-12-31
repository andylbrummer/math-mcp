---
sidebar_position: 5
title: Square Lattice Bragg Scattering
---

# Square Lattice Bragg Scattering

<div style={{textAlign: 'center', margin: '2rem 0'}}>

![Square Lattice Bragg Scattering](/img/demos/bragg_square.webp)

</div>

## Seeing Atoms with Waves

When waves scatter from a periodic structure like a crystal, they create **diffraction patterns** that reveal the atomic arrangement. This is how we determine the structure of everything from simple salts to complex proteins.

## Square Lattice Structure

The square lattice is the simplest 2D crystal structure:

```
    ●───●───●───●───●
    │   │   │   │   │
    ●───●───●───●───●
    │   │   │   │   │
    ●───●───●───●───●
    │   │   │   │   │
    ●───●───●───●───●

    Lattice constant: a
    Basis vectors: a₁ = (a,0), a₂ = (0,a)
```

Real materials with square-like arrangements:
- Surface of cubic crystals (NaCl {100} face)
- 2D electron systems in semiconductors
- Optical lattices for ultracold atoms

## Bragg's Law

Constructive interference occurs when:

$$
2d \sin\theta = n\lambda
$$

Where:
- $d$ is the spacing between lattice planes
- $\theta$ is the scattering angle
- $\lambda$ is the wavelength
- $n$ is the diffraction order

For a square lattice, the allowed scattering vectors form a **reciprocal lattice** - also square!

## The Simulation

```
Grid: 512 × 512 points
Lattice spacing: 10.0 units
Potential depth: 25.0 units
Well radius: 2.0 units
Incoming momentum: kx = 5.0
```

### What to Watch For

1. **Initial approach**: Gaussian wavepacket traveling toward the crystal
2. **Scattering**: Part reflects, part transmits, part diffracts into Bragg peaks
3. **Diffraction pattern**: Spots appear at specific angles in k-space
4. **Fourfold symmetry**: Pattern reflects the square symmetry of the lattice

## Video

The animation above shows the wavepacket scattering in real-time.

## The Reciprocal Lattice

The diffraction pattern reveals the **reciprocal lattice** - the Fourier transform of the real-space lattice:

```
Real Space (atoms)         Reciprocal Space (diffraction)

    ● ● ● ●                      ★   ★   ★
    ● ● ● ●        FFT           ★   ★   ★
    ● ● ● ●       ───→           ★   ★   ★
    ● ● ● ●                      ★   ★   ★

Square lattice             Square pattern of spots
spacing = a                spacing = 2π/a
```

## Run It Yourself

```bash
claude -p "Demonstrate Bragg scattering from a square lattice and save to /tmp/bragg_square.gif" \
  --allowedTools "mcp__quantum-mcp__*"
```

## Physics Insights

### Why Specific Angles?

The crystal acts as a **perfect diffraction grating** in 2D. Only waves scattered at certain angles add constructively - all others cancel out by destructive interference.

### What Sets the Intensity?

The brightness of each diffraction spot depends on:
1. **Structure factor**: How atoms are arranged in the unit cell
2. **Form factor**: The scattering strength of each atom
3. **Temperature**: Thermal motion reduces peak intensity (Debye-Waller factor)

## Comparison with Other Lattices

<div style={{textAlign: 'center', margin: '2rem 0'}}>

See related demos for hexagonal and triangular lattice animations.

</div>

Different lattices produce different diffraction patterns:
- **Square**: Fourfold symmetric spots
- **Hexagonal**: Sixfold symmetric spots
- **Triangular**: Also sixfold, but different arrangement

## Related Demos

- [Hexagonal Lattice](./bragg-hexagonal) - Graphene-like structure
- [Triangular Lattice](./bragg-triangular) - Close-packed structure
- [Double-Slit](./double-slit) - Simpler interference
