---
sidebar_position: 7
title: Triangular Lattice Bragg Scattering
---

# Triangular Lattice Bragg Scattering

<div style={{textAlign: 'center', margin: '2rem 0'}}>

![Triangular Lattice Bragg Scattering](/img/demos/bragg_triangular.webp)

</div>

## The Densest Packing

The triangular lattice represents **close packing** in two dimensions - the most efficient way to arrange circles (or spheres) on a plane. Each atom has **six nearest neighbors** at equal distances.

## Structure

```
         ●     ●     ●     ●
        / \   / \   / \   / \
       ●───●───●───●───●───●
        \ / \ / \ / \ / \ /
         ●───●───●───●───●
        / \ / \ / \ / \ / \
       ●───●───●───●───●───●
        \ / \ / \ / \ / \ /
         ●     ●     ●     ●

    Each atom touches 6 neighbors
    Maximum 2D packing fraction: π/(2√3) ≈ 0.9069
```

### Key Properties

| Property | Value |
|----------|-------|
| Nearest neighbors | 6 |
| Coordination number | 6 |
| Packing fraction | 90.69% |
| Rotational symmetry | 6-fold (C₆) |
| Basis vectors | a₁ = (a, 0), a₂ = (a/2, a√3/2) |

## Where It's Found

- **Noble gas monolayers** on surfaces
- **Colloidal crystals** (particles in suspension)
- **Vortex lattices** in superconductors
- **Bubble rafts** (soap bubbles on water)
- **Base layer of FCC/HCP crystals**

## The Simulation

```
Grid: 512 × 512 points
Lattice spacing: 10.0 units
Potential depth: 25.0 units
Well radius: 2.0 units
One atom per unit cell
```

## Video

The animation above shows the wavepacket scattering in real-time.

## Diffraction Pattern

The triangular lattice produces a **hexagonal diffraction pattern**:

```
          ★
         / \
       ★     ★
       │     │
   ★───●─────●───★
       │     │
       ★     ★
         \ /
          ★

    Sixfold symmetric spots
    (same symmetry as hexagonal lattice)
```

### Comparing Triangular vs Hexagonal

Both produce sixfold symmetric patterns, but:

| Feature | Triangular | Hexagonal |
|---------|------------|-----------|
| Atoms per cell | 1 | 2 |
| Missing spots | None | Some (structure factor) |
| Intensity ratios | All equal | Varies with (h,k) |

## The Physics

### Miller Indices in 2D

Diffraction spots are labeled by Miller indices (h, k):

```
                    (0,2)
                     ★
                    /
        (−1,1) ★   /
                \ /
    (−1,0) ★────●────★ (1,0)
                /\
               /  ★ (1,−1)
              /
             ★
           (0,−2)
```

### Spot Intensity

For a single-atom basis (triangular lattice), all spots have equal **structure factor** F = 1. The intensity is:

$$
I_{hk} = |F_{hk}|^2 \cdot \text{(geometric factors)}
$$

## Run It Yourself

```bash
claude -p "Demonstrate Bragg scattering from a triangular lattice and save to /tmp/bragg_triangular.gif" \
  --allowedTools "mcp__quantum-mcp__*"
```

## Physical Significance

### 2D Melting

The triangular lattice is key to understanding **2D phase transitions**:

1. **Solid**: Long-range positional and orientational order
2. **Hexatic phase**: Lost positional order, retained orientational order
3. **Liquid**: No long-range order

This is the famous **KTHNY theory** (Kosterlitz-Thouless-Halperin-Nelson-Young).

### Defects

Perfect triangular lattices are rare. Defects create:
- **Dislocations**: Missing half-lines of atoms
- **Grain boundaries**: Interfaces between domains
- **Vacancies**: Missing atoms

These appear as diffuse scattering in the diffraction pattern.

## Comparison

See related demos for square and hexagonal lattice animations.

## Related Demos

- [Square Lattice](./bragg-square) - Lower symmetry structure
- [Hexagonal Lattice](./bragg-hexagonal) - Graphene structure
- [Single-Slit](./single-slit) - Basic diffraction
