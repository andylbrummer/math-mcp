---
sidebar_position: 6
title: Hexagonal Lattice Bragg Scattering
---

# Hexagonal Lattice Bragg Scattering

<div style={{textAlign: 'center', margin: '2rem 0'}}>

<video width="100%" autoPlay loop muted playsInline>
  <source src={require('@site/static/img/demos/bragg_hexagonal.webm').default} type="video/webm" />
  <img src={require('@site/static/img/demos/bragg_hexagonal.webp').default} alt="Hexagonal Lattice Bragg Scattering" />
</video>

</div>

## The Structure of Graphene

The hexagonal (honeycomb) lattice is the structure of **graphene** - a single layer of carbon atoms that won the 2010 Nobel Prize in Physics. It's also found in hexagonal boron nitride and the surfaces of many materials.

## Honeycomb Structure

The hexagonal lattice has **two atoms per unit cell**, creating the distinctive honeycomb pattern:

```
        ●───●       ●───●       ●───●
       /     \     /     \     /     \
      ●       ●───●       ●───●       ●
       \     /     \     /     \     /
        ●───●       ●───●       ●───●
       /     \     /     \     /     \
      ●       ●───●       ●───●       ●
       \     /     \     /     \     /
        ●───●       ●───●       ●───●

    Two-atom basis creates the honeycomb
```

### Key Features

- **Two atoms per unit cell** (A and B sublattices)
- **Three nearest neighbors** per atom
- **Sixfold rotational symmetry**
- **Dirac cones** in the band structure (for graphene)

## Why Graphene is Special

Graphene's electrons behave as massless Dirac fermions, leading to:
- Extremely high electron mobility
- Unusual quantum Hall effect
- Potential for quantum computing
- Revolutionary material applications

Understanding its diffraction pattern helps characterize graphene samples.

## The Simulation

```
Grid: 512 × 512 points
Lattice constant: 10.0 units
Potential depth: 25.0 units
Two atoms per unit cell
Incoming momentum: kx = 5.0
```

## Animation

The video above shows the wavepacket scattering from the honeycomb lattice, revealing the sixfold symmetric Bragg pattern.

## The Diffraction Pattern

The hexagonal lattice produces a **sixfold symmetric** diffraction pattern:

```
                 ★
                / \
              ★     ★
               \   /
            ★───●───★
               /   \
              ★     ★
                \ /
                 ★

    Six primary diffraction spots
    (plus higher-order spots)
```

The two-atom basis creates additional features:
- Some spots are **systematically absent** (structure factor = 0)
- Remaining spots show specific intensity ratios
- Pattern uniquely identifies the honeycomb structure

## Reciprocal Lattice

The reciprocal lattice of a hexagonal lattice is **also hexagonal**, but rotated 30°:

| Real Space | Reciprocal Space |
|------------|-----------------|
| Hexagonal, angle = 60° | Hexagonal, angle = 60° |
| Spacing = a | Spacing = 4π/(√3 a) |
| Aligned with x | Rotated 30° |

## Run It Yourself

```bash
claude -p "Demonstrate Bragg scattering from a hexagonal lattice and save to /tmp/bragg_hexagonal.gif" \
  --allowedTools "mcp__quantum-mcp__*"
```

## X-ray Crystallography

This same principle is used to determine:
- **Crystal structures** of new materials
- **Protein structures** (X-ray crystallography)
- **DNA structure** (Franklin and Wilkins' famous Photo 51)

The pattern of spots tells us:
1. **Positions**: Crystal symmetry
2. **Intensities**: Atomic positions within the unit cell
3. **Widths**: Crystal quality and domain size

## Real-World Applications

### Graphene Characterization

Low-Energy Electron Diffraction (LEED) on graphene shows the hexagonal pattern, confirming:
- Single-layer vs multi-layer samples
- Orientation relative to substrate
- Defect density and crystal quality

### Material Identification

Different materials produce unique "fingerprint" patterns:
- Hexagonal: Graphene, hBN, MoS₂
- Square: Cubic crystals
- Complex: Proteins, minerals

## Related Demos

- [Square Lattice](./bragg-square) - Simpler structure
- [Triangular Lattice](./bragg-triangular) - Close-packed atoms
- [Triple-Slit](./triple-slit) - Multi-beam interference
