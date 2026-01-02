---
sidebar_position: 3
title: Double-Slit Interference
---

# Double-Slit Interference

<div style={{textAlign: 'center', margin: '2rem 0'}}>

<video width="100%" autoPlay loop muted playsInline>
  <source src={require('@site/static/img/demos/double_slit.webm').default} type="video/webm" />
  <img src={require('@site/static/img/demos/double_slit.webp').default} alt="Double-Slit Interference" />
</video>

</div>

## The Most Famous Experiment in Physics

The double-slit experiment is often called "the most beautiful experiment in physics." It demonstrates the fundamental mystery of quantum mechanics: **wave-particle duality**.

> "I think I can safely say that nobody understands quantum mechanics."
> — Richard Feynman

When particles are sent through two slits one at a time, they still form an interference pattern - as if each particle interferes with itself.

## The Physics

Two waves from the slits interfere:
- **Constructive interference** (bright bands): Waves arrive in phase
- **Destructive interference** (dark bands): Waves arrive out of phase

### The Interference Condition

Bright fringes appear where: `d sin(θ) = nλ` (n = 0, ±1, ±2, ...)

Where:
- **d** is the slit separation
- **θ** is the angle from center
- **λ** is the wavelength
- **n** is the fringe order

The central bright fringe (n=0) is the brightest, with diminishing intensity for higher orders.

## What Makes It Quantum?

The truly strange thing: even when particles pass through one at a time, the interference pattern still builds up. This means:

1. Each particle somehow "knows about" both slits
2. The particle doesn't have a definite path until measured
3. The wavefunction passes through both slits simultaneously

## The Simulation

```
Grid: 512 × 512 points
Slit width: 6.0 units
Slit separation: 30.0 units
Wavepacket width: 25.0 units
Initial momentum: kx = 3.0
```

## Animation

The video above shows the wavepacket evolution in real-time, demonstrating the famous quantum interference pattern.

## Anatomy of the Pattern

```
                    ┌─────────────────────────────────────┐
                    │     Double-Slit Intensity Pattern    │
                    ├─────────────────────────────────────┤
      Intensity     │                 ██                   │
         ↑          │                ████                  │
         │          │               ██████                 │
         │          │        ██    ████████    ██         │
         │          │       ████  ██████████  ████        │
         │          │    ██████████████████████████████   │
         └──────────┴─────────────────────────────────────┘
                              ← Position →

                    Central    Secondary   Tertiary
                    Maximum     Maxima      Maxima
```

The pattern shows:
- **Central maximum**: Brightest fringe at center
- **Side maxima**: Regular spacing determined by d and λ
- **Envelope**: Overall intensity modulated by single-slit diffraction

## Run It Yourself

```bash
claude -p "Simulate double-slit interference: Create a barrier at x=85 with two slits \
  (separation=20, height=10), use an elliptical wavepacket with width=[15,50] \
  (vertical long axis to span slits), show potential overlay, add sensor line at x=220, \
  and save to /tmp/double_slit.gif" --allowedTools "mcp__quantum-mcp__*"
```

## The Fringe Spacing

The distance between adjacent bright fringes on a screen at distance L: `Δy = λL/d`

This is how the wavelength of light (or matter waves) can be measured.

## Historical Note

- **1801**: Thomas Young first demonstrated double-slit interference with light
- **1927**: Davisson and Germer showed electron diffraction
- **1961**: Jönsson demonstrated double-slit interference with electrons
- **2012**: Demonstrated with molecules containing 800+ atoms

## Related Demos

- [Single-Slit Diffraction](./single-slit) - Understand diffraction first
- [Triple-Slit Interference](./triple-slit) - Add more complexity
