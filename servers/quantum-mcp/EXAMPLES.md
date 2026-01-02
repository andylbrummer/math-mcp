# scicomp-quantum-mcp Examples

Explore quantum mechanics with practical wave function simulations and visualizations.

## 🎬 Interactive Visual Demos

For dramatic real-time visualizations, try the **interactive demos**:

📺 **Watch animations:**
- [Single-Slit Diffraction](https://andylbrummer.github.io/math-mcp/docs/demos/single-slit) - Classic wave diffraction
- [Double-Slit Interference](https://andylbrummer.github.io/math-mcp/docs/demos/double-slit) - Quantum interference
- [Triple-Slit Interference](https://andylbrummer.github.io/math-mcp/docs/demos/triple-slit) - Complex multi-slit patterns
- [Bragg Scattering](https://andylbrummer.github.io/math-mcp/docs/demos/bragg-square) - Crystal lattice diffraction

🚀 **Run with Claude:**
```bash
claude -p "Simulate double-slit interference and save to /tmp/demo.gif" \
  --allowedTools "mcp__quantum-mcp__*"
```

---

## 🌊 Example 1: Particle in a Box (Classic Quantum System)

**Setup:** Electron confined to 1D box with infinite potential walls

### Create the Potential
```python
from quantum_mcp import create_lattice_potential

# Create box potential: infinite walls at x=0 and x=100
# Zero potential inside the box
box_potential = create_custom_potential(
    function="0 if 0 < x < 100 else 1000",
    grid_size=[256]
)
```

### Create Initial Wave Packet
```python
from quantum_mcp import create_gaussian_wavepacket

# Start particle in center of box
wavepacket = create_gaussian_wavepacket(
    grid_size=[256],
    position=[128],           # Center of box
    momentum=[10.0],          # Moving to the right
    width=10                  # Narrow initial state
)
```

### Simulate Time Evolution
```python
from quantum_mcp import solve_schrodinger

# Evolve for 100 time steps
trajectory = solve_schrodinger(
    potential=box_potential,
    initial_state=wavepacket,
    time_steps=100,
    dt=0.1                    # Time step
)

# Animate the probability density
from quantum_mcp import render_video
video = render_video(
    simulation_id=trajectory,
    output_path="particle_in_box.mp4"
)

print("✓ Wave packet bounces between walls with interference patterns!")
```

**Visual Output:** Watch the wavefunction oscillate and create standing wave patterns 🎬

---

## 🔗 Example 2: Double Slit Interference

**Classic quantum experiment:** Observe interference from two slits

### Create Slit Potential (2D)
```python
# Two slits separated by 40 units, width 20 each
# Wall in y-direction at x=128
slit_potential = create_custom_potential(
    function="0 if (40 < y < 60) or (80 < y < 100) else 100",
    grid_size=[256, 256]
)
```

### Plane Wave Incident
```python
from quantum_mcp import create_plane_wave

# Particle coming from left
incident = create_plane_wave(
    grid_size=[256, 256],
    momentum=[5.0, 0.0]      # Moving in x-direction
)
```

### Observe Interference Pattern
```python
# Run 2D simulation
result = solve_schrodinger_2d(
    potential=slit_potential,
    initial_state=incident,
    time_steps=200,
    dt=0.05
)

# Analyze probability distribution
final_state = get_simulation_result(result)
intensity = abs(final_state)**2

print("📊 Interference fringes visible on right side of slits!")
```

**Pattern:** Characteristic alternating bright/dark bands from constructive/destructive interference ✨

---

## ⚛️ Example 3: Quantum Tunneling

**Problem:** Electron tunneling through potential barrier

### Barrier Potential
```python
# Rectangular barrier: height 10 eV, width 20 units
barrier = create_custom_potential(
    function="10 if 100 < x < 120 else 0",
    grid_size=[256]
)
```

### Low-Energy Particle (Classically Forbidden)
```python
# Particle with energy = 5 eV (less than barrier height 10 eV)
particle = create_gaussian_wavepacket(
    grid_size=[256],
    position=[50],
    momentum=[3.0],         # Low kinetic energy
    width=8
)
```

### Simulate Tunneling
```python
result = solve_schrodinger(
    potential=barrier,
    initial_state=particle,
    time_steps=150,
    dt=0.1
)

# Analyze probability on right side
final_state = get_simulation_result(result)
prob_right = sum(abs(final_state[150:])**2)

print(f"🔍 Tunneling probability: {prob_right*100:.1f}%")
print("✓ Particle found on right side despite classical prohibition!")
```

**Key Insight:** Pure quantum effect - no classical explanation! 🚀

---

## 🏃 Example 4: Crystal Lattice Dynamics

**Electrons in crystalline solid**

### Square Lattice Potential
```python
from quantum_mcp import create_lattice_potential

# Periodic square lattice with spacing 20
lattice = create_lattice_potential(
    lattice_type="square",
    grid_size=[256],
    spacing=20,
    depth=5.0              # Potential well depth
)
```

### Wave Packet in Lattice
```python
# Quantum particle in periodic potential
particle = create_gaussian_wavepacket(
    grid_size=[256],
    position=[128],
    momentum=[2.0],
    width=12
)

# Simulate band structure effects
result = solve_schrodinger(
    potential=lattice,
    initial_state=particle,
    time_steps=100,
    dt=0.15
)

print("⚡ Bloch wave formation - periodic modulation of propagation!")
```

**Physical Result:** Demonstrates energy bands in solids 🧬

---

## 📈 Example 5: Wavefunction Analysis

**Extract physics from quantum states**

### Compute Observables
```python
from quantum_mcp import analyze_wavefunction

# After simulation
final_state = get_simulation_result(trajectory)

# Get position and momentum expectation values
observables = analyze_wavefunction(
    wavefunction=final_state,
    dx=0.5
)

print(f"⟨x⟩ = {observables['position_mean']} ± {observables['position_std']}")
print(f"⟨p⟩ = {observables['momentum_mean']} ± {observables['momentum_std']}")
print(f"⟨E⟩ = {observables['energy']} eV")
```

### Uncertainty Relation Check
```python
Δx = observables['position_std']
Δp = observables['momentum_std']
uncertainty_product = Δx * Δp

print(f"\nUncertainty Product: ΔxΔp = {uncertainty_product:.4f}")
print(f"ℏ/2 = {0.5:.4f}")
if uncertainty_product >= 0.5:
    print("✓ Satisfies Heisenberg Uncertainty Principle!")
```

---

## 🎓 Example 6: Progressive Complexity

### Beginner: Harmonic Oscillator
```python
# Quadratic potential: V(x) = x²/2
ho_potential = create_custom_potential(
    function="0.5*x**2",
    grid_size=[256]
)

# Gaussian ground state will oscillate in place
```

### Intermediate: Finite Potential Well
```python
# Box with soft walls instead of infinite
well = create_custom_potential(
    function="100 if abs(x-128) > 50 else 0",
    grid_size=[256]
)

# Particle can escape through tunneling
```

### Advanced: 2D Double Well
```python
# Two potential minima in 2D
dw_2d = create_custom_potential(
    function="(x-64)**2/100 + (x-192)**2/100 + y**2/50",
    grid_size=[256, 256]
)

# Quantum tunneling between wells in 2D
```

---

## 💡 Key Patterns

### Pattern 1: Superposition Dynamics
1. Create potential landscape
2. Initialize superposition of states
3. Observe evolution and interference

### Pattern 2: Scattering Problems
1. Plane wave incident
2. Potential barrier/well
3. Analyze reflection/transmission

### Pattern 3: Bound States
1. Confining potential
2. Study energy levels
3. Observe standing wave patterns

---

## 🔧 Advanced: GPU Acceleration

```python
# For 2D or fine-grid simulations, use GPU:
import os
os.environ['MCP_USE_GPU'] = '1'

# 500x500 grid simulation runs much faster!
result = solve_schrodinger_2d(
    potential=large_potential,
    initial_state=initial,
    time_steps=1000,
    dt=0.01,
    use_gpu=True
)
```

---

## 🌟 Integration with Other MCPs

- **Combine with Math MCP:** Derive analytical solutions, verify numerics
- **Connect to Neural MCP:** Train neural networks to predict scattering cross-sections
- **Use with Visualization:** Create publication-quality diagrams

See [API Reference](https://andylbrummer.github.io/math-mcp/api/quantum-mcp) for complete documentation.
