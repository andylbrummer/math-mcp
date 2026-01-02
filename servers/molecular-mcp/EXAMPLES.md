# scicomp-molecular-mcp Examples

Simulate molecular dynamics and explore particle interactions at the nanoscale.

## 🎬 Interactive Visual Demos

For spectacular real-time visualizations, try the **interactive demos**:

📺 **Watch animations:**
- [Galaxy Collision](https://andylbrummer.github.io/math-mcp/docs/demos/galaxy-collision) - N-body gravitational dynamics

🚀 **Run with Claude:**
```bash
claude -p "Simulate two galaxies colliding with gravitational N-body dynamics" \
  --allowedTools "mcp__molecular-mcp__*"
```

**Galaxy Collision Physics Notes:**
- View bounds use INITIAL frame only (prevents "postage stamp" effect)
- Slow approach velocity (0.15) - merge happens near end
- Per-particle colors: blue=#4da6ff, red=#ff6b6b distinguish galaxies

---

## 🧪 Example 1: Simple Liquid Simulation

**Simulate argon atoms interacting via Lennard-Jones potential**

### Initialize System
```python
from molecular_mcp import create_particles, add_potential

# Create 64 atoms in 4×4×4 Å box at 300K
system = create_particles(
    n_particles=64,
    box_size=[10.0, 10.0, 10.0],  # Ångströms
    temperature=300                 # Kelvin
)

# Add Lennard-Jones interaction (realistic for noble gases)
system = add_potential(
    system_id=system,
    potential_type="lennard_jones",
    sigma=3.4,                      # Å (atom size)
    epsilon=0.238                   # kcal/mol (interaction strength)
)

print("✓ System created: 64 argon atoms at 300K")
```

### Run NVE Simulation (Constant Energy)
```python
from molecular_mcp import run_md

# Microcanonical ensemble: E, V, N constant
trajectory = run_md(
    system_id=system,
    n_steps=1000,
    dt=0.001  # 1 fs timestep
)

print("🔬 Simulation complete: 1 picosecond of dynamics")
```

### Analyze Structure
```python
from molecular_mcp import compute_rdf

# Radial distribution function shows atomic arrangement
g_r = compute_rdf(
    trajectory_id=trajectory,
    n_bins=100
)

print("📊 g(r) peaks show:")
print("   First peak at ~3.8 Å - nearest neighbor distance")
print("   Second peak at ~7.0 Å - second shell structure")
```

**Result:** Realistic liquid argon structure with proper nearest-neighbor coordination ✨

---

## 🌡️ Example 2: Temperature Control (NVT Ensemble)

**Maintain constant temperature using thermostat**

### Heat a System from 100K to 500K
```python
# Start at low temperature
cold_system = create_particles(
    n_particles=100,
    box_size=[12.0, 12.0, 12.0],
    temperature=100
)

cold_system = add_potential(
    system_id=cold_system,
    potential_type="lennard_jones"
)

# NVT at 100K (solid phase)
result_100K = run_nvt(
    system_id=cold_system,
    n_steps=500,
    temperature=100,
    dt=0.001
)

# Now heat to 500K (liquid/gas transition)
result_500K = run_nvt(
    system_id=cold_system,
    n_steps=1000,
    temperature=500,
    dt=0.001
)

print("🔥 System behavior:")
print("   100K: Solid - atoms vibrate around fixed positions")
print("   500K: Liquid - atoms move freely")
```

### Track Temperature Change
```python
from molecular_mcp import analyze_temperature

# Get thermodynamic data from trajectory
thermo = analyze_temperature(trajectory_id=result_500K)

print(f"Temperature: {thermo['temperature']:.1f} K")
print(f"Kinetic Energy: {thermo['kinetic_energy']:.2f} kcal/mol")
print(f"Potential Energy: {thermo['potential_energy']:.2f} kcal/mol")
print(f"Total Energy: {thermo['total_energy']:.2f} kcal/mol")
```

---

## 💨 Example 3: Pressure Control (NPT Ensemble)

**Explore phase diagram at different pressures**

### Simulate at High Pressure
```python
from molecular_mcp import run_npt

# Create system
system = create_particles(
    n_particles=125,  # 5×5×5 cube
    box_size=[10.0, 10.0, 10.0],
    temperature=300
)

system = add_potential(
    system_id=system,
    potential_type="lennard_jones"
)

# NPT: Constant T and P (isobaric conditions)
# Simulates system compressed to 100 atm
trajectory = run_npt(
    system_id=system,
    n_steps=2000,
    temperature=300,
    pressure=100,          # atmospheres
    dt=0.001
)

print("🔬 System at 100 atm pressure")
```

### Detect Phase Transitions
```python
from molecular_mcp import detect_phase_transition

# Scan trajectory for phase changes
phases = detect_phase_transition(trajectory_id=trajectory)

for event in phases:
    print(f"Phase change at step {event['step']}")
    print(f"   Density change: {event['density_change']:.2f}")
    print(f"   Type: {event['transition_type']}")
```

**Result:** Observe liquid-solid transitions with pressure 🔄

---

## 📊 Example 4: Diffusion Analysis

**Calculate how fast molecules move (diffusivity)**

### Long Trajectory
```python
# Run long simulation to track atomic motion
long_traj = run_md(
    system_id=system,
    n_steps=5000,        # 5 ps trajectory
    dt=0.001
)
```

### Mean Squared Displacement
```python
from molecular_mcp import compute_msd

# MSD increases with time - indicates diffusion
msd = compute_msd(trajectory_id=long_traj)

print("📈 Diffusion Analysis:")
print(f"   MSD at 1 ps: {msd[1000]:.2f} Å²")
print(f"   MSD at 5 ps: {msd[5000]:.2f} Å²")

# Diffusion coefficient: D = MSD/(6t)
# Needed for viscosity, conductivity calculations
```

**Application:** Validate against experimental diffusion measurements 🎯

---

## 🏛️ Example 5: Ionic System (Coulomb Interaction)

**Simulate charged particles (salts, ionic liquids)**

### Lennard-Jones + Coulomb
```python
# Start with regular LJ particles
system = create_particles(
    n_particles=200,
    box_size=[15.0, 15.0, 15.0],
    temperature=298
)

# Add Lennard-Jones (short-range repulsion)
system = add_potential(
    system_id=system,
    potential_type="lennard_jones",
    sigma=3.0,
    epsilon=0.192
)

# Add Coulomb (long-range attraction/repulsion)
system = add_potential(
    system_id=system,
    potential_type="coulomb"
)

print("⚡ Ionic system: Charges attract/repel over long distances")
```

### Track Ion Clustering
```python
# Run simulation
result = run_npt(
    system_id=system,
    n_steps=3000,
    temperature=298,
    pressure=1.0,
    dt=0.001
)

# Analyze structure
rdf = compute_rdf(trajectory_id=result, n_bins=100)

print("🔌 Ionic clustering visible in g(r):")
print("   Opposite charges: enhanced peak (attraction)")
print("   Same charges: suppressed at short distance (repulsion)")
```

---

## 🎓 Example 6: Progressive Complexity

### Beginner: Single Timestep
```python
# Just create and analyze
system = create_particles(
    n_particles=8,
    box_size=[5.0, 5.0, 5.0],
    temperature=300
)
```

### Intermediate: Simple Dynamics
```python
# Small NVE run at constant energy
traj = run_md(
    system_id=system,
    n_steps=100,
    dt=0.001
)
```

### Advanced: Multi-Ensemble Study
```python
# NVE → NVT → NPT workflow
# Track structural changes with temperature/pressure
# Compute diffusion, viscosity, equation of state
```

---

## 💡 Key Patterns

### Pattern 1: Equilibration
1. Create system at target T
2. Run NVT to equilibrate
3. Switch to NVE or NPT for production

### Pattern 2: Phase Diagram Mapping
1. Vary temperature and pressure
2. Run NPT at each (T,P)
3. Measure density to build phase diagram

### Pattern 3: Property Calculation
1. Run long equilibrium trajectory
2. Compute RDF, MSD, energy
3. Calculate diffusion, compressibility

---

## 🔧 Advanced: GPU Acceleration

```python
# For 10,000+ particles, use GPU:
import os
os.environ['MCP_USE_GPU'] = '1'

large_system = create_particles(
    n_particles=10000,
    box_size=[50.0, 50.0, 50.0],
    temperature=300
)

# GPU simulation 50-100x faster!
result = run_md(
    system_id=large_system,
    n_steps=10000,
    dt=0.001,
    use_gpu=True
)
```

---

## 🌟 Real-World Applications

- **Material Science:** Predict crystal structures, melting points
- **Drug Discovery:** Simulate protein-ligand binding
- **Battery Design:** Study ion transport in electrolytes
- **Lubricants:** Understand film formation and friction

See [API Reference](https://andylbrummer.github.io/math-mcp/api/molecular-mcp) for complete documentation.
