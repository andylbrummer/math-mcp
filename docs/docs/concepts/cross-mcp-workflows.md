---
sidebar_position: 3
---

# Cross-MCP Workflows

One of the most powerful features of the Math-Physics-ML MCP system is the ability to chain tools across different servers, creating sophisticated computational pipelines.

## How It Works

Each MCP server produces URI references that can be consumed by other servers:

```
Math MCP          Quantum MCP         Molecular MCP       Neural MCP
─────────         ───────────         ─────────────       ──────────
array://...   →   potential://...  →  trajectory://...  → model://...
               ↘                   ↗
                 └─ shared data ─┘
```

## Workflow Patterns

### Pattern 1: Math → Quantum

Use Math MCP to create custom potential functions, then simulate quantum dynamics:

```python
# Step 1: Create custom potential with Math MCP
# Double-well potential for quantum tunneling studies
potential_array = math_mcp.create_array(
    shape=[512],
    fill_type="function",
    function="0.01*(x-128)**2*(x-384)**2 - 500"
)
# Returns: array://abc123...

# Step 2: Use in Quantum MCP simulation
potential = quantum_mcp.create_custom_potential(
    array_uri="array://abc123...",
    grid_size=[512]
)

# Step 3: Create initial state and simulate
psi = quantum_mcp.create_gaussian_wavepacket(
    grid_size=[512],
    position=[128],
    momentum=[0],
    width=15
)

simulation = quantum_mcp.solve_schrodinger(
    potential=potential["potential_id"],
    initial_state=psi,
    time_steps=5000,
    dt=0.05,
    use_gpu=True
)
```

### Pattern 2: Molecular → Math Analysis

Run molecular simulations, then analyze results with advanced math tools:

```python
# Step 1: Run molecular dynamics simulation
system = molecular_mcp.create_particles(
    n_particles=1000,
    box_size=[20, 20, 20],
    temperature=1.0
)

molecular_mcp.add_potential(
    system_id=system["system_id"],
    potential_type="lennard_jones"
)

trajectory = molecular_mcp.run_md(
    system_id=system["system_id"],
    n_steps=100000,
    dt=0.002
)

# Step 2: Get trajectory data
data = molecular_mcp.get_trajectory(
    trajectory_id=trajectory["trajectory_id"]
)

# Step 3: Perform FFT analysis on velocity autocorrelation
# to extract vibrational frequencies
velocities = data["velocities"]
fft_result = math_mcp.fft(
    array=velocities,
    use_gpu=True
)

# Step 4: Find peaks in frequency spectrum
peaks = math_mcp.find_roots(
    function="derivative of spectrum",
    variables=["frequency"],
    initial_guess=[0.1]
)
```

### Pattern 3: Neural → Quantum Surrogate

Train neural networks on quantum simulation data for fast surrogate models:

```python
# Step 1: Generate quantum simulation dataset
training_data = []
for barrier_height in range(5, 50, 5):
    potential = quantum_mcp.create_custom_potential(
        grid_size=[256],
        function=f"{barrier_height}*exp(-(x-128)**2/200)"
    )

    psi = quantum_mcp.create_gaussian_wavepacket(
        grid_size=[256],
        position=[64],
        momentum=[2.0],
        width=10
    )

    sim = quantum_mcp.solve_schrodinger(
        potential=potential["potential_id"],
        initial_state=psi,
        time_steps=1000,
        dt=0.1
    )

    # Calculate transmission coefficient
    result = quantum_mcp.get_simulation_result(sim["simulation_id"])
    training_data.append({
        "barrier_height": barrier_height,
        "transmission": calculate_transmission(result)
    })

# Step 2: Train neural network to predict transmission
model = neural_mcp.define_model(
    architecture="custom",
    num_classes=1  # Regression output
)

# Custom dataset from quantum simulations
experiment = neural_mcp.train_model(
    model_id=model["model_id"],
    dataset=training_data,  # Custom dataset
    epochs=100
)

# Step 3: Use trained model for fast predictions
# 1000x faster than full quantum simulation!
```

### Pattern 4: Multi-Scale Simulation

Combine molecular and quantum simulations for multi-scale physics:

```python
# Quantum region: Electron density calculation
electron_potential = quantum_mcp.create_lattice_potential(
    lattice_type="square",
    grid_size=[64, 64],
    depth=10,
    spacing=8
)

electron_sim = quantum_mcp.solve_schrodinger_2d(
    potential=electron_potential["potential_id"],
    initial_state=initial_wavefunction,
    time_steps=500,
    dt=0.01
)

# Extract effective potential for classical nuclei
electron_density = quantum_mcp.get_simulation_result(
    electron_sim["simulation_id"]
)

# Classical region: Nuclear dynamics with quantum-derived forces
nuclear_system = molecular_mcp.create_particles(
    n_particles=100,
    box_size=[64, 64, 64],
    temperature=300
)

# Add effective potential from quantum calculation
molecular_mcp.add_potential(
    system_id=nuclear_system["system_id"],
    potential_type="custom",
    potential_data=electron_density["effective_potential"]
)

nuclear_trajectory = molecular_mcp.run_nvt(
    system_id=nuclear_system["system_id"],
    n_steps=10000,
    temperature=300
)
```

## Real-World Applications

### Drug Discovery Pipeline

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Math MCP  │     │ Molecular   │     │ Neural MCP  │
│             │     │    MCP      │     │             │
│ Generate    │────▶│ Simulate    │────▶│ Predict     │
│ molecular   │     │ binding     │     │ binding     │
│ geometries  │     │ dynamics    │     │ affinity    │
└─────────────┘     └─────────────┘     └─────────────┘
```

### Materials Science

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Quantum MCP │     │   Math MCP  │     │ Molecular   │
│             │     │             │     │    MCP      │
│ Calculate   │────▶│ Fit force   │────▶│ Large-scale │
│ electronic  │     │ field       │     │ simulation  │
│ structure   │     │ parameters  │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
```

### Machine Learning for Physics

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Quantum/    │     │   Math MCP  │     │ Neural MCP  │
│ Molecular   │     │             │     │             │
│ Generate    │────▶│ Feature     │────▶│ Train       │
│ training    │     │ extraction  │     │ surrogate   │
│ data        │     │ & analysis  │     │ model       │
└─────────────┘     └─────────────┘     └─────────────┘
```

## Best Practices

### 1. Data Format Compatibility

All MCPs use consistent data formats:
- **Arrays**: NumPy-compatible, serialized as JSON lists
- **URIs**: `type://uuid` format for referencing objects
- **Complex numbers**: `{"real": x, "imag": y}` dictionaries

### 2. Error Propagation

Handle errors at each step:

```python
# Check each step succeeded before proceeding
potential = quantum_mcp.create_custom_potential(...)
if "error" in potential:
    raise RuntimeError(f"Potential creation failed: {potential['error']}")

simulation = quantum_mcp.solve_schrodinger(
    potential=potential["potential_id"],
    ...
)
if "error" in simulation:
    raise RuntimeError(f"Simulation failed: {simulation['error']}")
```

### 3. Memory Management

For large workflows, clean up intermediate results:

```python
# Large simulation workflow
for config in configurations:
    # Run simulation
    result = run_simulation(config)

    # Extract only needed data
    summary = extract_summary(result)
    summaries.append(summary)

    # Allow garbage collection of large arrays
    del result

# Analyze summaries
final_result = analyze_summaries(summaries)
```

### 4. Parallel Execution

Independent operations can run in parallel:

```python
# These can run simultaneously
import concurrent.futures

with concurrent.futures.ThreadPoolExecutor() as executor:
    # Start all simulations
    futures = [
        executor.submit(quantum_mcp.solve_schrodinger, ...)
        for config in configurations
    ]

    # Collect results
    results = [f.result() for f in futures]
```

## Example: Complete Physics Pipeline

Here's a complete example combining all four MCPs:

```python
# Goal: Study quantum effects on molecular diffusion

# 1. MATH MCP: Create analytical potential
potential_expr = math_mcp.symbolic_simplify(
    expression="a*exp(-(x-x0)**2/(2*s**2)) + b*sin(k*x)"
)

potential_array = math_mcp.create_array(
    shape=[256],
    fill_type="function",
    function="5*exp(-(x-128)**2/100) + 2*sin(0.1*x)"
)

# 2. QUANTUM MCP: Solve for quantum ground state
potential = quantum_mcp.create_custom_potential(
    array_uri=potential_array["array_id"],
    grid_size=[256]
)

# Find ground state via imaginary time evolution
ground_state = quantum_mcp.solve_schrodinger(
    potential=potential["potential_id"],
    initial_state=trial_wavefunction,
    time_steps=1000,
    dt=-0.1j  # Imaginary time
)

# 3. MOLECULAR MCP: Classical dynamics with quantum corrections
system = molecular_mcp.create_particles(
    n_particles=500,
    box_size=[50, 50, 50],
    temperature=1.0
)

# Add quantum-corrected potential
molecular_mcp.add_potential(
    system_id=system["system_id"],
    potential_type="custom",
    quantum_correction=ground_state
)

trajectory = molecular_mcp.run_nvt(
    system_id=system["system_id"],
    n_steps=500000,
    temperature=1.0
)

msd = molecular_mcp.compute_msd(trajectory["trajectory_id"])
diffusion = msd["diffusion_coefficient"]

# 4. NEURAL MCP: Train model to predict diffusion from parameters
training_data = collect_diffusion_data()  # Many simulations

model = neural_mcp.define_model(
    architecture="custom",
    num_classes=1
)

neural_mcp.train_model(
    model_id=model["model_id"],
    dataset=training_data,
    epochs=50
)

# Now predict diffusion 1000x faster!
predicted_D = neural_mcp.predict(model["model_id"], new_params)
```

This workflow demonstrates the power of combining mathematical analysis, quantum mechanics, classical dynamics, and machine learning in a single unified pipeline.
