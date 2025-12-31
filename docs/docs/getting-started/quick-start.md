---
sidebar_position: 1
---

# Quick Start

Get started with the Math-Physics-ML MCP servers in minutes. This guide provides simple examples for each MCP server to help you understand their capabilities.

## Prerequisites

Before starting, ensure you have:

1. Installed the MCP servers (see [Installation](./installation.md))
2. Configured Claude Desktop or Claude Code (see [Configuration](./configuration.md))

## Math MCP

The Math MCP provides symbolic algebra and GPU-accelerated numerical computing.

### Symbolic Algebra

Solve equations symbolically:

```python
# Solve quadratic equation
symbolic_solve(equations="x**2 - 5*x + 6", variables=["x"])
# Result: x = 2, x = 3

# Solve system of equations
symbolic_solve(
    equations=["x + y - 5", "x - y - 1"],
    variables=["x", "y"]
)
# Result: x = 3, y = 2
```

Compute derivatives:

```python
# First derivative
symbolic_diff(expression="x**3 + 2*x**2 + x", variable="x")
# Result: 3*x**2 + 4*x + 1

# Second derivative
symbolic_diff(expression="sin(x)", variable="x", order=2)
# Result: -sin(x)
```

Compute integrals:

```python
# Indefinite integral
symbolic_integrate(expression="x**2", variable="x")
# Result: x**3/3

# Definite integral
symbolic_integrate(expression="x**2", variable="x", limits=[0, 1])
# Result: 1/3
```

### Numerical Computing

Create and manipulate arrays:

```python
# Create random matrix
arr1 = create_array(shape=[100, 100], fill_type="random")

# Create matrix using a function
arr2 = create_array(
    shape=[10, 10],
    fill_type="function",
    function="x**2 + y"
)

# Create linspace array
arr3 = create_array(
    shape=[100],
    fill_type="linspace",
    linspace_range=[0, 10]
)
```

Matrix operations:

```python
# GPU-accelerated matrix multiplication
arr1 = create_array(shape=[1000, 1000], fill_type="random")
arr2 = create_array(shape=[1000, 1000], fill_type="random")
result = matrix_multiply(a=arr1, b=arr2, use_gpu=True)

# Solve linear system Ax = b
a = create_array(shape=[100, 100], fill_type="random")
b = create_array(shape=[100], fill_type="random")
solution = solve_linear_system(a=a, b=b, use_gpu=True)
```

### FFT Analysis

Fast Fourier Transform for signal processing:

```python
# Create a signal with multiple frequencies
signal = create_array(
    shape=[1024],
    fill_type="function",
    function="sin(2*pi*5*x) + 0.5*sin(2*pi*10*x)"
)

# Compute FFT
spectrum = fft(array=signal, use_gpu=True)

# Inverse FFT
reconstructed = ifft(array=spectrum, use_gpu=True)
```

### Optimization

Find function minima and roots:

```python
# Minimize a function
optimize_function(
    function="x**2 + y**2",
    variables=["x", "y"],
    initial_guess=[1.0, 1.0],
    method="BFGS"
)
# Result: x = 0, y = 0

# Find roots of equation
find_roots(
    function="x**2 - 4",
    variables=["x"],
    initial_guess=[1.0]
)
# Result: x = 2.0
```

## Quantum MCP

The Quantum MCP simulates quantum mechanical systems and solves the Schrödinger equation.

### Create Potentials

Define potential energy landscapes:

```python
# Create square lattice potential
potential = create_lattice_potential(
    lattice_type="square",
    grid_size=[256],
    depth=10.0,
    spacing=1.0
)

# Create custom potential from function
potential = create_custom_potential(
    grid_size=[256],
    function="10*exp(-(x-128)**2/100)"
)
```

### Create Wavefunctions

Initialize quantum states:

```python
# Create Gaussian wavepacket
psi = create_gaussian_wavepacket(
    grid_size=[256],
    position=[64],
    momentum=[2.0],
    width=5.0
)

# Create plane wave
plane = create_plane_wave(
    grid_size=[256],
    momentum=[1.0]
)
```

### Solve Schrödinger Equation

Time-evolve quantum systems:

```python
# 1D time-dependent Schrödinger equation
simulation = solve_schrodinger(
    potential=potential,
    initial_state=psi,
    time_steps=100,
    dt=0.1,
    use_gpu=True
)

# 2D Schrödinger equation
simulation_2d = solve_schrodinger_2d(
    potential=potential_2d,
    initial_state=psi_2d,
    time_steps=100,
    dt=0.1,
    use_gpu=True
)
```

### Analysis and Visualization

Analyze simulation results:

```python
# Analyze wavefunction properties
analysis = analyze_wavefunction(
    wavefunction=psi,
    dx=1.0
)
# Returns: energy, position uncertainty, momentum uncertainty

# Visualize potential
visualize_potential(
    potential_id=potential,
    output_path="potential.png"
)

# Create animation of wavefunction evolution
render_video(
    simulation_id=simulation,
    output_path="evolution.mp4",
    fps=30
)
```

## Molecular MCP

The Molecular MCP performs classical molecular dynamics simulations.

### Create Particle Systems

Initialize molecular systems:

```python
# Create particle system
system = create_particles(
    n_particles=1000,
    box_size=[50, 50, 50],
    temperature=1.0
)
```

### Add Interactions

Define interatomic potentials:

```python
# Add Lennard-Jones potential
add_potential(
    system_id=system,
    potential_type="lennard_jones",
    epsilon=1.0,
    sigma=1.0
)

# Add Coulomb potential (for charged systems)
add_potential(
    system_id=system,
    potential_type="coulomb"
)
```

### Run Simulations

Perform molecular dynamics:

```python
# NVE ensemble (constant energy)
trajectory = run_md(
    system_id=system,
    n_steps=10000,
    dt=0.001,
    use_gpu=True
)

# NVT ensemble (constant temperature)
trajectory = run_nvt(
    system_id=system,
    n_steps=10000,
    temperature=1.5
)

# NPT ensemble (constant pressure and temperature)
trajectory = run_npt(
    system_id=system,
    n_steps=10000,
    temperature=1.5,
    pressure=1.0
)
```

### Analyze Trajectories

Compute structural and dynamical properties:

```python
# Radial distribution function
rdf = compute_rdf(
    trajectory_id=trajectory,
    n_bins=100
)

# Mean squared displacement
msd = compute_msd(trajectory_id=trajectory)

# Temperature analysis
temp_stats = analyze_temperature(trajectory_id=trajectory)

# Detect phase transitions
phase = detect_phase_transition(trajectory_id=trajectory)

# Visualize density field
density_field(
    trajectory_id=trajectory,
    frame=-1  # Last frame
)

# Create trajectory animation
render_trajectory(
    trajectory_id=trajectory,
    output_path="trajectory.mp4"
)
```

## Neural MCP

The Neural MCP enables neural network training and experimentation.

### Define Models

Create neural network architectures:

```python
# Use pretrained model
model = define_model(
    architecture="resnet18",
    num_classes=10,
    pretrained=True
)

# Create custom architecture
model = define_model(
    architecture="mobilenet",
    num_classes=100,
    pretrained=False
)
```

### Load Datasets

Access standard datasets:

```python
# Load training data
train_dataset = load_dataset(
    dataset_name="CIFAR10",
    split="train"
)

# Load test data
test_dataset = load_dataset(
    dataset_name="CIFAR10",
    split="test"
)

# Create dataloader with batching
dataloader = create_dataloader(
    dataset_id=train_dataset,
    batch_size=32,
    shuffle=True
)
```

### Train Models

Train neural networks:

```python
# Train with default parameters
experiment = train_model(
    model_id=model,
    dataset_id=train_dataset,
    epochs=10,
    batch_size=64,
    learning_rate=0.001,
    use_gpu=True
)

# Monitor training progress
status = get_experiment_status(experiment_id=experiment)
```

### Evaluate and Analyze

Assess model performance:

```python
# Evaluate on test set
metrics = evaluate_model(
    model_id=model,
    dataset_id=test_dataset
)

# Get model architecture summary
summary = get_model_summary(model_id=model)

# Compute advanced metrics
advanced_metrics = compute_metrics(
    model_id=model,
    dataset_id=test_dataset,
    metrics=["precision", "recall", "f1"]
)

# Visualize predictions
visualize_predictions(
    model_id=model,
    dataset_id=test_dataset,
    n_samples=10
)

# Generate confusion matrix
confusion_matrix(
    model_id=model,
    dataset_id=test_dataset
)

# Plot training curves
plot_training_curves(
    experiment_id=experiment,
    output_path="training.png"
)
```

### Hyperparameter Tuning

Optimize model hyperparameters:

```python
# Run hyperparameter search
tune_hyperparameters(
    model_id=model,
    dataset_id=train_dataset,
    param_grid={
        "learning_rate": [0.001, 0.01, 0.1],
        "batch_size": [32, 64, 128]
    },
    n_trials=10
)
```

### Export Models

Save models for deployment:

```python
# Export to ONNX
export_model(
    model_id=model,
    format="onnx",
    output_path="model.onnx"
)

# Export to TorchScript
export_model(
    model_id=model,
    format="torchscript",
    output_path="model.pt"
)
```

## Cross-MCP Workflows

MCPs can work together using resource URIs:

```python
# Math → Quantum workflow
# 1. Create potential array in Math MCP
potential_array = create_array(
    shape=[256],
    fill_type="function",
    function="10*exp(-(x-128)**2/100)"
)
# Returns: array://abc123...

# 2. Use in Quantum MCP
simulation = solve_schrodinger(
    potential="array://abc123...",  # Reference Math MCP array
    initial_state=psi,
    time_steps=100
)
```

## GPU Acceleration

Enable GPU acceleration for significant speedups:

```python
# Math MCP: 100x speedup on large matrices
result = matrix_multiply(a=large_matrix_1, b=large_matrix_2, use_gpu=True)

# Quantum MCP: 6x speedup on wave simulations
sim = solve_schrodinger(potential=pot, initial_state=psi, time_steps=1000, use_gpu=True)

# Molecular MCP: 10x+ speedup on MD simulations
traj = run_md(system_id=sys, n_steps=100000, dt=0.001, use_gpu=True)

# Neural MCP: Essential for practical training
exp = train_model(model_id=model, dataset_id=data, epochs=50, use_gpu=True)
```

## Next Steps

- Explore [API Reference](../api/overview) for complete tool documentation
- Learn about [Architecture](../concepts/architecture) for deeper understanding
- Read about [GPU Acceleration](../concepts/gpu-acceleration) for performance optimization
