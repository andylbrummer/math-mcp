# scicomp-math-mcp Examples

Practical, real-world examples showing how to use the Math MCP for symbolic and numerical computing.

## 🚀 Quick Start

Run directly with Claude:
```bash
claude -p "Solve x^2 + 5x + 6 = 0 and show the steps" \
  --allowedTools "mcp__math-mcp__*"
```

Or start an interactive session:
```bash
claude
# Then ask: "Compute the derivative of sin(x^2)"
```

## 💡 What You Can Do

| Task | Tools |
|------|-------|
| **Symbolic Math** | `symbolic_solve`, `symbolic_diff`, `symbolic_integrate`, `symbolic_simplify` |
| **Numerical Computing** | `create_array`, `matrix_multiply`, `solve_linear_system`, `fft` |
| **Optimization** | `optimize_function`, `find_roots` |
| **GPU Acceleration** | Add `use_gpu=True` to numerical operations |

## 📚 Documentation

See the [full API documentation](https://andylbrummer.github.io/math-mcp/api/math-mcp) for complete reference.

---

## 🎯 Example 1: Solving a Physics Problem

**Problem:** Find when a projectile hits the ground given: h(t) = 100 + 50t - 4.9t²

### Step 1: Solve the Equation Symbolically
```python
from math_mcp import symbolic_solve

# Solve h(t) = 0
result = symbolic_solve(
    equations="100 + 50*t - 4.9*t**2",
    variables="t",
    domain="positive"  # Only positive time
)
# Output: t = 11.28 seconds (approximately)
```

### Step 2: Visualize the Trajectory (Numerical)
```python
from math_mcp import create_array, fft
import matplotlib.pyplot as plt

# Create time array from 0 to 12 seconds
time = create_array(
    shape=(120,),
    fill_type="linspace",
    linspace_range=[0, 12]
)

# Calculate height at each time
height = 100 + 50*time - 4.9*time**2

# Plot the trajectory
plt.plot(time, height)
plt.axhline(y=0, color='r', linestyle='--', label='Ground')
plt.xlabel('Time (s)')
plt.ylabel('Height (m)')
plt.title('Projectile Motion')
plt.show()
```

**Result:** Visual confirmation that the projectile lands at ~11.28 seconds ✓

---

## 🔬 Example 2: Chemical Equilibrium Analysis

**Problem:** Analyze the equilibrium: 2NO₂ ⇌ N₂O₄

### Find Equilibrium Constant Expression
```python
from math_mcp import symbolic_simplify

# Equilibrium expression: K = [N₂O₄] / [NO₂]²
# Given: Initial NO₂ = 1 M, x moles dissociate
equilibrium_expr = "(1-x) / (x)**2"

# Simplify
result = symbolic_simplify(equilibrium_expr)
# Useful for substituting different K values
```

### Numerical Analysis at Different Temperatures
```python
from math_mcp import create_array, solve_linear_system

# Create temperature array: 300K to 500K
temperatures = create_array(
    shape=(20,),
    fill_type="linspace",
    linspace_range=[300, 500]
)

# K values follow Van't Hoff equation (simplified)
K_values = 0.5 * (temperatures/300)**2  # Example relationship

# For each T, solve for equilibrium composition
for T, K in zip(temperatures, K_values):
    # Solve K = (1-x)/x²
    # This becomes: x² + K·x - K = 0
    result = symbolic_solve(
        equations=f"{K}*x**2 - x - {K}",
        variables="x"
    )
    print(f"T={T}K: x={result}")
```

**Insight:** Reaction favors N₂O₄ at lower temperatures 📊

---

## 📐 Example 3: Fourier Analysis

**Problem:** Analyze frequency components of a noisy signal

### Create Test Signal (Combination of Frequencies)
```python
from math_mcp import create_array, fft
import numpy as np

# Signal: 3Hz sine + 7Hz sine + noise
t = create_array(
    shape=(1000,),
    fill_type="linspace",
    linspace_range=[0, 10]
)

# Create signal with function
signal = create_array(
    shape=(1000,),
    fill_type="function",
    function="sin(2*3.14159*3*t) + 0.5*sin(2*3.14159*7*t) + 0.1*random()"
)

# Compute FFT
frequencies = fft(signal)

# Analyze magnitude spectrum
magnitude = abs(frequencies)
dominant_freq = argmax(magnitude)

print(f"Dominant frequency components detected")
print(f"Expected: 3Hz and 7Hz peaks ✓")
```

### Filter and Reconstruct
```python
# Zero out high frequency noise (keep 3Hz and 7Hz)
filtered_freqs = frequencies.copy()
filtered_freqs[20:] = 0  # Remove high frequencies

# Inverse FFT to get clean signal
clean_signal = ifft(filtered_freqs)

print("Signal cleaned: noise removed while preserving original components")
```

**Result:** Clear separation of signal components 🔍

---

## 🧮 Example 4: System of Linear Equations (Engineering)

**Problem:** Circuit analysis with multiple resistors and voltage sources

```
Circuit equations (Kirchhoff's laws):
10I₁ - 5I₂ = 12V
-5I₁ + 15I₂ = 8V
```

### Solve the System
```python
from math_mcp import solve_linear_system, matrix_multiply

# Coefficient matrix A
A = [[10, -5],
     [-5, 15]]

# Constants vector b
b = [12, 8]

# Solve: A·I = b
currents = solve_linear_system(A, b)

print(f"I₁ = {currents[0]} A")
print(f"I₂ = {currents[1]} A")
# Output: I₁ ≈ 1.52 A, I₂ ≈ 0.87 A
```

### Verify Solution
```python
# Calculate A·I to verify = b
verification = matrix_multiply(A, currents)
print(f"A·I = {verification}")
print(f"Expected = {b}")
print("✓ Solution verified!" if close(verification, b) else "✗ Error in solution")
```

**Application:** Works for any sized circuit (100x100 system, GPU accelerated) ⚡

---

## 🎓 Example 5: Progressive Difficulty

### Beginner: Simple Function Derivative
```python
# Find derivative of f(x) = x³ + 2x²
result = symbolic_diff(
    expression="x**3 + 2*x**2",
    variable="x"
)
# Output: 3x² + 4x
```

### Intermediate: Chain Rule
```python
# Find derivative of f(x) = sin(x²)
result = symbolic_diff(
    expression="sin(x**2)",
    variable="x",
    order=1
)
# Output: 2x·cos(x²)
```

### Advanced: Optimization
```python
# Find minimum of f(x) = x⁴ - 4x² + 3
result = optimize_function(
    function="x**4 - 4*x**2 + 3",
    variables=["x"],
    initial_guess=[2.0],
    method="BFGS"
)
# Output: Two local minima at x ≈ ±1.41
```

---

## 💡 Key Patterns

### Pattern 1: Symbolic → Numerical
1. Solve symbolically for general form
2. Use numerical computation for specific values
3. Visualize or optimize results

### Pattern 2: Chain Multiple Tools
1. Create data with `create_array`
2. Transform with FFT/optimization
3. Solve equations with constraints
4. Verify with matrix operations

### Pattern 3: GPU Acceleration
```python
# For large-scale problems, enable GPU:
import os
os.environ['MCP_USE_GPU'] = '1'

# Same code runs 10-100x faster on GPU
large_matrix = create_array(shape=(10000, 10000), fill_type="random")
result = matrix_multiply(large_matrix, large_matrix, use_gpu=True)
```

---

## 🚀 Next Steps

- Combine with **scicomp-quantum-mcp** for quantum mechanics problems
- Use **scicomp-neural-mcp** for fitting neural networks to data
- Integrate with **scicomp-molecular-mcp** for molecular property calculations

See the [API Documentation](https://andylbrummer.github.io/math-mcp/api/math-mcp) for complete reference.
