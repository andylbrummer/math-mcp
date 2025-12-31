# Triple-Slit Interference Demo

## The Prompt

```
Now show me triple-slit interference. Three slits should create a more
complex pattern with subsidiary maxima. Compare it to the double-slit case.
```

## MCP Tool Sequence

### Step 1: Create Triple-Slit Potential

```tool
mcp__quantum-mcp__create_custom_potential
  grid_size: [256, 256]
  function: "1000 * ((np.abs(x - 128) < 3) & (np.abs(y - 98) > 5) & (np.abs(y - 128) > 5) & (np.abs(y - 158) > 5))"
```

Three slits at y = 98, 128, 158 (equal spacing of 30 units).

### Step 2: Wide Coherent Wavepacket

```tool
mcp__quantum-mcp__create_gaussian_wavepacket
  grid_size: [256, 256]
  position: [40, 128]
  momentum: [2.5, 0]
  width: 30
```

### Step 3: Simulate

```tool
mcp__quantum-mcp__solve_schrodinger_2d
  potential: "potential://triple-slit"
  initial_state: <wavefunction>
  time_steps: 600
  dt: 0.02
```

### Step 4: Render

```tool
mcp__quantum-mcp__render_video
  simulation_id: "simulation://triple-slit"
  output_path: "/tmp/triple_slit.mp4"
  fps: 30
```

## Expected Results

### Pattern Structure
```
Double-Slit:           Triple-Slit:

  ██    ██    ██         ████  ██  ████  ██  ████
                             ↑        ↑
                         subsidiary  subsidiary
                           maxima     maxima
```

Key differences:
- **Sharper principal maxima** (narrower peaks)
- **Subsidiary maxima** appear between main peaks
- **More complex structure** from 3-wave interference

### Intensity Formula

For N slits:
$$I(\theta) = I_0 \left(\frac{\sin(N\beta/2)}{\sin(\beta/2)}\right)^2$$

where $\beta = \frac{2\pi d \sin\theta}{\lambda}$

For N=3: Principal maxima with 1 subsidiary maximum between each pair.

## Comparison Prompt

```
Create a side-by-side comparison showing single, double, and triple slit
interference patterns from the same incoming wavepacket.

For each case, extract the intensity pattern at a detector screen position
x = 200 and plot them together.
```

### Expected Comparison Tool Calls

1. Run all three simulations
2. Use `analyze_wavefunction` on final states
3. Extract 1D slices at detector position
4. Compare the patterns

## Toward Diffraction Gratings

```
What happens with 5 slits? 10 slits? Show me how the pattern evolves
as we approach a diffraction grating with many slits.
```

As N increases:
- Principal maxima become **sharper** (width ~ 1/N)
- More subsidiary maxima appear (N-2 between each pair)
- Approaches delta-function peaks of ideal grating
