---
sidebar_position: 1
---

# API Reference Overview

The Math-Physics-ML MCP System provides 74 tools across 5 specialized MCP servers.

## Quick Navigation

- **[Math MCP](math-mcp)** - 14 tools for symbolic algebra and numerical computing
- **[Quantum MCP](quantum-mcp)** - 12 tools for wave mechanics and simulations
- **[Molecular MCP](molecular-mcp)** - 15 tools for molecular dynamics
- **[Neural MCP](neural-mcp)** - 16 tools for neural network training
- **[LLM MCP](llm-mcp)** - 17 tools for language model training and fine-tuning

## Tool Categories

### Math MCP

| Category | Tools | Purpose |
|----------|-------|---------|
| Discovery | `info` | Progressive capability discovery |
| Symbolic | `symbolic_solve`, `symbolic_diff`, `symbolic_integrate`, `symbolic_simplify` | Equation solving, calculus |
| Numerical | `create_array`, `matrix_multiply`, `solve_linear_system` | Array operations, linear algebra |
| Transforms | `fft`, `ifft` | Fast Fourier Transform |
| Optimization | `optimize_function`, `find_roots` | Function minimization, root finding |

### Quantum MCP

| Category | Tools | Purpose |
|----------|-------|---------|
| Discovery | `info` | Progressive capability discovery |
| Potentials | `create_lattice_potential`, `create_custom_potential` | Define quantum potentials |
| Wavepackets | `create_gaussian_wavepacket`, `create_plane_wave` | Initial states |
| Simulations | `solve_schrodinger`, `solve_schrodinger_2d` | Time evolution |
| Analysis | `analyze_wavefunction`, `get_simulation_result` | Extract observables |
| Visualization | `render_video`, `visualize_potential` | Generate plots/videos |

### Molecular MCP

| Category | Tools | Purpose |
|----------|-------|---------|
| Discovery | `info` | Progressive capability discovery |
| System | `create_particles`, `add_potential` | Define particle systems |
| Simulations | `run_md`, `run_nvt`, `run_npt` | Run MD in various ensembles |
| Analysis | `compute_rdf`, `compute_msd`, `analyze_temperature`, `detect_phase_transition` | Structural and thermodynamic analysis |
| Visualization | `density_field`, `render_trajectory` | Generate visualizations |

### Neural MCP

| Category | Tools | Purpose |
|----------|-------|---------|
| Discovery | `info` | Progressive capability discovery |
| Models | `define_model`, `load_pretrained`, `get_model_summary` | Model management |
| Data | `load_dataset`, `create_dataloader` | Dataset handling |
| Training | `train_model`, `get_experiment_status` | Training workflows |
| Evaluation | `evaluate_model`, `compute_metrics` | Model assessment |
| Tuning | `tune_hyperparameters` | Hyperparameter optimization |
| Visualization | `plot_training_curves`, `confusion_matrix` | Training analysis |
| Deployment | `export_model` | Model export |

### LLM MCP

| Category | Tools | Purpose |
|----------|-------|---------|
| Discovery | `info` | Progressive capability discovery |
| Models | `create_model`, `get_model_config`, `list_models` | GPT/Mamba model management |
| Tokenizers | `create_tokenizer`, `tokenize_text` | Text tokenization |
| Datasets | `load_dataset`, `prepare_dataset` | Training data preparation |
| Training | `create_trainer`, `train_step`, `get_training_status` | LLM training workflows |
| Evaluation | `evaluate_model`, `generate_text`, `compute_perplexity` | Model assessment |
| Checkpoints | `save_checkpoint`, `load_checkpoint` | Model persistence |
| Analysis | `analyze_attention`, `compute_gradient_norms` | Training analysis |

## Common Parameters

Most tools support these common parameters:

### GPU Acceleration
```python
use_gpu: bool = True  # Use GPU if available, fallback to CPU
```

### Progressive Discovery
```python
info(topic: Optional[str] = None)
# topic=None          → List all categories
# topic='category'    → Show tools in category
# topic='tool_name'   → Detailed help for tool
```

## Resource URIs

The system uses URI-based references for efficient data sharing:

| URI Pattern | Source | Purpose |
|-------------|--------|---------|
| `array://{id}` | Math MCP | Large numerical arrays |
| `potential://{id}` | Quantum MCP | Quantum potentials |
| `simulation://{id}` | Quantum MCP | Completed simulations |
| `system://{id}` | Molecular MCP | Particle systems |
| `trajectory://{id}` | Molecular MCP | MD trajectories |
| `model://{id}` | Neural/LLM MCP | Neural network and LLM models |
| `experiment://{id}` | Neural/LLM MCP | Training experiments |
| `tokenizer://{id}` | LLM MCP | Text tokenizers |
| `dataset://{id}` | Neural/LLM MCP | Training datasets |
| `checkpoint://{id}` | LLM MCP | Model checkpoints |

## Response Formats

All tools return JSON responses with a consistent structure:

### Success Response
```json
{
  "status": "success",
  "data": {
    // Tool-specific data
  }
}
```

### Error Response
```json
{
  "status": "error",
  "error": {
    "type": "ValueError",
    "message": "Clear error description",
    "suggestion": "How to fix the issue"
  }
}
```

### Async Task Response
```json
{
  "status": "submitted",
  "task_id": "task_abc123",
  "simulation_id": "sim_def456"  // Optional result ID
}
```

## GPU Support

All numerical tools support GPU acceleration:

```python
# Explicit GPU use
result = matrix_multiply(a, b, use_gpu=True)

# Automatic backend selection
xp = gpu_manager.get_array_module()
result = xp.matmul(a, b)
```

### GPU Availability Check

```python
# Via info tool
info(topic="gpu")
# Returns: {gpu_available: true, backend: "cuda", device_count: 1}
```

## Error Handling

Common error types and solutions:

| Error Type | Cause | Solution |
|------------|-------|----------|
| `ValueError` | Invalid input parameters | Check parameter ranges and types |
| `GPUMemoryError` | Insufficient GPU memory | Reduce array size or use CPU |
| `TimeoutError` | Operation took too long | Increase timeout or reduce complexity |
| `ResourceNotFoundError` | Invalid URI reference | Check URI exists and is accessible |

## Rate Limits

Default resource limits (configurable in config.kdl):

- Max array size: 100M elements
- Max particles: 10M
- Max time steps: 1M
- Max epochs: 1000
- Symbolic operation timeout: 30 seconds

## Next Steps

Explore the detailed API documentation for each server:

1. **[Math MCP API](math-mcp)** - Start here for foundational operations
2. **[Quantum MCP API](quantum-mcp)** - Quantum simulations
3. **[Molecular MCP API](molecular-mcp)** - Classical MD
4. **[Neural MCP API](neural-mcp)** - Deep learning
5. **[LLM MCP API](llm-mcp)** - Language model training and fine-tuning
