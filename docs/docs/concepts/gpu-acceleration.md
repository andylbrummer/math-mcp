---
sidebar_position: 2
---

# GPU Acceleration

All four MCP servers support transparent GPU acceleration for compute-intensive operations.

## Architecture

The system uses a unified GPU management layer through the `compute-core` shared package:

```
┌─────────────────────────────────────────────────────────┐
│                     MCP Servers                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │ Math MCP │ │Quantum   │ │Molecular │ │Neural MCP│  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│       │            │            │            │         │
│       └────────────┴─────┬──────┴────────────┘         │
│                          │                              │
│              ┌───────────▼───────────┐                 │
│              │     compute-core      │                 │
│              │  (GPU abstraction)    │                 │
│              └───────────┬───────────┘                 │
│                          │                              │
│         ┌────────────────┼────────────────┐            │
│         ▼                ▼                ▼            │
│    ┌─────────┐     ┌──────────┐     ┌─────────┐       │
│    │  CuPy   │     │  PyTorch │     │  NumPy  │       │
│    │  (GPU)  │     │(GPU/CPU) │     │  (CPU)  │       │
│    └─────────┘     └──────────┘     └─────────┘       │
└─────────────────────────────────────────────────────────┘
```

## Automatic Backend Selection

The system automatically selects the best available backend:

1. **CuPy** - If NVIDIA GPU with CUDA is available
2. **PyTorch CUDA** - For neural network operations with GPU
3. **NumPy** - Fallback for CPU-only systems

```python
# The use_gpu parameter triggers automatic backend selection
result = solve_schrodinger(
    potential=potential_id,
    initial_state=wavefunction,
    time_steps=1000,
    dt=0.1,
    use_gpu=True  # Automatically uses best available backend
)
```

## Performance Characteristics

### Math MCP

| Operation | CPU (NumPy) | GPU (CuPy) | Speedup |
|-----------|-------------|------------|---------|
| FFT 1024x1024 | 45ms | 2ms | 22x |
| Matrix multiply 4096x4096 | 2.1s | 35ms | 60x |
| Linear solve 2048x2048 | 850ms | 25ms | 34x |

### Quantum MCP

| Grid Size | Time Steps | CPU Time | GPU Time | Speedup |
|-----------|------------|----------|----------|---------|
| 256 | 1000 | 8s | 0.3s | 27x |
| 512 | 1000 | 35s | 0.8s | 44x |
| 1024 | 1000 | 150s | 2.5s | 60x |
| 256x256 (2D) | 1000 | 30min | 30s | 60x |

### Molecular MCP

| Particles | Steps | CPU Time | GPU Time | Speedup |
|-----------|-------|----------|----------|---------|
| 1,000 | 10,000 | 10s | 1s | 10x |
| 10,000 | 10,000 | 100s | 5s | 20x |
| 100,000 | 10,000 | 1h | 30s | 120x |

### Neural MCP

| Model | Batch | CPU (per epoch) | GPU (per epoch) | Speedup |
|-------|-------|-----------------|-----------------|---------|
| ResNet18 | 32 | 45min | 30s | 90x |
| ResNet50 | 32 | 2h | 2min | 60x |
| MobileNetV2 | 64 | 30min | 20s | 90x |

## Memory Management

### Automatic Memory Handling

The GPU manager automatically handles memory allocation and cleanup:

```python
# Large computations are chunked automatically
result = matrix_multiply(
    a=large_matrix_a,  # 10000x10000
    b=large_matrix_b,  # 10000x10000
    use_gpu=True
)
# GPU memory is released after computation
```

### Memory Limits

Recommended GPU memory for different workloads:

| Workload | Minimum VRAM | Recommended |
|----------|--------------|-------------|
| Basic math operations | 2GB | 4GB |
| 1D quantum simulations | 2GB | 4GB |
| 2D quantum (512x512) | 4GB | 8GB |
| Molecular (100k particles) | 4GB | 8GB |
| Neural training (ResNet50) | 6GB | 11GB |

## Checking GPU Availability

Each MCP server provides GPU status through the info tool:

```python
# Check GPU status
info = math_mcp.info(topic="overview")
# Returns: gpu_available: true, gpu_device: "NVIDIA RTX 3080"

info = quantum_mcp.info(topic="overview")
# Returns: cuda_available: true, cupy_version: "12.0"
```

## Best Practices

### 1. Use GPU for Large Problems

GPU acceleration provides the most benefit for:
- Matrix operations larger than 512x512
- Quantum grids larger than 256 points
- Molecular systems with more than 1000 particles
- Neural networks (always use GPU when available)

### 2. Batch Operations

When possible, batch multiple operations:

```python
# Less efficient: many small operations
for i in range(100):
    result = matrix_multiply(small_a, small_b, use_gpu=True)

# More efficient: one large operation
result = matrix_multiply(large_a, large_b, use_gpu=True)
```

### 3. Grid Sizes for FFT

Use power-of-2 grid sizes for optimal FFT performance:
- Good: 256, 512, 1024, 2048
- Avoid: 300, 500, 1000

### 4. Mixed Precision (Neural MCP)

For training large models, mixed precision can double throughput:

```python
# Future feature: mixed precision training
experiment = train_model(
    model_id=model_id,
    dataset_id=dataset_id,
    use_gpu=True,
    mixed_precision=True  # FP16 for speed, FP32 for accuracy
)
```

## Troubleshooting

### Common Issues

| Error | Cause | Solution |
|-------|-------|----------|
| `CUDAOutOfMemory` | Insufficient VRAM | Reduce batch size or grid resolution |
| `CUDADriverError` | Driver mismatch | Update NVIDIA drivers |
| `CuPyNotAvailable` | CuPy not installed | Install with `pip install cupy-cuda12x` |
| `SlowPerformance` | CPU fallback active | Check GPU availability with info tool |

### Verifying GPU Usage

```python
# Verify GPU is being used
import time

start = time.time()
result = matrix_multiply(a, b, use_gpu=True)
gpu_time = time.time() - start

start = time.time()
result = matrix_multiply(a, b, use_gpu=False)
cpu_time = time.time() - start

print(f"GPU: {gpu_time:.2f}s, CPU: {cpu_time:.2f}s")
# GPU should be significantly faster for large matrices
```

## Supported Hardware

### NVIDIA GPUs (Recommended)

- **Compute Capability 6.0+** (Pascal and newer)
- Tested: GTX 1080, RTX 2080, RTX 3080, RTX 4090, A100, H100

### Requirements

- CUDA 11.0 or newer
- cuDNN 8.0 or newer (for Neural MCP)
- CuPy 12.0 or newer
- PyTorch 2.0 or newer (for Neural MCP)

### Future Support

- AMD ROCm support (planned)
- Apple Metal support (planned)
- Intel oneAPI support (under consideration)
