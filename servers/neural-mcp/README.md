# scicomp-neural-mcp

mcp-name: io.github.andylbrummer/neural-mcp

MCP server for neural network training and experimentation.

## Overview

This server provides tools for building, training, and analyzing neural networks:

- **Model building** - Pre-trained models (ResNet, MobileNet) and custom architectures
- **Dataset loading** - CIFAR-10, MNIST, ImageNet datasets with standard preprocessing
- **Training** - Full training loops with configurable learning rates and batch sizes
- **Evaluation** - Model evaluation, metrics computation, and analysis
- **Hyperparameter tuning** - Automated hyperparameter search
- **Export** - Export models to ONNX and TorchScript formats
- **GPU acceleration** - Optional CUDA acceleration for training

## Installation & Usage

```bash
# Run directly with uvx (no installation required)
uvx scicomp-neural-mcp

# Or install with pip
pip install scicomp-neural-mcp

# With GPU support (recommended for training)
pip install scicomp-neural-mcp[gpu]

# Run as command
scicomp-neural-mcp
```

## Available Tools

### Model Management
- `define_model` - Create neural network models (ResNet18, MobileNet, custom)
- `load_pretrained` - Load pretrained models from torchvision or Hugging Face
- `get_model_summary` - Get detailed layer-by-layer breakdown
- `export_model` - Export to ONNX or TorchScript

### Data Loading
- `load_dataset` - Load standard datasets (CIFAR-10, MNIST, ImageNet)
- `create_dataloader` - Create batched dataloaders with shuffling

### Training
- `train_model` - Train model on dataset with configurable parameters
- `get_experiment_status` - Monitor training progress
- `evaluate_model` - Evaluate on test set

### Analysis
- `compute_metrics` - Compute detailed performance metrics
- `confusion_matrix` - Generate confusion matrices
- `plot_training_curves` - Visualize loss and accuracy curves
- `visualize_predictions` - Inspect model predictions on samples

### Hyperparameter Optimization
- `tune_hyperparameters` - Automated hyperparameter search

## Configuration

Enable GPU acceleration with environment variable:

```bash
MCP_USE_GPU=1 scicomp-neural-mcp
```

## Examples

### 📖 Code Examples
Practical tutorials in [EXAMPLES.md](EXAMPLES.md):
- MNIST digit recognition (complete workflow)
- Transfer learning with ResNet
- Hyperparameter optimization
- Confusion matrix analysis
- Progressive learning path (beginner → advanced)

### 📚 Full Documentation
See the [API documentation](https://andylbrummer.github.io/math-mcp/api/neural-mcp) for complete API reference.

## Part of Math-Physics-ML MCP System

Part of a comprehensive system for scientific computing. See the [documentation](https://andylbrummer.github.io/math-mcp/) for the complete ecosystem.
