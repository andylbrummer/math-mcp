---
sidebar_position: 5
---

# Neural MCP

Neural network training and experimentation with GPU acceleration.

## Overview

Neural MCP provides 16 tools for deep learning workflows:
- **Discovery**: Progressive capability exploration
- **Model Definition**: Create and load neural network architectures
- **Data**: Load datasets and create data loaders
- **Training**: Train models with progress tracking
- **Evaluation**: Assess model performance
- **Tuning**: Hyperparameter optimization
- **Visualization**: Training curves and analysis
- **Deployment**: Export models for production

## Live Examples

These examples were generated using the actual Neural MCP tools:

### Defining a Model

```python
# Create a ResNet18 model for 10-class classification
result = define_model(
    architecture="resnet18",
    num_classes=10,
    pretrained=False
)
```

**Actual Result:**
```json
{
  "model_id": "model://e849779b-a9f6-4642-b2c1-b88a8810787b",
  "architecture": "resnet18",
  "num_classes": 10
}
```

### Loading a Dataset

```python
# Load CIFAR-10 training data
result = load_dataset(
    dataset_name="CIFAR10",
    split="train"
)
```

**Actual Result:**
```json
{
  "dataset_id": "dataset://2b47c74e-f589-47fd-8a6e-6c35c9a65351",
  "name": "CIFAR10",
  "split": "train"
}
```

### Model Summary

```python
result = get_model_summary(
    model_id="model://e849779b-a9f6-4642-b2c1-b88a8810787b"
)
```

**Actual Result:**
```json
{
  "architecture": "resnet18",
  "num_classes": 10,
  "total_params": 11000000,
  "trainable_params": 11000000
}
```

ResNet18 has 11 million trainable parameters!

## Tools Reference

### Model Definition

#### define_model

Create a neural network model from predefined architectures.

**Parameters:**
- `architecture` (string): Model architecture
  - `"resnet18"` - ResNet with 18 layers (11M params)
  - `"mobilenet"` - MobileNetV2 (3.5M params)
  - `"custom"` - Custom architecture specification
- `num_classes` (integer, optional): Output classes (default: 10)
- `pretrained` (boolean, optional): Use ImageNet weights (default: false)

**Returns:**
- `model_id`: URI reference to the model
- `architecture`: Architecture name
- `num_classes`: Number of output classes

**Example - Transfer Learning:**
```python
# Start with ImageNet-pretrained weights
model = define_model(
    architecture="resnet18",
    num_classes=100,  # Fine-tune for 100 classes
    pretrained=True
)
```

#### load_pretrained

Load a pretrained model from various sources.

**Parameters:**
- `model_name` (string): Model identifier
- `source` (string, optional): Source repository
  - `"torchvision"` (default)
  - `"huggingface"`

**Example:**
```python
# Load ResNet50 from torchvision
model = load_pretrained(
    model_name="resnet50",
    source="torchvision"
)
```

#### get_model_summary

Get detailed model architecture breakdown.

**Parameters:**
- `model_id` (string): Model URI

**Returns:**
- `architecture`: Model type
- `num_classes`: Output dimension
- `total_params`: Total parameters
- `trainable_params`: Trainable parameters
- `layers`: Layer-by-layer breakdown (if available)

### Data Tools

#### load_dataset

Load a standard dataset.

**Parameters:**
- `dataset_name` (string): Dataset name
  - `"CIFAR10"` - 60k 32×32 color images, 10 classes
  - `"MNIST"` - 70k 28×28 grayscale digits, 10 classes
  - `"ImageNet"` - 1.2M images, 1000 classes
- `split` (string, optional): Data split
  - `"train"` (default)
  - `"test"`
  - `"val"`

**Returns:**
- `dataset_id`: URI reference
- `name`: Dataset name
- `split`: Which split

**Example:**
```python
train_data = load_dataset("CIFAR10", split="train")
test_data = load_dataset("CIFAR10", split="test")
```

#### create_dataloader

Create a batched data loader for training.

**Parameters:**
- `dataset_id` (string): Dataset URI
- `batch_size` (integer, optional): Batch size (default: 32)
- `shuffle` (boolean, optional): Shuffle data (default: true)

**Returns:**
- `dataloader_id`: URI reference
- `n_batches`: Number of batches

**Example:**
```python
loader = create_dataloader(
    dataset_id=train_data["dataset_id"],
    batch_size=128,
    shuffle=True
)
```

### Training Tools

#### train_model

Train a model on a dataset.

**Parameters:**
- `model_id` (string): Model URI
- `dataset_id` (string): Training dataset URI
- `epochs` (integer, optional): Training epochs (default: 10)
- `batch_size` (integer, optional): Batch size (default: 32)
- `learning_rate` (number, optional): Learning rate (default: 0.001)
- `use_gpu` (boolean, optional): Use GPU (default: true)

**Returns:**
- `experiment_id`: URI to track training
- `task_id`: For async progress monitoring

**Example - Full Training:**
```python
experiment = train_model(
    model_id=model["model_id"],
    dataset_id=train_data["dataset_id"],
    epochs=50,
    batch_size=128,
    learning_rate=0.001,
    use_gpu=True
)
```

#### get_experiment_status

Monitor training progress.

**Parameters:**
- `experiment_id` (string): Experiment URI

**Returns:**
- `state`: "running", "completed", "failed"
- `current_epoch`: Current epoch number
- `train_loss`: Current training loss
- `train_accuracy`: Current training accuracy
- `val_loss`: Validation loss (if val set provided)
- `val_accuracy`: Validation accuracy

### Evaluation Tools

#### evaluate_model

Evaluate model on a test set.

**Parameters:**
- `model_id` (string): Model URI
- `dataset_id` (string): Test dataset URI

**Returns:**
- `accuracy`: Top-1 accuracy
- `loss`: Average loss
- `top5_accuracy`: Top-5 accuracy (for classification)
- `per_class_accuracy`: Accuracy by class

**Example:**
```python
metrics = evaluate_model(
    model_id=model["model_id"],
    dataset_id=test_data["dataset_id"]
)
# Expected: accuracy ~93% for ResNet18 on CIFAR10
```

#### compute_metrics

Compute advanced metrics.

**Parameters:**
- `model_id` (string): Model URI
- `dataset_id` (string): Dataset URI
- `metrics` (array): Metrics to compute
  - `"precision"`
  - `"recall"`
  - `"f1"`
  - `"confusion_matrix"`

**Returns:**
- Requested metrics for each class

### Hyperparameter Tuning

#### tune_hyperparameters

Automated hyperparameter search.

**Parameters:**
- `model_id` (string): Model URI
- `dataset_id` (string): Dataset URI
- `param_grid` (object): Parameters to search
- `n_trials` (integer, optional): Number of trials (default: 10)

**Returns:**
- `best_params`: Optimal hyperparameters
- `best_accuracy`: Best achieved accuracy
- `all_results`: Results from all trials

**Example:**
```python
tuning = tune_hyperparameters(
    model_id=model["model_id"],
    dataset_id=train_data["dataset_id"],
    param_grid={
        "learning_rate": [0.0001, 0.001, 0.01],
        "batch_size": [32, 64, 128],
        "optimizer": ["adam", "sgd"]
    },
    n_trials=20
)
```

### Visualization Tools

#### plot_training_curves

Generate loss and accuracy plots.

**Parameters:**
- `experiment_id` (string): Experiment URI
- `output_path` (string, optional): Save location

**Returns:**
- `loss_curve`: Path to loss plot
- `accuracy_curve`: Path to accuracy plot

#### confusion_matrix

Generate classification confusion matrix.

**Parameters:**
- `model_id` (string): Model URI
- `dataset_id` (string): Dataset URI

**Returns:**
- `matrix`: Confusion matrix array
- `labels`: Class labels
- `plot_path`: Path to visualization

#### visualize_predictions

Show sample predictions.

**Parameters:**
- `model_id` (string): Model URI
- `dataset_id` (string): Dataset URI
- `n_samples` (integer, optional): Number of samples (default: 10)

**Returns:**
- `predictions`: Sample predictions with images

### Deployment

#### export_model

Export model for production deployment.

**Parameters:**
- `model_id` (string): Model URI
- `format` (string, optional): Export format
  - `"onnx"` (default) - Open Neural Network Exchange
  - `"torchscript"` - PyTorch JIT
- `output_path` (string, optional): Save location

**Returns:**
- `export_path`: Path to exported model
- `format`: Export format used

**Example:**
```python
# Export for deployment
export_model(
    model_id=model["model_id"],
    format="onnx",
    output_path="model.onnx"
)
```

## Complete Training Pipeline

```python
# 1. Define model
model = define_model(
    architecture="resnet18",
    num_classes=10,
    pretrained=True  # Transfer learning
)

# 2. Load data
train_data = load_dataset("CIFAR10", split="train")
test_data = load_dataset("CIFAR10", split="test")

# 3. Create data loaders
train_loader = create_dataloader(
    train_data["dataset_id"],
    batch_size=128,
    shuffle=True
)

# 4. Train
experiment = train_model(
    model_id=model["model_id"],
    dataset_id=train_data["dataset_id"],
    epochs=30,
    batch_size=128,
    learning_rate=0.001,
    use_gpu=True
)

# 5. Monitor progress
while True:
    status = get_experiment_status(experiment["experiment_id"])
    print(f"Epoch {status['current_epoch']}: "
          f"Loss={status['train_loss']:.4f}, "
          f"Acc={status['train_accuracy']:.2%}")
    if status["state"] == "completed":
        break

# 6. Evaluate
metrics = evaluate_model(
    model_id=model["model_id"],
    dataset_id=test_data["dataset_id"]
)
print(f"Test Accuracy: {metrics['accuracy']:.2%}")

# 7. Visualize
plot_training_curves(
    experiment_id=experiment["experiment_id"],
    output_path="training.png"
)
confusion_matrix(
    model_id=model["model_id"],
    dataset_id=test_data["dataset_id"]
)

# 8. Export for deployment
export_model(
    model_id=model["model_id"],
    format="onnx",
    output_path="cifar10_resnet18.onnx"
)
```

## Supported Architectures

| Architecture | Parameters | CIFAR10 Accuracy | Training Time* |
|-------------|------------|------------------|----------------|
| ResNet18    | 11M        | ~93%            | 15 min         |
| ResNet50    | 25M        | ~94%            | 30 min         |
| MobileNetV2 | 3.5M       | ~91%            | 10 min         |
| VGG16       | 138M       | ~92%            | 45 min         |

*Training time for 50 epochs on NVIDIA RTX 3080

## Performance Tips

1. **Batch Size**: Larger batches = faster training, but may need to adjust LR
2. **Learning Rate**: Start with 0.001, use scheduler for decay
3. **Transfer Learning**: Pretrained weights dramatically speed convergence
4. **Mixed Precision**: Enable for 2x speedup on modern GPUs
5. **Data Augmentation**: Improves generalization significantly

## Error Handling

| Error | Cause | Solution |
|-------|-------|----------|
| `CUDAOutOfMemory` | Batch too large | Reduce batch_size |
| `LossIsNaN` | LR too high | Reduce learning_rate |
| `Overfitting` | Not enough regularization | Add dropout, data augmentation |
| `SlowConvergence` | LR too low | Increase learning_rate |
