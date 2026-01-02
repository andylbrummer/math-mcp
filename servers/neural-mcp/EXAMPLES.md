# scicomp-neural-mcp Examples

Train neural networks and explore deep learning through practical examples.

## 🚀 Quick Start

Run directly with Claude:
```bash
claude -p "Train a simple neural network on MNIST and show accuracy" \
  --allowedTools "mcp__neural-mcp__*"
```

Or start an interactive session:
```bash
claude
# Then ask: "Load CIFAR-10 and train a ResNet model"
```

## 💡 What You Can Do

| Task | Tools |
|------|-------|
| **Model Building** | `define_model`, `load_pretrained`, `get_model_summary` |
| **Data Loading** | `load_dataset`, `create_dataloader` |
| **Training** | `train_model`, `get_experiment_status`, `evaluate_model` |
| **Analysis** | `compute_metrics`, `confusion_matrix`, `visualize_predictions` |
| **Optimization** | `tune_hyperparameters`, `plot_training_curves` |
| **Export** | `export_model` (ONNX, TorchScript) |

## 📚 Documentation

See the [full API documentation](https://andylbrummer.github.io/math-mcp/api/neural-mcp) for complete reference.

---

## 🧠 Example 1: Handwritten Digit Recognition (MNIST)

**Classic beginner project: Build and train image classifier**

### Step 1: Load Data
```python
from neural_mcp import load_dataset

# Load MNIST training data
train_data = load_dataset(
    dataset_name="MNIST",
    split="train"
)

# Load test data
test_data = load_dataset(
    dataset_name="MNIST",
    split="test"
)

print("✓ Loaded 60,000 training images")
print("✓ Loaded 10,000 test images")
print("  Each image: 28×28 grayscale (10 classes: 0-9)")
```

### Step 2: Build Model
```python
from neural_mcp import define_model

# Simple but effective model
model = define_model(
    architecture="custom",
    num_classes=10
)

print("🏗️ Model created:")
print("   Input: 28×28 images")
print("   Hidden: Two fully connected layers (128, 64 units)")
print("   Output: 10 probability scores (softmax)")
```

### Step 3: Train
```python
from neural_mcp import train_model

# Train the model
experiment = train_model(
    model_id=model,
    dataset_id=train_data,
    epochs=10,
    batch_size=32,
    learning_rate=0.001
)

print("🚀 Training started...")
print("   10 epochs, batch size 32")
print("   Learning rate: 0.001 (adam optimizer)")
```

### Step 4: Monitor Progress
```python
from neural_mcp import get_experiment_status

# Check training in real-time
status = get_experiment_status(experiment_id=experiment)

print(f"Epoch {status['current_epoch']}/{status['total_epochs']}")
print(f"Accuracy: {status['accuracy']:.1%}")
print(f"Loss: {status['loss']:.4f}")
```

### Step 5: Evaluate
```python
from neural_mcp import evaluate_model

# Test on unseen data
results = evaluate_model(
    model_id=model,
    dataset_id=test_data
)

print("📊 Test Results:")
print(f"   Accuracy: {results['accuracy']:.1%}")  # Should be ~97%+
print(f"   Precision: {results['precision']:.3f}")
print(f"   Recall: {results['recall']:.3f}")
```

---

## 🎨 Example 2: Image Classification - Advanced

**Classify dogs vs cats with pre-trained ResNet**

### Use Transfer Learning
```python
from neural_mcp import load_pretrained, train_model

# Load ResNet18 pre-trained on ImageNet
model = load_pretrained(
    model_name="resnet18",
    source="torchvision"
)

print("🔄 Transfer Learning:")
print("   Model: ResNet18 (pre-trained on 1000 ImageNet classes)")
print("   Strategy: Fine-tune last layer for binary classification")
```

### Train on Custom Data
```python
# Assume you have custom dog/cat dataset
custom_data = load_dataset(
    dataset_name="custom_pets",
    split="train"
)

# Quick training: only 2-3 epochs needed
# because weights already learned from ImageNet
fine_tuned = train_model(
    model_id=model,
    dataset_id=custom_data,
    epochs=3,
    batch_size=64,
    learning_rate=0.0001  # Small LR - preserve pre-trained weights
)

print("⚡ Fast training: 3 epochs with 10,000 images")
```

### Visualize Predictions
```python
from neural_mcp import visualize_predictions

# See what the model learned
visualizations = visualize_predictions(
    model_id=model,
    dataset_id=custom_data,
    n_samples=16
)

print("🖼️  Showing 16 random predictions:")
print("   Green boxes: Correct predictions ✓")
print("   Red boxes: Mistakes ✗ (what did it confuse?)")
```

---

## 📈 Example 3: Hyperparameter Optimization

**Find the best hyperparameters automatically**

### Search Space
```python
from neural_mcp import tune_hyperparameters

# Define parameter ranges to search
param_grid = {
    'learning_rate': [0.0001, 0.001, 0.01],
    'batch_size': [16, 32, 64],
    'dropout': [0.0, 0.5, 0.8]
}

# Grid search: try all 3×3×3 = 27 combinations
best_params = tune_hyperparameters(
    model_id=model,
    dataset_id=train_data,
    param_grid=param_grid,
    n_trials=27
)

print("🔍 Hyperparameter Search Results:")
print(f"   Best learning rate: {best_params['learning_rate']}")
print(f"   Best batch size: {best_params['batch_size']}")
print(f"   Best dropout: {best_params['dropout']}")
print(f"   Validation accuracy: {best_params['best_accuracy']:.1%}")
```

### Use Best Parameters
```python
# Re-train with optimal hyperparameters
final_model = train_model(
    model_id=model,
    dataset_id=train_data,
    epochs=20,
    batch_size=best_params['batch_size'],
    learning_rate=best_params['learning_rate']
)

print("✨ Training with optimized hyperparameters...")
```

---

## 🎯 Example 4: Confusion Analysis

**Understand which classes are hard to distinguish**

### Generate Confusion Matrix
```python
from neural_mcp import confusion_matrix

# After training, analyze errors
cm = confusion_matrix(
    model_id=model,
    dataset_id=test_data
)

print("🎯 Confusion Matrix (10×10 for MNIST):")
print("   Diagonal = correct predictions (100% is perfect)")
print("   Off-diagonal = confusions")
print("\nMost confused pairs:")
print("   4 vs 9: Model confuses '4' with '9' often")
print("   3 vs 8: Model confuses '3' with '8' often")
```

### Interpret Results
```python
print("\n💡 Insights:")
print("   - 4 and 9 look similar, need more examples")
print("   - 0 and 6 never confused (visually distinct)")
print("   - Model struggles with: 4, 9, 3, 8")
print("   → Consider data augmentation or extra training")
```

---

## 📊 Example 5: Training Visualization

**Watch loss decrease and accuracy improve**

### Plot Training Curves
```python
from neural_mcp import plot_training_curves

# Generate graphs from experiment
plots = plot_training_curves(
    experiment_id=experiment,
    output_path="training_curves.png"
)

print("📈 Generated plots:")
print("   - Loss over epochs (decreasing)")
print("   - Training vs validation accuracy")
print("   - Learning rate schedule (if used)")
```

### Expected Patterns
```
Loss Curve:
  ▲                      ✓ Healthy training
  │   ╱╲╱╲╱╲
  │  ╱  ╲    ╲___
  │╱
  └─────────────────> Time

Accuracy Curve:
    _____
   ╱     ╲___   ✓ Good: rises then plateaus
  ╱           ▔▔
```

---

## 🤖 Example 6: Custom Architecture

**Build your own neural network**

### Simple 2-Layer Network
```python
model = define_model(
    architecture="custom",
    num_classes=10
)

print("🏗️ Architecture:")
print("   Input: 28×28 = 784 neurons")
print("   Hidden 1: 256 neurons + ReLU")
print("   Hidden 2: 128 neurons + ReLU")
print("   Output: 10 neurons + Softmax")
print("   Total parameters: ~115,000")
```

### Add Complexity
```python
# More layers = more capacity (but risks overfitting)
# More regularization = prevent overfitting

model_deep = define_model(
    architecture="custom",
    num_classes=10
    # Note: architecture can be customized in your config
)

# Train with dropout for regularization
result = train_model(
    model_id=model_deep,
    dataset_id=train_data,
    epochs=20,
    batch_size=32,
    learning_rate=0.001
)
```

---

## 🎓 Example 7: Progressive Learning Path

### Beginner: MNIST
```
1. Load MNIST
2. Train simple model (3 min)
3. Evaluate on test set
4. Accuracy ~97%
```

### Intermediate: CIFAR-10
```
1. Load CIFAR-10 (colored objects)
2. Use ResNet18 (pre-trained)
3. Fine-tune on CIFAR
4. Accuracy ~90%
```

### Advanced: ImageNet
```
1. Load ImageNet subset
2. Custom architecture
3. Hyperparameter tuning
4. Accuracy ~75%+ on 1000 classes
```

---

## 💡 Key Patterns

### Pattern 1: Standard Workflow
1. Load dataset
2. Define or load model
3. Train (monitor with get_experiment_status)
4. Evaluate on test set
5. Analyze confusion matrix

### Pattern 2: Optimization
1. Baseline model (simple)
2. Tune hyperparameters
3. Use transfer learning
4. Add regularization
5. Iterate → better accuracy

### Pattern 3: Debugging
1. Train/val curves diverge? → Overfit (add dropout)
2. Both stay flat? → Underfit (bigger model)
3. Training slow? → Reduce learning rate or try SGD
4. Oscillating? → Learning rate too high

---

## 🔧 Advanced: GPU Training

```python
# For large models/datasets, use GPU:
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU 0

result = train_model(
    model_id=model,
    dataset_id=train_data,
    epochs=100,
    batch_size=512,        # Larger batches on GPU
    learning_rate=0.001,
    use_gpu=True
)

print("⚡ Training 10-50x faster on GPU!")
```

---

## 📦 Model Export

**Deploy your trained model**

```python
from neural_mcp import export_model

# Export to ONNX (universal format)
export_model(
    model_id=model,
    format="onnx",
    output_path="digit_classifier.onnx"
)

# Or TorchScript (PyTorch native)
export_model(
    model_id=model,
    format="torchscript",
    output_path="digit_classifier.pt"
)

print("✓ Model ready for deployment!")
```

---

## 🌟 Integration Examples

- **Combine with Math MCP:** Use symbolic differentiation to understand gradients
- **Use with Quantum MCP:** Train networks to predict quantum properties
- **Deploy Results:** Use exported models with molecular simulations

See [API Reference](https://andylbrummer.github.io/math-mcp/api/neural-mcp) for complete documentation.
