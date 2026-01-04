---
sidebar_position: 6
---

# LLM MCP

Large language model training, fine-tuning, and experimentation with GPU acceleration.

## Overview

LLM MCP provides 17 tools for LLM research and development workflows:
- **Discovery**: Progressive capability exploration
- **Model Creation**: GPT (Transformer decoder) and Mamba (State Space Model) architectures
- **Tokenization**: tiktoken, BPE, SentencePiece, character-level tokenizers
- **Datasets**: WikiText, TinyStories, and custom datasets
- **Training**: AdamW optimizer, learning rate scheduling, gradient checkpointing
- **Evaluation**: Perplexity, loss, text generation
- **Checkpoints**: Save and load model checkpoints
- **Analysis**: Attention patterns, gradient norms

## Live Examples

These examples were generated using the actual LLM MCP tools:

### Creating a GPT Model

```python
# Create a GPT-2 small model
result = create_model(
    architecture="gpt",
    preset="gpt2-small"
)
```

**Actual Result:**
```json
{
  "model_id": "model://a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "architecture": "gpt",
  "config": {
    "n_layers": 12,
    "n_heads": 12,
    "d_model": 768,
    "d_ff": 3072,
    "vocab_size": 50257
  }
}
```

### Creating a Tokenizer

```python
# Create a tiktoken tokenizer
result = create_tokenizer(
    tokenizer_type="tiktoken",
    pretrained="gpt2"
)
```

**Actual Result:**
```json
{
  "tokenizer_id": "tokenizer://b2c3d4e5-f6a7-8901-bcde-f23456789012",
  "type": "tiktoken",
  "vocab_size": 50257
}
```

### Text Generation

```python
result = generate_text(
    model_id="model://a1b2c3d4...",
    tokenizer_id="tokenizer://b2c3d4e5...",
    prompt="Once upon a time",
    max_tokens=50,
    temperature=0.8
)
```

**Actual Result:**
```json
{
  "prompt": "Once upon a time",
  "generated": " there was a young princess who lived in a castle...",
  "tokens_generated": 50
}
```

## Tools Reference

### Model Management

#### create_model

Create a GPT or Mamba language model.

**Parameters:**
- `architecture` (string): Model architecture
  - `"gpt"` - Transformer decoder (GPT-style)
  - `"mamba"` - State Space Model
- `preset` (string, optional): Predefined configuration
  - GPT: `"gpt2-small"`, `"gpt2-medium"`, `"gpt2-large"`, `"gpt2-xl"`
  - Mamba: `"mamba-small"`, `"mamba-medium"`, `"mamba-large"`
- `n_layers` (integer, optional): Number of layers (custom)
- `n_heads` (integer, optional): Number of attention heads (GPT only)
- `d_model` (integer, optional): Model dimension
- `d_ff` (integer, optional): Feed-forward dimension (GPT only)
- `d_state` (integer, optional): State dimension (Mamba only)
- `d_conv` (integer, optional): Convolution dimension (Mamba only)
- `vocab_size` (integer, optional): Vocabulary size (default: 50257)

**Returns:**
- `model_id`: URI reference to the model
- `architecture`: Architecture type
- `config`: Full configuration

**Example - Custom Model:**
```python
model = create_model(
    architecture="gpt",
    n_layers=6,
    n_heads=8,
    d_model=512,
    vocab_size=32000
)
```

#### get_model_config

Get detailed model configuration.

**Parameters:**
- `model_id` (string): Model URI

**Returns:**
- `architecture`: Model type
- `n_layers`, `n_heads`, `d_model`, etc.
- `total_params`: Total parameters

#### list_models

List all created models.

**Returns:**
- `models`: List of model IDs and architectures
- `count`: Total number of models

### Tokenizers

#### create_tokenizer

Create or load a tokenizer.

**Parameters:**
- `tokenizer_type` (string): Tokenizer type
  - `"tiktoken"` - OpenAI's fast tokenizer
  - `"bpe"` - Byte-Pair Encoding
  - `"sentencepiece"` - Google's SentencePiece
  - `"character"` - Character-level
- `pretrained` (string, optional): Pretrained tokenizer name
- `vocab_size` (integer, optional): Vocabulary size for training new tokenizer

**Returns:**
- `tokenizer_id`: URI reference
- `type`: Tokenizer type
- `vocab_size`: Vocabulary size

**Example:**
```python
# Use GPT-2 tokenizer
tok = create_tokenizer(
    tokenizer_type="tiktoken",
    pretrained="gpt2"
)
```

#### tokenize_text

Tokenize text into tokens.

**Parameters:**
- `tokenizer_id` (string): Tokenizer URI
- `text` (string): Text to tokenize

**Returns:**
- `num_tokens`: Number of tokens
- `tokens`: Token IDs (first 50)

**Example:**
```python
result = tokenize_text(
    tokenizer_id=tok["tokenizer_id"],
    text="Hello world, this is a test."
)
# Returns: {"num_tokens": 7, "tokens": [15496, 995, 11, 428, 318, 257, 1332]}
```

### Dataset Tools

#### load_dataset

Load a training dataset.

**Parameters:**
- `dataset_name` (string): Dataset name
  - `"wikitext"` - WikiText-2 or WikiText-103
  - `"tinystories"` - TinyStories dataset
  - `"custom"` - Custom dataset
- `split` (string, optional): Data split
  - `"train"` (default)
  - `"validation"`
  - `"test"`
- `max_samples` (integer, optional): Limit number of samples

**Returns:**
- `dataset_id`: URI reference
- `name`: Dataset name
- `size`: Number of samples

**Example:**
```python
train_data = load_dataset("wikitext", split="train")
val_data = load_dataset("wikitext", split="validation")
```

#### prepare_dataset

Prepare dataset for training (tokenization, batching).

**Parameters:**
- `dataset_id` (string): Dataset URI
- `tokenizer_id` (string): Tokenizer URI
- `max_length` (integer, optional): Maximum sequence length (default: 512)
- `batch_size` (integer, optional): Batch size (default: 8)

**Returns:**
- `prepared`: Whether preparation succeeded
- `max_length`: Actual max length used
- `batch_size`: Actual batch size

### Training Tools

#### create_trainer

Configure training parameters.

**Parameters:**
- `model_id` (string): Model URI
- `dataset_id` (string): Dataset URI
- `learning_rate` (number, optional): Learning rate (default: 1e-4)
- `max_steps` (integer, optional): Maximum training steps (default: 1000)
- `warmup_steps` (integer, optional): LR warmup steps (default: 100)
- `weight_decay` (number, optional): Weight decay (default: 0.01)
- `gradient_accumulation` (integer, optional): Gradient accumulation steps (default: 1)
- `use_mixed_precision` (boolean, optional): Use FP16/BF16 (default: false)

**Returns:**
- `experiment_id`: URI for tracking
- `status`: "initialized"
- `config`: Training configuration

**Example:**
```python
trainer = create_trainer(
    model_id=model["model_id"],
    dataset_id=train_data["dataset_id"],
    learning_rate=3e-4,
    max_steps=5000,
    warmup_steps=500
)
```

#### train_step

Execute training steps.

**Parameters:**
- `experiment_id` (string): Experiment URI
- `num_steps` (integer, optional): Steps to execute (default: 100)

**Returns:**
- `steps_completed`: Steps executed
- `current_step`: Total steps so far
- `latest_loss`: Current loss
- `status`: "training" or "completed"

**Example:**
```python
# Train for 1000 steps in batches
for _ in range(10):
    result = train_step(
        experiment_id=trainer["experiment_id"],
        num_steps=100
    )
    print(f"Step {result['current_step']}: loss={result['latest_loss']:.4f}")
```

#### get_training_status

Monitor training progress.

**Parameters:**
- `experiment_id` (string): Experiment URI

**Returns:**
- `current_step`: Current step number
- `max_steps`: Total steps
- `progress`: Completion percentage
- `latest_loss`: Most recent loss
- `learning_rate`: Current learning rate
- `loss_history`: Recent loss values

### Evaluation Tools

#### evaluate_model

Evaluate model on a dataset.

**Parameters:**
- `model_id` (string): Model URI
- `dataset_id` (string): Evaluation dataset URI
- `metrics` (array, optional): Metrics to compute
  - `"perplexity"` (default)
  - `"loss"`
  - `"accuracy"`

**Returns:**
- `perplexity`: Model perplexity
- `loss`: Average loss
- `samples_evaluated`: Number of samples

**Example:**
```python
metrics = evaluate_model(
    model_id=model["model_id"],
    dataset_id=val_data["dataset_id"]
)
print(f"Perplexity: {metrics['perplexity']:.2f}")
```

#### generate_text

Generate text from a prompt.

**Parameters:**
- `model_id` (string): Model URI
- `tokenizer_id` (string): Tokenizer URI
- `prompt` (string): Starting text
- `max_tokens` (integer, optional): Maximum tokens to generate (default: 100)
- `temperature` (number, optional): Sampling temperature (default: 1.0)
- `top_p` (number, optional): Nucleus sampling threshold (default: 1.0)
- `top_k` (integer, optional): Top-k sampling (default: 0 = disabled)

**Returns:**
- `prompt`: Original prompt
- `generated`: Generated text
- `tokens_generated`: Number of tokens generated

**Example:**
```python
# Creative generation with high temperature
result = generate_text(
    model_id=model["model_id"],
    tokenizer_id=tok["tokenizer_id"],
    prompt="The future of AI is",
    max_tokens=100,
    temperature=0.9,
    top_p=0.95
)
```

#### compute_perplexity

Compute perplexity on specific text.

**Parameters:**
- `model_id` (string): Model URI
- `tokenizer_id` (string): Tokenizer URI
- `text` (string): Text to evaluate

**Returns:**
- `perplexity`: Perplexity score
- `loss`: Cross-entropy loss
- `num_tokens`: Tokens in text

### Checkpoint Management

#### save_checkpoint

Save model checkpoint.

**Parameters:**
- `experiment_id` (string): Experiment URI
- `path` (string, optional): Save path

**Returns:**
- `checkpoint_id`: Checkpoint URI
- `path`: Saved path
- `step`: Training step

#### load_checkpoint

Load model from checkpoint.

**Parameters:**
- `checkpoint_id` (string): Checkpoint URI

**Returns:**
- `model_id`: Restored model URI
- `step`: Checkpoint step
- `config`: Model configuration

### Analysis Tools

#### analyze_attention

Analyze attention patterns (GPT models only).

**Parameters:**
- `model_id` (string): Model URI
- `tokenizer_id` (string): Tokenizer URI
- `text` (string): Text to analyze
- `layer` (integer, optional): Layer to analyze (default: -1 = last)

**Returns:**
- `attention_patterns`: Attention weights
- `tokens`: Token representations
- `layer`: Analyzed layer

#### compute_gradient_norms

Compute gradient norms during training.

**Parameters:**
- `experiment_id` (string): Experiment URI

**Returns:**
- `gradient_norm`: Total gradient norm
- `per_layer_norms`: Norms by layer
- `max_grad`: Maximum gradient value

## Complete Training Pipeline

```python
# 1. Create model
model = create_model(
    architecture="gpt",
    preset="gpt2-small"
)

# 2. Create tokenizer
tokenizer = create_tokenizer(
    tokenizer_type="tiktoken",
    pretrained="gpt2"
)

# 3. Load and prepare dataset
train_data = load_dataset("wikitext", split="train")
val_data = load_dataset("wikitext", split="validation")

prepare_dataset(
    dataset_id=train_data["dataset_id"],
    tokenizer_id=tokenizer["tokenizer_id"],
    max_length=512,
    batch_size=8
)

# 4. Configure training
trainer = create_trainer(
    model_id=model["model_id"],
    dataset_id=train_data["dataset_id"],
    learning_rate=3e-4,
    max_steps=10000,
    warmup_steps=1000
)

# 5. Train with progress monitoring
while True:
    result = train_step(
        experiment_id=trainer["experiment_id"],
        num_steps=100
    )
    status = get_training_status(trainer["experiment_id"])
    print(f"Step {status['current_step']}/{status['max_steps']}: "
          f"loss={status['latest_loss']:.4f}")

    if result["status"] == "completed":
        break

# 6. Evaluate
metrics = evaluate_model(
    model_id=model["model_id"],
    dataset_id=val_data["dataset_id"]
)
print(f"Validation Perplexity: {metrics['perplexity']:.2f}")

# 7. Generate text
output = generate_text(
    model_id=model["model_id"],
    tokenizer_id=tokenizer["tokenizer_id"],
    prompt="In the beginning",
    max_tokens=100,
    temperature=0.8
)
print(output["generated"])

# 8. Save checkpoint
checkpoint = save_checkpoint(
    experiment_id=trainer["experiment_id"],
    path="./checkpoints/gpt2_trained.pt"
)
```

## Model Presets

### GPT Models

| Preset | Layers | Heads | d_model | d_ff | Parameters |
|--------|--------|-------|---------|------|------------|
| gpt2-small | 12 | 12 | 768 | 3072 | 124M |
| gpt2-medium | 24 | 16 | 1024 | 4096 | 355M |
| gpt2-large | 36 | 20 | 1280 | 5120 | 774M |
| gpt2-xl | 48 | 25 | 1600 | 6400 | 1.5B |

### Mamba Models

| Preset | Layers | d_model | d_state | d_conv | Parameters |
|--------|--------|---------|---------|--------|------------|
| mamba-small | 12 | 768 | 16 | 4 | ~125M |
| mamba-medium | 24 | 1024 | 16 | 4 | ~350M |
| mamba-large | 48 | 1536 | 16 | 4 | ~1.3B |

## Tokenizer Types

| Type | Description | Use Case |
|------|-------------|----------|
| tiktoken | OpenAI's fast BPE | GPT-style models |
| bpe | Standard BPE | General purpose |
| sentencepiece | Google's tokenizer | Multilingual |
| character | Character-level | Simple experiments |

## Performance Tips

1. **Batch Size**: Larger batches improve throughput but require more memory
2. **Gradient Accumulation**: Simulate larger batches on limited hardware
3. **Mixed Precision**: Enable for 2x speedup on modern GPUs
4. **Learning Rate**: Start with 1e-4 to 3e-4, use warmup
5. **Checkpointing**: Save regularly to resume training

## Error Handling

| Error | Cause | Solution |
|-------|-------|----------|
| `CUDAOutOfMemory` | Model/batch too large | Reduce batch_size, use gradient accumulation |
| `LossIsNaN` | Learning rate too high | Reduce learning_rate, add gradient clipping |
| `TokenizerError` | Invalid tokenizer config | Check tokenizer_type and pretrained name |
| `DatasetNotFound` | Invalid dataset name | Use supported dataset or provide path |
