---
sidebar_position: 7
title: GPT Training Demo
---

# GPT Model Training

Train a GPT-style language model from scratch using the LLM MCP tools.

## Overview

This demo shows how to:
- Create a GPT model with custom architecture
- Set up tokenization with tiktoken
- Load and prepare training data
- Train with progress monitoring
- Evaluate and generate text

## The Workflow

### Step 1: Create Model

```bash
claude -p "Create a small GPT model with 4 layers, 4 heads, 256 model dimension" \
  --allowedTools "mcp__llm-mcp__*"
```

**Result:**
```json
{
  "model_id": "model://gpt-custom-abc123",
  "architecture": "gpt",
  "config": {
    "n_layers": 4,
    "n_heads": 4,
    "d_model": 256,
    "vocab_size": 50257
  }
}
```

### Step 2: Create Tokenizer

```bash
claude -p "Create a tiktoken tokenizer using GPT-2 encoding" \
  --allowedTools "mcp__llm-mcp__*"
```

**Result:**
```json
{
  "tokenizer_id": "tokenizer://tiktoken-xyz789",
  "type": "tiktoken",
  "vocab_size": 50257
}
```

### Step 3: Load Dataset

```bash
claude -p "Load the TinyStories dataset for training" \
  --allowedTools "mcp__llm-mcp__*"
```

**Result:**
```json
{
  "dataset_id": "dataset://tinystories-train",
  "name": "tinystories",
  "size": 2119719
}
```

### Step 4: Configure Training

```bash
claude -p "Create a trainer for my GPT model with learning rate 3e-4, \
  max steps 1000, and warmup 100 steps" \
  --allowedTools "mcp__llm-mcp__*"
```

**Result:**
```json
{
  "experiment_id": "experiment://train-gpt-001",
  "status": "initialized",
  "config": {
    "learning_rate": 0.0003,
    "max_steps": 1000,
    "warmup_steps": 100
  }
}
```

### Step 5: Train

```bash
claude -p "Train my model for 500 steps and report the loss" \
  --allowedTools "mcp__llm-mcp__*"
```

**Result:**
```json
{
  "steps_completed": 500,
  "current_step": 500,
  "latest_loss": 2.34,
  "status": "training"
}
```

### Step 6: Generate Text

```bash
claude -p "Generate 50 tokens from the prompt 'Once upon a time' with temperature 0.8" \
  --allowedTools "mcp__llm-mcp__*"
```

**Result:**
```json
{
  "prompt": "Once upon a time",
  "generated": " there was a little girl named Lily. She loved to play...",
  "tokens_generated": 50
}
```

## Complete Training Session

Run a full training pipeline:

```bash
claude -p "Train a small GPT model on TinyStories: create a 4-layer model, \
  load TinyStories dataset, train for 1000 steps with lr=3e-4, \
  then generate a story starting with 'The little robot'" \
  --allowedTools "mcp__llm-mcp__*"
```

## Model Presets

| Preset | Layers | Parameters | Use Case |
|--------|--------|------------|----------|
| gpt2-small | 12 | 124M | General text |
| gpt2-medium | 24 | 355M | Better quality |
| gpt2-large | 36 | 774M | High quality |

## Training Tips

### Learning Rate

- Start with `3e-4` for small models
- Use `1e-4` for larger models
- Enable warmup (10% of total steps)

### Batch Size

- Larger batches = smoother gradients
- Use gradient accumulation for limited memory

### Monitoring

Track these metrics:
- **Loss**: Should decrease steadily
- **Perplexity**: Lower is better (exponential of loss)
- **Learning rate**: Check warmup and decay

## Comparing Architectures

### GPT vs Mamba

```bash
# Train GPT model
claude -p "Create a GPT model with 6 layers and train on WikiText for 500 steps"

# Train Mamba model
claude -p "Create a Mamba model with 6 layers and train on WikiText for 500 steps"
```

Mamba (State Space Model) offers:
- Linear complexity vs quadratic for attention
- Better for very long sequences
- Competitive quality with less compute

## Evaluation Metrics

### Perplexity

Measures how "surprised" the model is by the text:
- `< 20`: Excellent (trained model)
- `20-50`: Good
- `50-100`: Moderate
- `> 100`: Poor (random init)

### Generation Quality

Assess generations with:
- **Coherence**: Do sentences make sense?
- **Fluency**: Is the language natural?
- **Relevance**: Does it follow the prompt?

## Run It Yourself

### Interactive Session

```bash
cd /path/to/math-mcp
claude --allowedTools "mcp__llm-mcp__*"
```

Then try prompts like:
- "Create a GPT model and show me the config"
- "Train my model for 100 steps"
- "Generate a story about a curious cat"

### One-Shot Training

```bash
claude -p "Train a GPT model on TinyStories for 500 steps and generate text" \
  --allowedTools "mcp__llm-mcp__*"
```

## Related Resources

- [LLM MCP API Reference](/api/llm-mcp) - Complete tool documentation
- [Neural MCP](/api/neural-mcp) - For image classification models
- [Machine Learning Examples](/examples/examples-ml-ai) - More prompt ideas
