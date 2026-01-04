# LLM MCP Server

MCP server for LLM training, fine-tuning, and experimentation. Part of the scicomp-mcp suite.

## Features

- **Model Architectures**: GPT (Transformer decoder) and Mamba (State Space Model)
- **Tokenizers**: tiktoken, BPE, SentencePiece, character-level
- **Training**: AdamW, learning rate scheduling, gradient checkpointing, mixed precision
- **Evaluation**: Perplexity, loss, text generation

## Installation

```bash
uv sync --all-extras
```

## Usage

```bash
scicomp-llm-mcp
```

## Tools

### Model Management
- `create_model` - Create GPT or Mamba architecture
- `get_model_config` - Get model configuration
- `list_models` - List all models

### Tokenizers
- `create_tokenizer` - Create or load tokenizer
- `tokenize_text` - Tokenize text

### Datasets
- `load_dataset` - Load training dataset
- `prepare_dataset` - Prepare for training

### Training
- `create_trainer` - Configure training
- `train_step` - Execute training steps
- `get_training_status` - Monitor progress

### Evaluation
- `evaluate_model` - Evaluate on dataset
- `generate_text` - Generate text
- `compute_perplexity` - Compute perplexity

### Checkpoints
- `save_checkpoint` - Save model checkpoint
- `load_checkpoint` - Load from checkpoint

### Analysis
- `analyze_attention` - Analyze attention patterns
- `compute_gradient_norms` - Compute gradient norms

## License

MIT
