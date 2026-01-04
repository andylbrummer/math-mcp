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
- `estimate_memory` - Estimate training memory requirements
- `compute_model_flops` - Compute model FLOPs
- `analyze_weights` - Analyze weight distributions
- `analyze_sparsity` - Compute model sparsity
- `analyze_norms` - Analyze layer norms
- `compare_models` - Compare model architectures

### Dataset Ablation
- `analyze_data_influence` - Compute sample influence
- `analyze_token_distribution` - Analyze token frequencies
- `analyze_sequences` - Compute sequence statistics
- `run_data_ablation` - Run ablation studies
- `suggest_augmentations` - Suggest data augmentation strategies

### Attention Visualization
- `visualize_attention` - Extract attention summary
- `analyze_attention_patterns` - Detect attention patterns
- `compute_head_rankings` - Rank heads by importance
- `compare_heads` - Compare attention heads

## License

MIT
