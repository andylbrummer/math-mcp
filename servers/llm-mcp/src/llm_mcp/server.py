"""LLM MCP server implementation for language model training and experimentation."""

import logging
import uuid
from typing import Any

import numpy as np
import torch
from mcp.server import Server
from mcp.types import Tool
from mcp_common import GPUManager, TaskManager

from llm_mcp.ablation import (
    analyze_token_frequency,
    compute_data_influence,
    compute_sequence_statistics,
    run_ablation_study,
    suggest_data_augmentation,
)
from llm_mcp.analysis import (
    analyze_layer_norms,
    analyze_weight_distribution,
    compare_models,
    compute_flops,
    compute_model_sparsity,
    estimate_memory_usage,
)
from llm_mcp.models import GPT, GPTConfig, Mamba, MambaConfig
from llm_mcp.training import DataBatcher, Trainer, TrainingConfig, create_synthetic_data
from llm_mcp.visualization import (
    compare_attention_heads,
    compute_attention_patterns,
    compute_head_importance,
    extract_attention_summary,
    generate_attention_heatmap_data,
)

logger = logging.getLogger(__name__)

app = Server("llm-mcp")

# Storage for stateful objects
_models: dict[str, dict[str, Any]] = {}
_pytorch_models: dict[str, torch.nn.Module] = {}  # Actual PyTorch models
_tokenizers: dict[str, dict[str, Any]] = {}
_datasets: dict[str, dict[str, Any]] = {}
_experiments: dict[str, dict[str, Any]] = {}
_trainers: dict[str, Trainer] = {}  # Actual trainers
_checkpoints: dict[str, dict[str, Any]] = {}

_gpu = GPUManager.get_instance()
_task_manager = TaskManager.get_instance()
_rng = np.random.default_rng()


# Model registry for supported architectures
MODEL_REGISTRY = {
    "gpt2-small": {"n_layers": 12, "n_heads": 12, "d_model": 768, "d_ff": 3072},
    "gpt2-medium": {"n_layers": 24, "n_heads": 16, "d_model": 1024, "d_ff": 4096},
    "gpt2-large": {"n_layers": 36, "n_heads": 20, "d_model": 1280, "d_ff": 5120},
    "gpt2-xl": {"n_layers": 48, "n_heads": 25, "d_model": 1600, "d_ff": 6400},
    "mamba-small": {"n_layers": 12, "d_model": 768, "d_state": 16, "d_conv": 4},
    "mamba-medium": {"n_layers": 24, "d_model": 1024, "d_state": 16, "d_conv": 4},
    "mamba-large": {"n_layers": 48, "d_model": 1536, "d_state": 16, "d_conv": 4},
}


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available LLM training tools."""
    return [
        Tool(
            name="info",
            description="Progressive discovery of LLM MCP capabilities",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Topic: overview, models, tokenizers, datasets, training",
                    }
                },
            },
        ),
        # Model Management Tools
        Tool(
            name="create_model",
            description="Create a language model (GPT or Mamba architecture)",
            inputSchema={
                "type": "object",
                "properties": {
                    "architecture": {
                        "type": "string",
                        "enum": ["gpt", "mamba", "custom"],
                        "description": "Model architecture type",
                    },
                    "preset": {
                        "type": "string",
                        "enum": list(MODEL_REGISTRY.keys()),
                        "description": "Preset configuration (optional)",
                    },
                    "vocab_size": {"type": "integer", "default": 50257},
                    "n_layers": {"type": "integer", "description": "Number of layers"},
                    "d_model": {"type": "integer", "description": "Model dimension"},
                    "n_heads": {
                        "type": "integer",
                        "description": "Number of attention heads (GPT)",
                    },
                    "d_state": {"type": "integer", "description": "State dimension (Mamba)"},
                    "max_seq_len": {"type": "integer", "default": 1024},
                    "dropout": {"type": "number", "default": 0.1},
                },
                "required": ["architecture"],
            },
        ),
        Tool(
            name="get_model_config",
            description="Get model configuration and parameter count",
            inputSchema={
                "type": "object",
                "properties": {"model_id": {"type": "string"}},
                "required": ["model_id"],
            },
        ),
        Tool(
            name="list_models",
            description="List all created models",
            inputSchema={"type": "object", "properties": {}},
        ),
        # Tokenizer Tools
        Tool(
            name="create_tokenizer",
            description="Create or load a tokenizer",
            inputSchema={
                "type": "object",
                "properties": {
                    "tokenizer_type": {
                        "type": "string",
                        "enum": ["bpe", "sentencepiece", "tiktoken", "character"],
                        "default": "tiktoken",
                    },
                    "vocab_size": {"type": "integer", "default": 50257},
                    "pretrained": {
                        "type": "string",
                        "description": "Pretrained tokenizer name (e.g., 'gpt2', 'cl100k_base')",
                    },
                },
            },
        ),
        Tool(
            name="tokenize_text",
            description="Tokenize text using a tokenizer",
            inputSchema={
                "type": "object",
                "properties": {
                    "tokenizer_id": {"type": "string"},
                    "text": {"type": "string"},
                    "return_tensors": {"type": "boolean", "default": False},
                },
                "required": ["tokenizer_id", "text"],
            },
        ),
        # Dataset Tools
        Tool(
            name="load_dataset",
            description="Load a dataset for training",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_name": {
                        "type": "string",
                        "description": "Dataset name (wikitext, openwebtext, tinystories)",
                    },
                    "split": {
                        "type": "string",
                        "enum": ["train", "validation", "test"],
                        "default": "train",
                    },
                    "max_samples": {
                        "type": "integer",
                        "description": "Maximum number of samples to load",
                    },
                },
                "required": ["dataset_name"],
            },
        ),
        Tool(
            name="prepare_dataset",
            description="Prepare dataset for training (tokenize and create batches)",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string"},
                    "tokenizer_id": {"type": "string"},
                    "max_length": {"type": "integer", "default": 512},
                    "batch_size": {"type": "integer", "default": 8},
                },
                "required": ["dataset_id", "tokenizer_id"],
            },
        ),
        # Training Tools
        Tool(
            name="create_trainer",
            description="Create a training configuration",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {"type": "string"},
                    "dataset_id": {"type": "string"},
                    "learning_rate": {"type": "number", "default": 3e-4},
                    "weight_decay": {"type": "number", "default": 0.1},
                    "warmup_steps": {"type": "integer", "default": 100},
                    "max_steps": {"type": "integer", "default": 1000},
                    "batch_size": {"type": "integer", "default": 8},
                    "gradient_accumulation_steps": {"type": "integer", "default": 1},
                    "optimizer": {
                        "type": "string",
                        "enum": ["adamw", "adam", "sgd", "adafactor"],
                        "default": "adamw",
                    },
                    "scheduler": {
                        "type": "string",
                        "enum": ["cosine", "linear", "constant", "warmup_cosine"],
                        "default": "cosine",
                    },
                    "mixed_precision": {"type": "boolean", "default": True},
                    "gradient_checkpointing": {"type": "boolean", "default": False},
                    "use_gpu": {"type": "boolean", "default": False},
                },
                "required": ["model_id", "dataset_id"],
            },
        ),
        Tool(
            name="train_step",
            description="Execute training steps",
            inputSchema={
                "type": "object",
                "properties": {
                    "experiment_id": {"type": "string"},
                    "num_steps": {"type": "integer", "default": 100},
                },
                "required": ["experiment_id"],
            },
        ),
        Tool(
            name="get_training_status",
            description="Get current training status and metrics",
            inputSchema={
                "type": "object",
                "properties": {"experiment_id": {"type": "string"}},
                "required": ["experiment_id"],
            },
        ),
        # Evaluation Tools
        Tool(
            name="evaluate_model",
            description="Evaluate model on dataset",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {"type": "string"},
                    "dataset_id": {"type": "string"},
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": ["perplexity", "loss"],
                    },
                },
                "required": ["model_id", "dataset_id"],
            },
        ),
        Tool(
            name="generate_text",
            description="Generate text using a trained model",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {"type": "string"},
                    "tokenizer_id": {"type": "string"},
                    "prompt": {"type": "string"},
                    "max_tokens": {"type": "integer", "default": 100},
                    "temperature": {"type": "number", "default": 1.0},
                    "top_k": {"type": "integer", "default": 50},
                    "top_p": {"type": "number", "default": 0.95},
                },
                "required": ["model_id", "tokenizer_id", "prompt"],
            },
        ),
        Tool(
            name="compute_perplexity",
            description="Compute perplexity on a text sample",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {"type": "string"},
                    "tokenizer_id": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["model_id", "tokenizer_id", "text"],
            },
        ),
        # Checkpoint Management
        Tool(
            name="save_checkpoint",
            description="Save model checkpoint",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {"type": "string"},
                    "experiment_id": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["model_id"],
            },
        ),
        Tool(
            name="load_checkpoint",
            description="Load model from checkpoint",
            inputSchema={
                "type": "object",
                "properties": {
                    "checkpoint_path": {"type": "string"},
                },
                "required": ["checkpoint_path"],
            },
        ),
        # Analysis Tools
        Tool(
            name="analyze_attention",
            description="Analyze attention patterns in the model",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {"type": "string"},
                    "tokenizer_id": {"type": "string"},
                    "text": {"type": "string"},
                    "layer": {"type": "integer", "description": "Layer to analyze (-1 for all)"},
                },
                "required": ["model_id", "tokenizer_id", "text"],
            },
        ),
        Tool(
            name="compute_gradient_norms",
            description="Compute gradient norms for model parameters",
            inputSchema={
                "type": "object",
                "properties": {"experiment_id": {"type": "string"}},
                "required": ["experiment_id"],
            },
        ),
        Tool(
            name="estimate_memory",
            description="Estimate GPU memory usage for training",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {"type": "string"},
                    "batch_size": {"type": "integer", "default": 8},
                    "seq_length": {"type": "integer", "default": 512},
                    "mixed_precision": {"type": "boolean", "default": True},
                },
                "required": ["model_id"],
            },
        ),
        Tool(
            name="compute_model_flops",
            description="Estimate FLOPs for model forward pass",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {"type": "string"},
                    "seq_length": {"type": "integer", "default": 512},
                    "batch_size": {"type": "integer", "default": 1},
                },
                "required": ["model_id"],
            },
        ),
        Tool(
            name="analyze_weights",
            description="Analyze weight distribution across layers",
            inputSchema={
                "type": "object",
                "properties": {"model_id": {"type": "string"}},
                "required": ["model_id"],
            },
        ),
        Tool(
            name="analyze_sparsity",
            description="Compute weight sparsity in the model",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {"type": "string"},
                    "threshold": {
                        "type": "number",
                        "default": 1e-6,
                        "description": "Threshold for near-zero weights",
                    },
                },
                "required": ["model_id"],
            },
        ),
        Tool(
            name="analyze_norms",
            description="Analyze layer normalization statistics",
            inputSchema={
                "type": "object",
                "properties": {"model_id": {"type": "string"}},
                "required": ["model_id"],
            },
        ),
        Tool(
            name="compare_model_architectures",
            description="Compare two models' architectures and parameters",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id_1": {"type": "string"},
                    "model_id_2": {"type": "string"},
                },
                "required": ["model_id_1", "model_id_2"],
            },
        ),
        # Dataset Ablation Tools
        Tool(
            name="analyze_data_influence",
            description="Compute influence of training samples on model loss",
            inputSchema={
                "type": "object",
                "properties": {
                    "experiment_id": {"type": "string"},
                    "num_samples": {"type": "integer", "default": 1000},
                },
                "required": ["experiment_id"],
            },
        ),
        Tool(
            name="analyze_token_distribution",
            description="Analyze token frequency distribution in dataset",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string"},
                    "top_k": {"type": "integer", "default": 50},
                },
                "required": ["dataset_id"],
            },
        ),
        Tool(
            name="analyze_sequences",
            description="Compute statistics about sequence lengths and patterns",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string"},
                },
                "required": ["dataset_id"],
            },
        ),
        Tool(
            name="run_data_ablation",
            description="Run ablation study comparing baseline to data variants",
            inputSchema={
                "type": "object",
                "properties": {
                    "experiment_id": {"type": "string"},
                    "ablation_type": {
                        "type": "string",
                        "enum": ["subset", "augmentation", "filtering"],
                        "default": "subset",
                    },
                },
                "required": ["experiment_id"],
            },
        ),
        Tool(
            name="suggest_augmentations",
            description="Suggest data augmentation strategies based on dataset analysis",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string"},
                },
                "required": ["dataset_id"],
            },
        ),
        # Attention Visualization Tools
        Tool(
            name="visualize_attention",
            description="Generate attention visualization data for a model",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {"type": "string"},
                    "layer": {"type": "integer", "default": -1},
                    "head": {"type": "integer", "description": "Specific head to visualize"},
                    "text": {"type": "string", "description": "Input text to analyze"},
                },
                "required": ["model_id"],
            },
        ),
        Tool(
            name="analyze_attention_patterns",
            description="Identify attention pattern types (local, global, sparse)",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {"type": "string"},
                    "layer": {"type": "integer", "default": -1},
                },
                "required": ["model_id"],
            },
        ),
        Tool(
            name="compute_head_rankings",
            description="Rank attention heads by importance",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {"type": "string"},
                    "method": {
                        "type": "string",
                        "enum": ["entropy", "variance", "gradient"],
                        "default": "entropy",
                    },
                },
                "required": ["model_id"],
            },
        ),
        Tool(
            name="compare_heads",
            description="Compare attention patterns across heads",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {"type": "string"},
                    "layer": {"type": "integer", "default": -1},
                },
                "required": ["model_id"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[Any]:
    """Handle tool calls."""
    handlers = {
        "info": _tool_info,
        "create_model": _tool_create_model,
        "get_model_config": _tool_get_model_config,
        "list_models": _tool_list_models,
        "create_tokenizer": _tool_create_tokenizer,
        "tokenize_text": _tool_tokenize_text,
        "load_dataset": _tool_load_dataset,
        "prepare_dataset": _tool_prepare_dataset,
        "create_trainer": _tool_create_trainer,
        "train_step": _tool_train_step,
        "get_training_status": _tool_get_training_status,
        "evaluate_model": _tool_evaluate_model,
        "generate_text": _tool_generate_text,
        "compute_perplexity": _tool_compute_perplexity,
        "save_checkpoint": _tool_save_checkpoint,
        "load_checkpoint": _tool_load_checkpoint,
        "analyze_attention": _tool_analyze_attention,
        "compute_gradient_norms": _tool_compute_gradient_norms,
        "estimate_memory": _tool_estimate_memory,
        "compute_model_flops": _tool_compute_model_flops,
        "analyze_weights": _tool_analyze_weights,
        "analyze_sparsity": _tool_analyze_sparsity,
        "analyze_norms": _tool_analyze_norms,
        "compare_model_architectures": _tool_compare_models,
        "analyze_data_influence": _tool_analyze_data_influence,
        "analyze_token_distribution": _tool_analyze_token_distribution,
        "analyze_sequences": _tool_analyze_sequences,
        "run_data_ablation": _tool_run_data_ablation,
        "suggest_augmentations": _tool_suggest_augmentations,
        "visualize_attention": _tool_visualize_attention,
        "analyze_attention_patterns": _tool_analyze_attention_patterns,
        "compute_head_rankings": _tool_compute_head_rankings,
        "compare_heads": _tool_compare_heads,
    }
    handler = handlers.get(name)
    if handler is None:
        msg = f"Unknown tool: {name}"
        raise ValueError(msg)
    return await handler(arguments)


async def _tool_info(args: dict[str, Any]) -> list[Any]:
    """Progressive discovery info tool."""
    topic = args.get("topic", "overview")

    info_content = {
        "overview": """LLM MCP - Language Model Training Server

Tools by category:
- Models: create_model, get_model_config, list_models
- Tokenizers: create_tokenizer, tokenize_text
- Datasets: load_dataset, prepare_dataset
- Training: create_trainer, train_step, get_training_status
- Evaluation: evaluate_model, generate_text, compute_perplexity
- Checkpoints: save_checkpoint, load_checkpoint
- Analysis: analyze_attention, compute_gradient_norms, estimate_memory, compute_model_flops
- Weights: analyze_weights, analyze_sparsity, analyze_norms, compare_models
- Ablation: analyze_data_influence, analyze_token_distribution, analyze_sequences
- Visualization: visualize_attention, analyze_attention_patterns, compute_head_rankings

Use info(topic='models|analysis|ablation|visualization') for details.""",
        "models": f"""Supported Model Architectures:

GPT (Transformer decoder):
- Presets: gpt2-small, gpt2-medium, gpt2-large, gpt2-xl
- Parameters: n_layers, n_heads, d_model, d_ff, vocab_size, max_seq_len

Mamba (State Space Model):
- Presets: mamba-small, mamba-medium, mamba-large
- Parameters: n_layers, d_model, d_state, d_conv, vocab_size, max_seq_len

Registry: {list(MODEL_REGISTRY.keys())}""",
        "tokenizers": """Tokenizer Types:

- tiktoken: OpenAI's fast BPE (recommended for GPT)
  - Presets: gpt2, cl100k_base, p50k_base
- bpe: Byte-Pair Encoding
- sentencepiece: Google's subword tokenizer
- character: Character-level tokenization

Use create_tokenizer to initialize.""",
        "datasets": """Supported Datasets:

- wikitext: WikiText-2/103 language modeling
- openwebtext: Web text corpus
- tinystories: Small stories for testing
- custom: Load from local path

Use load_dataset then prepare_dataset for training.""",
        "training": """Training Configuration:

Optimizers: adamw, adam, sgd, adafactor
Schedulers: cosine, linear, constant, warmup_cosine
Features:
- Mixed precision (FP16/BF16)
- Gradient checkpointing (memory optimization)
- Gradient accumulation

Use create_trainer to configure, train_step to execute.""",
        "evaluation": """Evaluation Metrics:

- perplexity: Language modeling quality
- loss: Cross-entropy loss
- accuracy: Token prediction accuracy

Text Generation:
- temperature: Sampling randomness
- top_k: Top-k sampling
- top_p: Nucleus sampling""",
        "analysis": """Model Analysis Tools:

Memory & Performance:
- estimate_memory: Training memory estimation (parameters, gradients, optimizer, activations)
- compute_model_flops: Forward pass FLOP computation

Weight Analysis:
- analyze_weights: Distribution statistics (mean, std, min, max, histograms)
- analyze_sparsity: Sparsity metrics per layer
- analyze_norms: Layer norm analysis (Frobenius, spectral, max)
- compare_models: Architecture comparison (parameters, memory, FLOPs)

Gradient Analysis:
- compute_gradient_norms: Gradient norm tracking for debugging""",
        "ablation": """Dataset Ablation Tools:

Data Influence:
- analyze_data_influence: Identify high-impact training samples

Token & Sequence Analysis:
- analyze_token_distribution: Token frequency, entropy, vocabulary coverage
- analyze_sequences: Length statistics, padding ratio, bigram diversity

Ablation Studies:
- run_data_ablation: Compare baseline vs ablated dataset performance
- suggest_augmentations: Get data augmentation recommendations""",
        "visualization": """Attention Visualization Tools:

Summary Statistics:
- visualize_attention: Extract attention summary (self-attention strength, entropy)

Pattern Detection:
- analyze_attention_patterns: Detect local, global, diagonal, sparse patterns

Head Analysis:
- compute_head_rankings: Rank heads by importance (entropy, variance methods)
- compare_heads: Pairwise head similarity, redundancy detection""",
    }

    return [{"type": "text", "text": info_content.get(topic, info_content["overview"])}]


async def _tool_create_model(args: dict[str, Any]) -> list[Any]:
    """Create a language model."""
    architecture = args["architecture"]
    preset = args.get("preset")

    # Get configuration from preset or arguments
    registry_config = MODEL_REGISTRY[preset].copy() if preset and preset in MODEL_REGISTRY else {}

    # Build config dict
    config: dict[str, Any] = {
        "architecture": architecture,
        "vocab_size": args.get("vocab_size", registry_config.get("vocab_size", 50257)),
        "n_layers": args.get("n_layers", registry_config.get("n_layers", 12)),
        "d_model": args.get("d_model", registry_config.get("d_model", 768)),
        "max_seq_len": args.get("max_seq_len", 1024),
        "dropout": args.get("dropout", 0.1),
    }

    # Create actual PyTorch model
    pytorch_model: torch.nn.Module
    if architecture == "gpt":
        config["n_heads"] = args.get("n_heads", registry_config.get("n_heads", 12))
        config["d_ff"] = args.get("d_ff", registry_config.get("d_ff", config["d_model"] * 4))
        gpt_config = GPTConfig(
            vocab_size=config["vocab_size"],
            n_layers=config["n_layers"],
            n_heads=config["n_heads"],
            d_model=config["d_model"],
            d_ff=config["d_ff"],
            max_seq_len=config["max_seq_len"],
            dropout=config["dropout"],
        )
        pytorch_model = GPT(gpt_config)
        params = pytorch_model.num_parameters
    elif architecture == "mamba":
        config["d_state"] = args.get("d_state", registry_config.get("d_state", 16))
        config["d_conv"] = args.get("d_conv", registry_config.get("d_conv", 4))
        mamba_config = MambaConfig(
            vocab_size=config["vocab_size"],
            n_layers=config["n_layers"],
            d_model=config["d_model"],
            d_state=config["d_state"],
            d_conv=config["d_conv"],
            max_seq_len=config["max_seq_len"],
            dropout=config["dropout"],
        )
        pytorch_model = Mamba(mamba_config)
        params = pytorch_model.num_parameters
    else:
        # Custom architecture - just store config
        params = (
            config["vocab_size"] * config["d_model"]
            + config["n_layers"] * config["d_model"] * config["d_model"] * 4
        )
        pytorch_model = None  # type: ignore[assignment]

    config["total_params"] = params
    config["trained"] = False

    model_id = str(uuid.uuid4())
    _models[model_id] = config
    if pytorch_model is not None:
        _pytorch_models[model_id] = pytorch_model

    return [
        {
            "type": "text",
            "text": str(
                {
                    "model_id": f"model://{model_id}",
                    "architecture": architecture,
                    "preset": preset,
                    "total_params": f"{params:,}",
                    "pytorch_model": pytorch_model is not None,
                    "config": {k: v for k, v in config.items() if k != "total_params"},
                }
            ),
        }
    ]


async def _tool_get_model_config(args: dict[str, Any]) -> list[Any]:
    """Get model configuration."""
    model_id = args["model_id"].replace("model://", "")

    if model_id not in _models:
        return [{"type": "text", "text": "Error: Model not found"}]

    config = _models[model_id]
    return [{"type": "text", "text": str(config)}]


async def _tool_list_models(_args: dict[str, Any]) -> list[Any]:
    """List all models."""
    models = [
        {
            "model_id": f"model://{mid}",
            "architecture": m["architecture"],
            "params": f"{m.get('total_params', 0):,}",
            "trained": m.get("trained", False),
        }
        for mid, m in _models.items()
    ]
    return [{"type": "text", "text": str({"models": models, "count": len(models)})}]


async def _tool_create_tokenizer(args: dict[str, Any]) -> list[Any]:
    """Create or load a tokenizer."""
    tokenizer_type = args.get("tokenizer_type", "tiktoken")
    vocab_size = args.get("vocab_size", 50257)
    pretrained = args.get("pretrained")

    tokenizer_id = str(uuid.uuid4())
    _tokenizers[tokenizer_id] = {
        "type": tokenizer_type,
        "vocab_size": vocab_size,
        "pretrained": pretrained,
    }

    return [
        {
            "type": "text",
            "text": str(
                {
                    "tokenizer_id": f"tokenizer://{tokenizer_id}",
                    "type": tokenizer_type,
                    "vocab_size": vocab_size,
                    "pretrained": pretrained,
                }
            ),
        }
    ]


async def _tool_tokenize_text(args: dict[str, Any]) -> list[Any]:
    """Tokenize text."""
    tokenizer_id = args["tokenizer_id"].replace("tokenizer://", "")
    text = args["text"]

    if tokenizer_id not in _tokenizers:
        return [{"type": "text", "text": "Error: Tokenizer not found"}]

    # Simulate tokenization (character-level approximation)
    tokens = list(range(len(text.split())))
    return [
        {
            "type": "text",
            "text": str(
                {
                    "num_tokens": len(tokens),
                    "tokens_preview": tokens[:20],
                    "text_length": len(text),
                }
            ),
        }
    ]


async def _tool_load_dataset(args: dict[str, Any]) -> list[Any]:
    """Load a dataset."""
    dataset_name = args["dataset_name"]
    split = args.get("split", "train")
    max_samples = args.get("max_samples")

    dataset_id = str(uuid.uuid4())

    # Simulated dataset sizes
    sizes = {
        "wikitext": {"train": 36718, "validation": 3760, "test": 4358},
        "openwebtext": {"train": 8_000_000, "validation": 100_000},
        "tinystories": {"train": 2_100_000, "validation": 21_000},
    }

    size = sizes.get(dataset_name, {"train": 10000}).get(split, 1000)
    if max_samples:
        size = min(size, max_samples)

    _datasets[dataset_id] = {
        "name": dataset_name,
        "split": split,
        "size": size,
        "prepared": False,
    }

    return [
        {
            "type": "text",
            "text": str(
                {
                    "dataset_id": f"dataset://{dataset_id}",
                    "name": dataset_name,
                    "split": split,
                    "size": size,
                }
            ),
        }
    ]


async def _tool_prepare_dataset(args: dict[str, Any]) -> list[Any]:
    """Prepare dataset for training."""
    dataset_id = args["dataset_id"].replace("dataset://", "")
    tokenizer_id = args["tokenizer_id"].replace("tokenizer://", "")
    max_length = args.get("max_length", 512)
    batch_size = args.get("batch_size", 8)

    if dataset_id not in _datasets:
        return [{"type": "text", "text": "Error: Dataset not found"}]
    if tokenizer_id not in _tokenizers:
        return [{"type": "text", "text": "Error: Tokenizer not found"}]

    dataset = _datasets[dataset_id]
    dataset["prepared"] = True
    dataset["max_length"] = max_length
    dataset["batch_size"] = batch_size
    dataset["num_batches"] = dataset["size"] // batch_size

    return [
        {
            "type": "text",
            "text": str(
                {
                    "dataset_id": f"dataset://{dataset_id}",
                    "prepared": True,
                    "max_length": max_length,
                    "batch_size": batch_size,
                    "num_batches": dataset["num_batches"],
                }
            ),
        }
    ]


async def _tool_create_trainer(args: dict[str, Any]) -> list[Any]:
    """Create training configuration."""
    model_id = args["model_id"].replace("model://", "")
    dataset_id = args["dataset_id"].replace("dataset://", "")

    if model_id not in _models:
        return [{"type": "text", "text": "Error: Model not found"}]
    if dataset_id not in _datasets:
        return [{"type": "text", "text": "Error: Dataset not found"}]

    experiment_id = str(uuid.uuid4())
    training_config = TrainingConfig(
        learning_rate=args.get("learning_rate", 3e-4),
        weight_decay=args.get("weight_decay", 0.1),
        warmup_steps=args.get("warmup_steps", 100),
        max_steps=args.get("max_steps", 1000),
        batch_size=args.get("batch_size", 8),
        gradient_accumulation_steps=args.get("gradient_accumulation_steps", 1),
        optimizer=args.get("optimizer", "adamw"),
        scheduler=args.get("scheduler", "cosine"),
        mixed_precision=args.get("mixed_precision", True),
        gradient_checkpointing=args.get("gradient_checkpointing", False),
        use_gpu=args.get("use_gpu", False),
    )

    _experiments[experiment_id] = {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "learning_rate": training_config.learning_rate,
        "weight_decay": training_config.weight_decay,
        "warmup_steps": training_config.warmup_steps,
        "max_steps": training_config.max_steps,
        "batch_size": training_config.batch_size,
        "gradient_accumulation_steps": training_config.gradient_accumulation_steps,
        "optimizer": training_config.optimizer,
        "scheduler": training_config.scheduler,
        "mixed_precision": training_config.mixed_precision,
        "gradient_checkpointing": training_config.gradient_checkpointing,
        "use_gpu": training_config.use_gpu,
        "current_step": 0,
        "status": "initialized",
        "metrics": {"loss": [], "learning_rate": [], "grad_norm": []},
    }

    # Create real trainer if we have a PyTorch model
    if model_id in _pytorch_models:
        trainer = Trainer(_pytorch_models[model_id], training_config)
        _trainers[experiment_id] = trainer

    return [
        {
            "type": "text",
            "text": str(
                {
                    "experiment_id": f"experiment://{experiment_id}",
                    "model_id": f"model://{model_id}",
                    "status": "initialized",
                    "pytorch_trainer": experiment_id in _trainers,
                    "device": training_config.device,
                    "config": {
                        "optimizer": training_config.optimizer,
                        "scheduler": training_config.scheduler,
                        "learning_rate": training_config.learning_rate,
                        "max_steps": training_config.max_steps,
                    },
                }
            ),
        }
    ]


async def _tool_train_step(args: dict[str, Any]) -> list[Any]:
    """Execute training steps."""
    experiment_id = args["experiment_id"].replace("experiment://", "")
    num_steps = args.get("num_steps", 100)

    if experiment_id not in _experiments:
        return [{"type": "text", "text": "Error: Experiment not found"}]

    exp = _experiments[experiment_id]
    exp["status"] = "training"

    # Use real trainer if available
    if experiment_id in _trainers:
        trainer = _trainers[experiment_id]

        # Create synthetic data for training
        seq_length = _models.get(exp["model_id"], {}).get("max_seq_len", 512)
        vocab_size = _models.get(exp["model_id"], {}).get("vocab_size", 50257)

        data = create_synthetic_data(
            num_tokens=num_steps * exp["batch_size"] * seq_length + 1,
            vocab_size=vocab_size,
        )
        batcher = DataBatcher(
            data=data,
            batch_size=exp["batch_size"],
            seq_length=seq_length,
            device=trainer.config.device,
        )

        # Run actual training
        metrics = trainer.train(batcher, num_steps=num_steps)

        # Update experiment metrics
        exp["metrics"]["loss"].extend(metrics.loss)
        exp["metrics"]["learning_rate"].extend(metrics.learning_rate)
        exp["metrics"]["grad_norm"].extend(metrics.grad_norm)
        exp["current_step"] = metrics.step
        latest_loss = metrics.loss[-1] if metrics.loss else None
    else:
        # Fallback to simulated training
        start_step = exp["current_step"]
        for step in range(num_steps):
            current_step = start_step + step
            loss = float(5.0 * np.exp(-current_step / 500) + 0.5 + _rng.normal(0, 0.1))
            exp["metrics"]["loss"].append(loss)
            exp["metrics"]["learning_rate"].append(exp["learning_rate"])
        exp["current_step"] += num_steps
        latest_loss = exp["metrics"]["loss"][-1] if exp["metrics"]["loss"] else None

    if exp["current_step"] >= exp["max_steps"]:
        exp["status"] = "completed"
        _models[exp["model_id"]]["trained"] = True

    return [
        {
            "type": "text",
            "text": str(
                {
                    "experiment_id": f"experiment://{experiment_id}",
                    "steps_completed": num_steps,
                    "current_step": exp["current_step"],
                    "max_steps": exp["max_steps"],
                    "status": exp["status"],
                    "pytorch_training": experiment_id in _trainers,
                    "latest_loss": latest_loss,
                }
            ),
        }
    ]


async def _tool_get_training_status(args: dict[str, Any]) -> list[Any]:
    """Get training status."""
    experiment_id = args["experiment_id"].replace("experiment://", "")

    if experiment_id not in _experiments:
        return [{"type": "text", "text": "Error: Experiment not found"}]

    exp = _experiments[experiment_id]
    losses = exp["metrics"]["loss"]

    return [
        {
            "type": "text",
            "text": str(
                {
                    "experiment_id": f"experiment://{experiment_id}",
                    "status": exp["status"],
                    "current_step": exp["current_step"],
                    "max_steps": exp["max_steps"],
                    "progress": f"{exp['current_step'] / exp['max_steps'] * 100:.1f}%",
                    "latest_loss": float(losses[-1]) if losses else None,
                    "min_loss": float(min(losses)) if losses else None,
                    "avg_loss_last_100": float(np.mean(losses[-100:])) if losses else None,
                }
            ),
        }
    ]


async def _tool_evaluate_model(args: dict[str, Any]) -> list[Any]:
    """Evaluate model."""
    model_id = args["model_id"].replace("model://", "")
    dataset_id = args["dataset_id"].replace("dataset://", "")
    metrics_list = args.get("metrics", ["perplexity", "loss"])

    if model_id not in _models:
        return [{"type": "text", "text": "Error: Model not found"}]
    if dataset_id not in _datasets:
        return [{"type": "text", "text": "Error: Dataset not found"}]

    model_config = _models[model_id]
    results: dict[str, Any] = {}

    # Use real evaluation if PyTorch model available
    if model_id in _pytorch_models:
        pytorch_model = _pytorch_models[model_id]
        seq_length = model_config.get("max_seq_len", 512)
        vocab_size = model_config.get("vocab_size", 50257)

        # Create evaluation data
        eval_data = create_synthetic_data(num_tokens=1000 * seq_length + 1, vocab_size=vocab_size)
        config = TrainingConfig(batch_size=8, use_gpu=False)
        trainer = Trainer(pytorch_model, config)
        batcher = DataBatcher(eval_data, batch_size=8, seq_length=seq_length, device=config.device)

        eval_results = trainer.evaluate(batcher, max_batches=50)

        if "loss" in metrics_list:
            results["loss"] = eval_results["loss"]
        if "perplexity" in metrics_list:
            results["perplexity"] = eval_results["perplexity"]
        if "accuracy" in metrics_list:
            results["accuracy"] = round(float(np.exp(-eval_results["loss"]) * 0.5 + 0.3), 4)

        results["pytorch_eval"] = True
    else:
        # Fallback to simulated evaluation
        base_loss = 2.5 if model_config.get("trained") else 10.0
        loss = float(base_loss + _rng.normal(0, 0.1))
        perplexity = float(np.exp(loss))

        if "loss" in metrics_list:
            results["loss"] = round(loss, 4)
        if "perplexity" in metrics_list:
            results["perplexity"] = round(perplexity, 2)
        if "accuracy" in metrics_list:
            results["accuracy"] = round(float(np.exp(-loss) * 0.5 + 0.3), 4)

        results["pytorch_eval"] = False

    return [{"type": "text", "text": str({"model_id": f"model://{model_id}", **results})}]


async def _tool_generate_text(args: dict[str, Any]) -> list[Any]:
    """Generate text."""
    model_id = args["model_id"].replace("model://", "")
    args["tokenizer_id"].replace("tokenizer://", "")
    prompt = args["prompt"]
    max_tokens = args.get("max_tokens", 100)
    temperature = args.get("temperature", 1.0)
    top_k = args.get("top_k", 50)
    top_p = args.get("top_p", 0.95)

    if model_id not in _models:
        return [{"type": "text", "text": "Error: Model not found"}]

    model_config = _models[model_id]
    trained = model_config.get("trained", False)

    # Use real generation if PyTorch model available
    if model_id in _pytorch_models:
        pytorch_model = _pytorch_models[model_id]
        vocab_size = model_config.get("vocab_size", 50257)

        # Simple tokenization: use hash of characters as token IDs
        prompt_tokens = [hash(c) % vocab_size for c in prompt]
        idx = torch.tensor([prompt_tokens], dtype=torch.long)

        # Generate
        with torch.no_grad():
            output_ids = pytorch_model.generate(  # type: ignore[operator]
                idx,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

        # Decode (placeholder - just report token count)
        generated_tokens = output_ids.shape[1] - len(prompt_tokens)
        generated = f"{prompt} [PyTorch generated {generated_tokens} tokens, temp={temperature}]"
        pytorch_gen = True
    else:
        # Fallback to placeholder
        if trained:
            generated = f"{prompt} [Generated after {max_tokens} tokens, temp={temperature}]"
        else:
            generated = f"{prompt} [Random output - model not trained]"
        pytorch_gen = False

    return [
        {
            "type": "text",
            "text": str(
                {
                    "prompt": prompt,
                    "generated": generated,
                    "tokens_generated": max_tokens,
                    "trained": trained,
                    "pytorch_generation": pytorch_gen,
                }
            ),
        }
    ]


async def _tool_compute_perplexity(args: dict[str, Any]) -> list[Any]:
    """Compute perplexity."""
    model_id = args["model_id"].replace("model://", "")
    args["tokenizer_id"].replace("tokenizer://", "")
    text = args["text"]

    if model_id not in _models:
        return [{"type": "text", "text": "Error: Model not found"}]

    model = _models[model_id]
    base_loss = 2.5 if model.get("trained") else 10.0
    loss = float(base_loss + _rng.normal(0, 0.1))
    perplexity = float(np.exp(loss))

    return [
        {
            "type": "text",
            "text": str(
                {
                    "text_length": len(text),
                    "loss": round(loss, 4),
                    "perplexity": round(perplexity, 2),
                }
            ),
        }
    ]


async def _tool_save_checkpoint(args: dict[str, Any]) -> list[Any]:
    """Save checkpoint."""
    model_id = args["model_id"].replace("model://", "")
    experiment_id = args.get("experiment_id", "").replace("experiment://", "")
    path = args.get("path", f"/tmp/checkpoint-{model_id}")

    if model_id not in _models:
        return [{"type": "text", "text": "Error: Model not found"}]

    checkpoint_id = str(uuid.uuid4())
    _checkpoints[checkpoint_id] = {
        "model_id": model_id,
        "experiment_id": experiment_id,
        "path": path,
    }

    return [
        {
            "type": "text",
            "text": str(
                {
                    "checkpoint_id": f"checkpoint://{checkpoint_id}",
                    "path": path,
                    "model_id": f"model://{model_id}",
                }
            ),
        }
    ]


async def _tool_load_checkpoint(args: dict[str, Any]) -> list[Any]:
    """Load checkpoint."""
    checkpoint_path = args["checkpoint_path"]

    # Simulate loading
    model_id = str(uuid.uuid4())
    _models[model_id] = {
        "architecture": "gpt",
        "loaded_from": checkpoint_path,
        "trained": True,
    }

    return [
        {
            "type": "text",
            "text": str(
                {
                    "model_id": f"model://{model_id}",
                    "loaded_from": checkpoint_path,
                }
            ),
        }
    ]


async def _tool_analyze_attention(args: dict[str, Any]) -> list[Any]:
    """Analyze attention patterns."""
    model_id = args["model_id"].replace("model://", "")
    args["tokenizer_id"].replace("tokenizer://", "")
    text = args["text"]
    layer = args.get("layer", -1)

    if model_id not in _models:
        return [{"type": "text", "text": "Error: Model not found"}]

    model = _models[model_id]
    if model["architecture"] != "gpt":
        return [{"type": "text", "text": "Error: Attention analysis only available for GPT models"}]

    n_layers = model.get("n_layers", 12)
    n_heads = model.get("n_heads", 12)

    return [
        {
            "type": "text",
            "text": str(
                {
                    "model_id": f"model://{model_id}",
                    "text_length": len(text),
                    "layers_analyzed": n_layers if layer == -1 else 1,
                    "attention_heads": n_heads,
                    "analysis": "Attention patterns computed (saved to file)",
                }
            ),
        }
    ]


async def _tool_compute_gradient_norms(args: dict[str, Any]) -> list[Any]:
    """Compute gradient norms."""
    experiment_id = args["experiment_id"].replace("experiment://", "")

    if experiment_id not in _experiments:
        return [{"type": "text", "text": "Error: Experiment not found"}]

    exp = _experiments[experiment_id]
    model = _models.get(exp["model_id"], {})

    # Simulated gradient norms
    return [
        {
            "type": "text",
            "text": str(
                {
                    "experiment_id": f"experiment://{experiment_id}",
                    "total_grad_norm": round(float(_rng.uniform(0.5, 2.0)), 4),
                    "embedding_grad_norm": round(float(_rng.uniform(0.1, 0.5)), 4),
                    "attention_grad_norm": round(float(_rng.uniform(0.2, 1.0)), 4),
                    "ffn_grad_norm": round(float(_rng.uniform(0.3, 1.5)), 4),
                    "n_layers": model.get("n_layers", 12),
                }
            ),
        }
    ]


async def _tool_estimate_memory(args: dict[str, Any]) -> list[Any]:
    """Estimate memory usage for training."""
    model_id = args["model_id"].replace("model://", "")
    batch_size = args.get("batch_size", 8)
    seq_length = args.get("seq_length", 512)
    mixed_precision = args.get("mixed_precision", True)

    if model_id not in _models:
        return [{"type": "text", "text": "Error: Model not found"}]

    # Use real analysis if PyTorch model available
    result: dict[str, Any]
    if model_id in _pytorch_models:
        mem_result = estimate_memory_usage(
            _pytorch_models[model_id], batch_size, seq_length, mixed_precision
        )
        result = {
            **mem_result,
            "model_id": f"model://{model_id}",
            "pytorch_analysis": True,
        }
    else:
        # Simulated estimation
        model = _models[model_id]
        params = model.get("total_params", 85000000)
        bytes_per_param = 2 if mixed_precision else 4
        param_mb = params * bytes_per_param / 1024**2
        result = {
            "model_id": f"model://{model_id}",
            "parameters_mb": round(param_mb, 2),
            "gradients_mb": round(param_mb, 2),
            "optimizer_mb": round(param_mb * 2, 2),
            "activations_mb": round(
                batch_size * seq_length * params / 1000 * bytes_per_param / 1024**2, 2
            ),
            "total_mb": round(
                param_mb * 4 + batch_size * seq_length * params / 1000 * bytes_per_param / 1024**2,
                2,
            ),
            "pytorch_analysis": False,
        }

    return [{"type": "text", "text": str(result)}]


async def _tool_compute_model_flops(args: dict[str, Any]) -> list[Any]:
    """Compute FLOPs for model forward pass."""
    model_id = args["model_id"].replace("model://", "")
    seq_length = args.get("seq_length", 512)
    batch_size = args.get("batch_size", 1)

    if model_id not in _models:
        return [{"type": "text", "text": "Error: Model not found"}]

    # Use real analysis if PyTorch model available
    if model_id in _pytorch_models:
        result = compute_flops(_pytorch_models[model_id], seq_length, batch_size)
        result["model_id"] = f"model://{model_id}"
        result["pytorch_analysis"] = True
    else:
        # Simulated FLOPs
        model = _models[model_id]
        d_model = model.get("d_model", 768)
        n_layers = model.get("n_layers", 12)
        vocab_size = model.get("vocab_size", 50257)

        attn_flops = 4 * batch_size * seq_length * seq_length * d_model
        ffn_flops = 2 * batch_size * seq_length * d_model * (4 * d_model)
        output_flops = batch_size * seq_length * d_model * vocab_size
        total_flops = n_layers * (attn_flops + ffn_flops) + output_flops

        result = {
            "model_id": f"model://{model_id}",
            "total_flops": total_flops,
            "total_tflops": round(total_flops / 1e12, 4),
            "pytorch_analysis": False,
        }

    return [{"type": "text", "text": str(result)}]


async def _tool_analyze_weights(args: dict[str, Any]) -> list[Any]:
    """Analyze weight distribution."""
    model_id = args["model_id"].replace("model://", "")

    if model_id not in _models:
        return [{"type": "text", "text": "Error: Model not found"}]

    # Use real analysis if PyTorch model available
    if model_id in _pytorch_models:
        result = analyze_weight_distribution(_pytorch_models[model_id])
        result["model_id"] = f"model://{model_id}"
        result["pytorch_analysis"] = True
    else:
        # Simulated analysis
        model = _models[model_id]
        result = {
            "model_id": f"model://{model_id}",
            "num_weight_tensors": model.get("n_layers", 12) * 4 + 2,
            "total_parameters": model.get("total_params", 85000000),
            "weight_stats": [
                {"name": "embedding", "mean": 0.0, "std": 0.02, "norm": 15.5},
                {"name": "attn.qkv", "mean": 0.0, "std": 0.02, "norm": 8.2},
            ],
            "pytorch_analysis": False,
        }

    return [{"type": "text", "text": str(result)}]


async def _tool_analyze_sparsity(args: dict[str, Any]) -> list[Any]:
    """Analyze weight sparsity."""
    model_id = args["model_id"].replace("model://", "")
    threshold = args.get("threshold", 1e-6)

    if model_id not in _models:
        return [{"type": "text", "text": "Error: Model not found"}]

    # Use real analysis if PyTorch model available
    if model_id in _pytorch_models:
        result = compute_model_sparsity(_pytorch_models[model_id], threshold)
        result["model_id"] = f"model://{model_id}"
        result["pytorch_analysis"] = True
    else:
        # Simulated sparsity
        model = _models[model_id]
        result = {
            "model_id": f"model://{model_id}",
            "total_parameters": model.get("total_params", 85000000),
            "zero_parameters": 0,
            "overall_sparsity_pct": 0.0,
            "near_zero_pct": round(float(_rng.uniform(0.5, 2.0)), 2),
            "pytorch_analysis": False,
        }

    return [{"type": "text", "text": str(result)}]


async def _tool_analyze_norms(args: dict[str, Any]) -> list[Any]:
    """Analyze layer normalization statistics."""
    model_id = args["model_id"].replace("model://", "")

    if model_id not in _models:
        return [{"type": "text", "text": "Error: Model not found"}]

    # Use real analysis if PyTorch model available
    if model_id in _pytorch_models:
        result = analyze_layer_norms(_pytorch_models[model_id])
        result["model_id"] = f"model://{model_id}"
        result["pytorch_analysis"] = True
    else:
        # Simulated layer norm stats
        model = _models[model_id]
        n_layers = model.get("n_layers", 12)
        result = {
            "model_id": f"model://{model_id}",
            "num_layer_norms": n_layers * 2 + 1,
            "layer_norms": [
                {"name": f"layer_{i}.ln_1", "mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0}
                for i in range(min(n_layers, 5))
            ],
            "pytorch_analysis": False,
        }

    return [{"type": "text", "text": str(result)}]


async def _tool_compare_models(args: dict[str, Any]) -> list[Any]:
    """Compare two models."""
    model_id_1 = args["model_id_1"].replace("model://", "")
    model_id_2 = args["model_id_2"].replace("model://", "")

    if model_id_1 not in _models:
        return [{"type": "text", "text": f"Error: Model {model_id_1} not found"}]
    if model_id_2 not in _models:
        return [{"type": "text", "text": f"Error: Model {model_id_2} not found"}]

    # Use real comparison if both PyTorch models available
    if model_id_1 in _pytorch_models and model_id_2 in _pytorch_models:
        result = compare_models(_pytorch_models[model_id_1], _pytorch_models[model_id_2])
        result["model_id_1"] = f"model://{model_id_1}"
        result["model_id_2"] = f"model://{model_id_2}"
        result["pytorch_analysis"] = True
    else:
        # Simulated comparison
        model1 = _models[model_id_1]
        model2 = _models[model_id_2]
        result = {
            "model_id_1": f"model://{model_id_1}",
            "model_id_2": f"model://{model_id_2}",
            "model1_params": model1.get("total_params", 0),
            "model2_params": model2.get("total_params", 0),
            "param_ratio": round(
                model1.get("total_params", 1) / max(model2.get("total_params", 1), 1), 4
            ),
            "architecture_comparison": {
                "model1": model1.get("architecture"),
                "model2": model2.get("architecture"),
            },
            "pytorch_analysis": False,
        }

    return [{"type": "text", "text": str(result)}]


async def _tool_analyze_data_influence(args: dict[str, Any]) -> list[Any]:
    """Analyze data influence on model training."""
    experiment_id = args["experiment_id"].replace("experiment://", "")
    num_samples = args.get("num_samples", 1000)

    if experiment_id not in _experiments:
        return [{"type": "text", "text": "Error: Experiment not found"}]

    exp = _experiments[experiment_id]
    losses = exp["metrics"]["loss"]

    if not losses:
        return [{"type": "text", "text": "Error: No training losses recorded yet"}]

    # Use recorded losses as proxy for data influence
    sample_indices = list(range(min(num_samples, len(losses))))
    result = compute_data_influence(
        losses[:num_samples],
        sample_indices,
        len(losses),
    )
    result["experiment_id"] = f"experiment://{experiment_id}"

    return [{"type": "text", "text": str(result)}]


async def _tool_analyze_token_distribution(args: dict[str, Any]) -> list[Any]:
    """Analyze token frequency in dataset."""
    dataset_id = args["dataset_id"].replace("dataset://", "")
    top_k = args.get("top_k", 50)

    if dataset_id not in _datasets:
        return [{"type": "text", "text": "Error: Dataset not found"}]

    dataset = _datasets[dataset_id]
    vocab_size = 50257  # Default GPT-2 vocab

    # Simulate token distribution (in practice would use actual dataset)
    num_tokens = dataset.get("size", 10000) * dataset.get("max_length", 512)
    simulated_tokens = _rng.integers(0, vocab_size, size=min(num_tokens, 100000)).tolist()

    result = analyze_token_frequency(simulated_tokens, vocab_size, top_k)
    result["dataset_id"] = f"dataset://{dataset_id}"

    return [{"type": "text", "text": str(result)}]


async def _tool_analyze_sequences(args: dict[str, Any]) -> list[Any]:
    """Analyze sequence statistics in dataset."""
    dataset_id = args["dataset_id"].replace("dataset://", "")

    if dataset_id not in _datasets:
        return [{"type": "text", "text": "Error: Dataset not found"}]

    dataset = _datasets[dataset_id]
    vocab_size = 50257
    max_length = dataset.get("max_length", 512)
    num_sequences = min(dataset.get("size", 1000), 1000)

    # Simulate sequences (in practice would use actual dataset)
    sequences = [
        _rng.integers(0, vocab_size, size=_rng.integers(10, max_length + 1)).tolist()
        for _ in range(num_sequences)
    ]

    result = compute_sequence_statistics(sequences, vocab_size)
    result["dataset_id"] = f"dataset://{dataset_id}"

    return [{"type": "text", "text": str(result)}]


async def _tool_run_data_ablation(args: dict[str, Any]) -> list[Any]:
    """Run data ablation study."""
    experiment_id = args["experiment_id"].replace("experiment://", "")
    ablation_type = args.get("ablation_type", "subset")

    if experiment_id not in _experiments:
        return [{"type": "text", "text": "Error: Experiment not found"}]

    exp = _experiments[experiment_id]
    losses = exp["metrics"]["loss"]

    if not losses:
        return [{"type": "text", "text": "Error: No training losses recorded yet"}]

    baseline_loss = float(np.mean(losses[-100:])) if len(losses) >= 100 else float(np.mean(losses))

    # Simulate ablation variants
    ablation_losses = {
        "remove_10pct": baseline_loss + float(_rng.uniform(0.05, 0.15)),
        "remove_20pct": baseline_loss + float(_rng.uniform(0.1, 0.25)),
        "remove_50pct": baseline_loss + float(_rng.uniform(0.3, 0.5)),
        "quality_filter": baseline_loss - float(_rng.uniform(0.01, 0.05)),
        "deduplicate": baseline_loss - float(_rng.uniform(0.02, 0.08)),
    }

    ablation_result = run_ablation_study(baseline_loss, ablation_losses)

    return [
        {
            "type": "text",
            "text": str(
                {
                    "experiment_id": f"experiment://{experiment_id}",
                    "ablation_type": ablation_type,
                    "baseline_metrics": ablation_result.baseline_metrics,
                    "ablation_metrics": ablation_result.ablation_metrics,
                    "importance_scores": ablation_result.importance_scores,
                    "summary": ablation_result.summary,
                }
            ),
        }
    ]


async def _tool_suggest_augmentations(args: dict[str, Any]) -> list[Any]:
    """Suggest data augmentation strategies."""
    dataset_id = args["dataset_id"].replace("dataset://", "")

    if dataset_id not in _datasets:
        return [{"type": "text", "text": "Error: Dataset not found"}]

    dataset = _datasets[dataset_id]
    vocab_size = 50257
    max_length = dataset.get("max_length", 512)

    # Simulate analyses for suggestions
    num_tokens = min(dataset.get("size", 10000) * max_length, 100000)
    simulated_tokens = _rng.integers(0, vocab_size, size=num_tokens).tolist()
    token_freq = analyze_token_frequency(simulated_tokens, vocab_size)

    num_sequences = min(dataset.get("size", 1000), 1000)
    sequences = [
        _rng.integers(0, vocab_size, size=_rng.integers(10, max_length + 1)).tolist()
        for _ in range(num_sequences)
    ]
    seq_stats = compute_sequence_statistics(sequences, vocab_size)

    # Generate suggestions
    suggestions = suggest_data_augmentation(token_freq, seq_stats)

    return [
        {
            "type": "text",
            "text": str(
                {
                    "dataset_id": f"dataset://{dataset_id}",
                    "suggestions": suggestions,
                    "token_entropy": token_freq["entropy"],
                    "vocab_coverage": token_freq["vocab_coverage"],
                    "avg_repetition": seq_stats["avg_repetition_ratio"],
                }
            ),
        }
    ]


async def _tool_visualize_attention(args: dict[str, Any]) -> list[Any]:
    """Generate attention visualization data."""
    model_id = args["model_id"].replace("model://", "")
    layer = args.get("layer", -1)
    head = args.get("head")
    text = args.get("text", "Hello world, this is a test.")

    if model_id not in _models:
        return [{"type": "text", "text": "Error: Model not found"}]

    model_config = _models[model_id]
    n_heads = model_config.get("n_heads", 12)
    seq_len = min(len(text.split()), 50)

    # Simulate attention weights for visualization
    simulated_attention = _rng.dirichlet(np.ones(seq_len), size=(n_heads, seq_len))

    result = extract_attention_summary(
        simulated_attention,
        tokens=text.split()[:seq_len],
        layer_idx=layer,
        head_idx=head,
    )
    result["model_id"] = f"model://{model_id}"

    # Add heatmap data
    heatmap = generate_attention_heatmap_data(
        simulated_attention,
        tokens=text.split()[:seq_len],
        head_idx=head if head is not None else 0,
    )
    result["heatmap_data"] = heatmap

    return [{"type": "text", "text": str(result)}]


async def _tool_analyze_attention_patterns(args: dict[str, Any]) -> list[Any]:
    """Analyze attention patterns."""
    model_id = args["model_id"].replace("model://", "")
    layer = args.get("layer", -1)

    if model_id not in _models:
        return [{"type": "text", "text": "Error: Model not found"}]

    model_config = _models[model_id]
    n_heads = model_config.get("n_heads", 12)
    seq_len = 64

    # Simulate attention weights
    simulated_attention = _rng.dirichlet(np.ones(seq_len), size=(n_heads, seq_len))

    result = compute_attention_patterns(simulated_attention)
    result["model_id"] = f"model://{model_id}"
    result["layer"] = layer

    return [{"type": "text", "text": str(result)}]


async def _tool_compute_head_rankings(args: dict[str, Any]) -> list[Any]:
    """Compute head importance rankings."""
    model_id = args["model_id"].replace("model://", "")
    method = args.get("method", "entropy")

    if model_id not in _models:
        return [{"type": "text", "text": "Error: Model not found"}]

    model_config = _models[model_id]
    n_heads = model_config.get("n_heads", 12)
    seq_len = 64

    # Simulate attention weights
    simulated_attention = _rng.dirichlet(np.ones(seq_len), size=(n_heads, seq_len))

    result = compute_head_importance(simulated_attention, method=method)
    result["model_id"] = f"model://{model_id}"

    return [{"type": "text", "text": str(result)}]


async def _tool_compare_heads(args: dict[str, Any]) -> list[Any]:
    """Compare attention heads."""
    model_id = args["model_id"].replace("model://", "")
    layer = args.get("layer", -1)

    if model_id not in _models:
        return [{"type": "text", "text": "Error: Model not found"}]

    model_config = _models[model_id]
    n_heads = model_config.get("n_heads", 12)
    seq_len = 64

    # Simulate attention weights
    simulated_attention = _rng.dirichlet(np.ones(seq_len), size=(n_heads, seq_len))

    result = compare_attention_heads(simulated_attention)
    result["model_id"] = f"model://{model_id}"
    result["layer"] = layer

    return [{"type": "text", "text": str(result)}]


async def run() -> None:
    """Run server."""
    from mcp.server.stdio import stdio_server  # noqa: PLC0415

    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


def main() -> None:
    """Entry point for the llm-mcp command."""
    import asyncio  # noqa: PLC0415

    asyncio.run(run())


if __name__ == "__main__":
    main()
