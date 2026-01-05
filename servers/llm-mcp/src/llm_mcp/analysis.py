"""Analysis tools for LLM models."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def estimate_memory_usage(
    model: nn.Module,
    batch_size: int,
    seq_length: int,
    mixed_precision: bool = True,
) -> dict[str, float]:
    """Estimate memory usage for training."""
    # Parameter memory
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    # Gradient memory (same as parameters)
    grad_bytes = param_bytes

    # Optimizer state (AdamW: 2 states per parameter)
    optimizer_bytes = param_bytes * 2

    # Activation memory (rough estimate based on model config)
    n_params = sum(p.numel() for p in model.parameters())
    bytes_per_param = 2 if mixed_precision else 4
    # Rough estimate: activations scale with batch_size * seq_length
    activation_bytes = batch_size * seq_length * (n_params // 1000) * bytes_per_param

    total_bytes = param_bytes + grad_bytes + optimizer_bytes + activation_bytes

    return {
        "parameters_mb": round(param_bytes / 1024**2, 2),
        "gradients_mb": round(grad_bytes / 1024**2, 2),
        "optimizer_mb": round(optimizer_bytes / 1024**2, 2),
        "activations_mb": round(activation_bytes / 1024**2, 2),
        "total_mb": round(total_bytes / 1024**2, 2),
        "total_gb": round(total_bytes / 1024**3, 3),
    }


def compute_flops(
    model: nn.Module,
    seq_length: int,
    batch_size: int = 1,
) -> dict[str, Any]:
    """Estimate FLOPs for forward pass."""
    config = getattr(model, "config", None)
    if config is None:
        return {"error": "Model config not found"}

    n_layers = getattr(config, "n_layers", 12)
    d_model = getattr(config, "d_model", 768)
    vocab_size = getattr(config, "vocab_size", 50257)

    # Embedding: vocab_size * d_model lookups
    embed_flops = batch_size * seq_length * d_model

    # Attention per layer: 4 * seq^2 * d_model (Q, K, V, output)
    attn_flops = 4 * batch_size * seq_length * seq_length * d_model

    # FFN per layer: 2 * seq * d_model * 4 * d_model
    d_ff = getattr(config, "d_ff", 4 * d_model)
    ffn_flops = 2 * batch_size * seq_length * d_model * d_ff

    # Per layer total
    layer_flops = attn_flops + ffn_flops

    # Total
    total_flops = embed_flops + n_layers * layer_flops

    # Output projection
    output_flops = batch_size * seq_length * d_model * vocab_size
    total_flops += output_flops

    return {
        "embedding_flops": embed_flops,
        "attention_flops_per_layer": attn_flops,
        "ffn_flops_per_layer": ffn_flops,
        "output_flops": output_flops,
        "total_flops": total_flops,
        "total_tflops": round(total_flops / 1e12, 4),
    }


def analyze_layer_norms(model: nn.Module) -> dict[str, Any]:
    """Analyze layer normalization statistics."""
    ln_stats = []

    for name, module in model.named_modules():
        if ("norm" in name.lower() or "ln" in name.lower()) and hasattr(module, "weight"):
            weight = module.weight.data
            ln_stats.append(
                {
                    "name": name,
                    "mean": round(float(weight.mean()), 4),
                    "std": round(float(weight.std()), 4),
                    "min": round(float(weight.min()), 4),
                    "max": round(float(weight.max()), 4),
                }
            )

    return {
        "num_layer_norms": len(ln_stats),
        "layer_norms": ln_stats[:10],  # Limit output
    }


def analyze_weight_distribution(model: nn.Module) -> dict[str, Any]:
    """Analyze weight distribution across layers."""
    weight_stats = []

    for name, param in model.named_parameters():
        if param.requires_grad and param.dim() >= 2:
            weight_stats.append(
                {
                    "name": name[:50],  # Truncate long names
                    "shape": list(param.shape),
                    "mean": round(float(param.data.mean()), 6),
                    "std": round(float(param.data.std()), 6),
                    "norm": round(float(param.data.norm()), 4),
                }
            )

    return {
        "num_weight_tensors": len(weight_stats),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "weight_stats": weight_stats[:10],  # Limit output
    }


@torch.no_grad()
def extract_attention_patterns(
    model: nn.Module,
    input_ids: torch.Tensor,
    layer_idx: int = -1,
) -> dict[str, Any]:
    """Extract attention patterns from GPT model."""
    # Check if model has transformer structure
    if not hasattr(model, "transformer"):
        return {"error": "Model does not have transformer attribute"}

    # Get attention weights by registering hooks
    attention_weights = []

    def hook_fn(_module: nn.Module, _inputs: Any, outputs: torch.Tensor) -> None:
        # Capture attention output
        attention_weights.append(outputs.detach())

    # Register hooks on attention layers
    hooks = []
    for name, module in model.named_modules():
        if "attn" in name and hasattr(module, "c_proj"):
            hooks.append(module.register_forward_hook(hook_fn))

    # Forward pass
    model.eval()
    _logits, _ = model(input_ids)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Analyze patterns
    result: dict[str, Any] = {
        "num_layers": len(attention_weights),
        "input_length": input_ids.shape[1],
    }

    if attention_weights:
        if layer_idx == -1:
            # All layers
            result["layer_shapes"] = [list(w.shape) for w in attention_weights]
        else:
            # Specific layer
            idx = min(layer_idx, len(attention_weights) - 1)
            result["layer_idx"] = idx
            result["output_shape"] = list(attention_weights[idx].shape)

    return result


def compute_model_sparsity(model: nn.Module, threshold: float = 1e-6) -> dict[str, Any]:
    """Compute weight sparsity in the model."""
    total_params = 0
    zero_params = 0
    near_zero_params = 0

    layer_sparsity = []

    for name, param in model.named_parameters():
        if param.dim() >= 2:
            n_params = param.numel()
            n_zeros = int((param.data == 0).sum())
            n_near_zero = int((param.data.abs() < threshold).sum())

            total_params += n_params
            zero_params += n_zeros
            near_zero_params += n_near_zero

            layer_sparsity.append(
                {
                    "name": name[:40],
                    "sparsity": round(n_zeros / n_params * 100, 2),
                    "near_zero_pct": round(n_near_zero / n_params * 100, 2),
                }
            )

    return {
        "total_parameters": total_params,
        "zero_parameters": zero_params,
        "overall_sparsity_pct": round(zero_params / total_params * 100, 2) if total_params else 0,
        "near_zero_pct": round(near_zero_params / total_params * 100, 2) if total_params else 0,
        "layer_sparsity": layer_sparsity[:10],
    }


def compare_models(model1: nn.Module, model2: nn.Module) -> dict[str, Any]:
    """Compare two models' architectures and parameters."""
    params1 = sum(p.numel() for p in model1.parameters())
    params2 = sum(p.numel() for p in model2.parameters())

    # Compare configs if available
    config1 = getattr(model1, "config", None)
    config2 = getattr(model2, "config", None)

    result: dict[str, Any] = {
        "model1_params": params1,
        "model2_params": params2,
        "param_ratio": round(params1 / params2, 4) if params2 else 0,
    }

    if config1 and config2:
        # Compare key attributes
        for attr in ["n_layers", "d_model", "n_heads", "vocab_size"]:
            v1 = getattr(config1, attr, None)
            v2 = getattr(config2, attr, None)
            if v1 is not None or v2 is not None:
                result[f"{attr}_comparison"] = {"model1": v1, "model2": v2}

    return result
