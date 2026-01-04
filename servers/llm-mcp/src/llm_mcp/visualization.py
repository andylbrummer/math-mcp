"""Visualization tools for LLM models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class AttentionVisualization:
    """Attention pattern visualization result."""

    layer: int
    head: int | None
    attention_matrix: list[list[float]] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def extract_attention_summary(
    attention_weights: np.ndarray,
    tokens: list[str] | None = None,
    layer_idx: int = -1,
    head_idx: int | None = None,
) -> dict[str, Any]:
    """Extract summary statistics from attention weights.

    Args:
        attention_weights: Attention matrix [batch, heads, seq, seq] or [seq, seq]
        tokens: Optional token strings
        layer_idx: Layer index (for reporting)
        head_idx: Optional specific head to analyze

    Returns:
        Dictionary with attention summary
    """
    weights = np.array(attention_weights)

    # Handle different input shapes: [batch, heads, seq, seq], [heads, seq, seq], [seq, seq]
    if weights.ndim == 4:
        if head_idx is not None:
            weights = weights[:, head_idx, :, :]
        weights = weights.mean(axis=(0, 1)) if weights.ndim == 4 else weights.mean(axis=0)
    elif weights.ndim == 3:
        weights = weights[head_idx] if head_idx is not None else weights.mean(axis=0)
    elif weights.ndim != 2:
        return {"error": f"Unexpected attention shape: {weights.shape}"}

    seq_len = weights.shape[0]

    # Compute attention statistics
    diagonal = np.diag(weights).mean()  # Self-attention
    upper_tri = np.triu(weights, k=1).sum() / max((seq_len * (seq_len - 1) / 2), 1)
    lower_tri = np.tril(weights, k=-1).sum() / max((seq_len * (seq_len - 1) / 2), 1)

    # Entropy per position (how distributed is attention)
    entropy_per_pos = []
    for i in range(seq_len):
        row = weights[i]
        row = row / (row.sum() + 1e-10)
        ent = -np.sum(row * np.log2(row + 1e-10))
        entropy_per_pos.append(float(ent))

    # Find attention peaks
    flat_idx = np.argsort(weights.flatten())[-10:][::-1]
    top_pairs = [(int(idx // seq_len), int(idx % seq_len)) for idx in flat_idx]

    result = {
        "layer": layer_idx,
        "head": head_idx,
        "seq_length": seq_len,
        "self_attention_strength": round(float(diagonal), 4),
        "forward_attention": round(float(lower_tri), 4),
        "backward_attention": round(float(upper_tri), 4),
        "mean_entropy": round(float(np.mean(entropy_per_pos)), 4),
        "max_entropy": round(float(np.max(entropy_per_pos)), 4),
        "min_entropy": round(float(np.min(entropy_per_pos)), 4),
        "top_attention_pairs": top_pairs[:5],
    }

    if tokens:
        result["tokens"] = tokens[:20]  # Limit output

    return result


def compute_attention_patterns(
    attention_weights: np.ndarray,
    _pattern_type: str = "all",
) -> dict[str, Any]:
    """Identify common attention patterns.

    Args:
        attention_weights: Attention matrix [seq, seq] or [heads, seq, seq]
        pattern_type: Type of pattern to detect: "all", "local", "global", "sparse"

    Returns:
        Dictionary with detected patterns
    """
    weights = np.array(attention_weights)

    if weights.ndim == 3:
        weights = weights.mean(axis=0)  # Average over heads

    seq_len = weights.shape[0]
    patterns: dict[str, Any] = {"detected_patterns": []}

    # Local attention pattern (diagonal band)
    local_band = 5
    local_mask = np.abs(np.arange(seq_len)[:, None] - np.arange(seq_len)) <= local_band
    local_strength = float(weights[local_mask].mean())
    global_strength = float(weights[~local_mask].mean()) if (~local_mask).sum() > 0 else 0

    if local_strength > 2 * global_strength:
        patterns["detected_patterns"].append("local_attention")
        patterns["local_window_size"] = local_band

    # First token attention (often [CLS] or [BOS])
    first_col = weights[:, 0].mean()
    if first_col > 0.2:
        patterns["detected_patterns"].append("first_token_attention")
        patterns["first_token_strength"] = round(float(first_col), 4)

    # Uniform/global attention
    if np.std(weights) < 0.05:
        patterns["detected_patterns"].append("uniform_attention")

    # Sparse attention (few strong connections)
    threshold = np.percentile(weights, 95)
    sparse_ratio = (weights > threshold).sum() / weights.size
    if sparse_ratio < 0.1:
        patterns["detected_patterns"].append("sparse_attention")
        patterns["sparsity_ratio"] = round(float(sparse_ratio), 4)

    # Diagonal attention (strong self-attention)
    diag_strength = float(np.diag(weights).mean())
    off_diag_sum = weights.sum() - np.diag(weights).sum()
    off_diag_strength = float(off_diag_sum / max(weights.size - seq_len, 1))
    if diag_strength > 3 * off_diag_strength:
        patterns["detected_patterns"].append("diagonal_attention")
        patterns["diagonal_strength"] = round(diag_strength, 4)

    patterns["local_attention_strength"] = round(local_strength, 4)
    patterns["global_attention_strength"] = round(global_strength, 4)

    return patterns


def compute_head_importance(
    attention_weights: np.ndarray,
    method: str = "entropy",
) -> dict[str, Any]:
    """Compute importance scores for attention heads.

    Args:
        attention_weights: Attention matrix [heads, seq, seq] or [batch, heads, seq, seq]
        method: Scoring method: "entropy", "gradient", "variance"

    Returns:
        Dictionary with head importance scores
    """
    weights = np.array(attention_weights)

    # Handle batch dimension
    if weights.ndim == 4:
        weights = weights.mean(axis=0)  # [heads, seq, seq]

    if weights.ndim != 3:
        return {"error": f"Expected 3D attention weights, got {weights.ndim}D"}

    n_heads = weights.shape[0]
    seq_len = weights.shape[1]

    head_scores = []

    for h in range(n_heads):
        head_attn = weights[h]

        if method == "entropy":
            # Higher entropy = more uniform = less "focused"
            entropies = []
            for i in range(seq_len):
                row = head_attn[i]
                row = row / (row.sum() + 1e-10)
                ent = -np.sum(row * np.log2(row + 1e-10))
                entropies.append(ent)
            score = float(np.mean(entropies))
        elif method == "variance":
            # Higher variance = more diverse attention patterns
            score = float(np.var(head_attn))
        else:  # "gradient" - would need gradients, simulate
            score = float(np.std(head_attn) * np.mean(np.abs(head_attn)))

        head_scores.append(
            {
                "head": h,
                "score": round(score, 4),
                "max_attention": round(float(head_attn.max()), 4),
                "mean_attention": round(float(head_attn.mean()), 4),
            }
        )

    # Rank heads by score
    if method == "entropy":
        # Lower entropy = more focused = potentially more important
        head_scores.sort(key=lambda x: x["score"])
    else:
        # Higher score = more important
        head_scores.sort(key=lambda x: -x["score"])

    return {
        "method": method,
        "num_heads": n_heads,
        "seq_length": seq_len,
        "head_rankings": head_scores,
        "most_important_head": head_scores[0]["head"],
        "least_important_head": head_scores[-1]["head"],
    }


def generate_attention_heatmap_data(
    attention_weights: np.ndarray,
    tokens: list[str] | None = None,
    head_idx: int = 0,
    max_seq: int = 50,
) -> dict[str, Any]:
    """Generate data for attention heatmap visualization.

    Args:
        attention_weights: Attention matrix [heads, seq, seq] or [seq, seq]
        tokens: Optional token labels
        head_idx: Head to visualize (if multi-head)
        max_seq: Maximum sequence length to include

    Returns:
        Dictionary with heatmap data
    """
    weights = np.array(attention_weights)

    # Extract specific head if needed
    if weights.ndim == 3:
        weights = weights[head_idx]

    # Truncate if too long
    if weights.shape[0] > max_seq:
        weights = weights[:max_seq, :max_seq]

    seq_len = weights.shape[0]

    # Normalize for visualization
    weights_normalized = (weights - weights.min()) / (weights.max() - weights.min() + 1e-10)

    # Convert to nested list for JSON serialization
    heatmap_data = weights_normalized.tolist()

    # Generate labels
    labels = tokens[:seq_len] if tokens else [str(i) for i in range(seq_len)]

    return {
        "head": head_idx,
        "seq_length": seq_len,
        "heatmap": heatmap_data,
        "row_labels": labels,
        "col_labels": labels,
        "value_range": {
            "min": round(float(weights.min()), 4),
            "max": round(float(weights.max()), 4),
        },
    }


def compare_attention_heads(
    attention_weights: np.ndarray,
) -> dict[str, Any]:
    """Compare attention patterns across heads.

    Args:
        attention_weights: Attention matrix [heads, seq, seq]

    Returns:
        Dictionary with head comparison
    """
    weights = np.array(attention_weights)

    if weights.ndim != 3:
        return {"error": "Expected 3D attention weights [heads, seq, seq]"}

    n_heads = weights.shape[0]

    # Compute pairwise similarity between heads
    head_patterns = weights.reshape(n_heads, -1)  # [heads, seq*seq]

    similarities = np.zeros((n_heads, n_heads))
    for i in range(n_heads):
        for j in range(n_heads):
            # Cosine similarity
            sim = np.dot(head_patterns[i], head_patterns[j])
            norm = np.linalg.norm(head_patterns[i]) * np.linalg.norm(head_patterns[j])
            similarities[i, j] = sim / (norm + 1e-10)

    # Find redundant heads (high similarity)
    redundant_pairs = []
    for i in range(n_heads):
        for j in range(i + 1, n_heads):
            if similarities[i, j] > 0.9:
                redundant_pairs.append((i, j, round(float(similarities[i, j]), 4)))

    # Find diverse heads (low average similarity)
    avg_similarities = similarities.mean(axis=1)
    diverse_heads = np.argsort(avg_similarities)[:3].tolist()
    similar_heads = np.argsort(avg_similarities)[-3:].tolist()

    return {
        "num_heads": n_heads,
        "similarity_matrix": [[round(float(s), 4) for s in row] for row in similarities],
        "redundant_pairs": redundant_pairs[:5],
        "most_diverse_heads": diverse_heads,
        "most_similar_heads": similar_heads,
        "avg_pairwise_similarity": round(float(similarities.mean()), 4),
    }
