"""Dataset ablation tools for analyzing training data impact."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class AblationResult:
    """Results from an ablation study."""

    baseline_metrics: dict[str, float] = field(default_factory=dict)
    ablation_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    importance_scores: dict[str, float] = field(default_factory=dict)
    summary: str = ""


def compute_data_influence(
    model_losses: list[float],
    sample_indices: list[int],
    total_samples: int,
) -> dict[str, Any]:
    """Compute influence of training samples on model loss.

    Uses a simple approximation based on loss contribution per sample.

    Args:
        model_losses: List of losses for each sample
        sample_indices: Indices of samples in the dataset
        total_samples: Total number of samples in dataset

    Returns:
        Dictionary with influence metrics
    """
    losses = np.array(model_losses)
    indices = np.array(sample_indices)

    # Compute influence as deviation from mean
    mean_loss = float(np.mean(losses))
    std_loss = float(np.std(losses))

    # High-influence samples have unusual losses
    influence_scores = np.abs(losses - mean_loss) / (std_loss + 1e-8)

    # Find most influential samples
    top_k = min(10, len(indices))
    top_indices = np.argsort(influence_scores)[-top_k:][::-1]

    return {
        "mean_loss": round(mean_loss, 4),
        "std_loss": round(std_loss, 4),
        "high_influence_samples": [int(indices[i]) for i in top_indices],
        "high_influence_scores": [round(float(influence_scores[i]), 4) for i in top_indices],
        "low_influence_count": int(np.sum(influence_scores < 0.5)),
        "high_influence_count": int(np.sum(influence_scores > 2.0)),
        "total_samples": total_samples,
    }


def analyze_token_frequency(
    token_ids: list[int],
    vocab_size: int,
    top_k: int = 50,
) -> dict[str, Any]:
    """Analyze token frequency distribution in dataset.

    Args:
        token_ids: List of token IDs
        vocab_size: Vocabulary size
        top_k: Number of top tokens to return

    Returns:
        Dictionary with frequency analysis
    """
    tokens = np.array(token_ids)

    # Count frequencies
    counts = np.bincount(tokens, minlength=vocab_size)

    # Statistics
    total_tokens = len(tokens)
    unique_tokens = int(np.sum(counts > 0))
    coverage = unique_tokens / vocab_size

    # Top tokens
    top_indices = np.argsort(counts)[-top_k:][::-1]
    top_counts = counts[top_indices]

    # Entropy
    probs = counts / (total_tokens + 1e-8)
    entropy = -float(np.sum(probs[probs > 0] * np.log2(probs[probs > 0] + 1e-10)))

    return {
        "total_tokens": total_tokens,
        "unique_tokens": unique_tokens,
        "vocab_coverage": round(coverage * 100, 2),
        "entropy": round(entropy, 4),
        "top_tokens": [int(i) for i in top_indices],
        "top_counts": [int(c) for c in top_counts],
        "max_frequency": int(np.max(counts)),
        "min_nonzero_frequency": int(np.min(counts[counts > 0])) if unique_tokens > 0 else 0,
    }


def compute_sequence_statistics(
    sequences: list[list[int]],
    vocab_size: int,
) -> dict[str, Any]:
    """Compute statistics about sequence lengths and patterns.

    Args:
        sequences: List of token sequences
        vocab_size: Vocabulary size

    Returns:
        Dictionary with sequence statistics
    """
    lengths = [len(seq) for seq in sequences]
    lengths_array = np.array(lengths)

    # Length statistics
    mean_length = float(np.mean(lengths_array))
    std_length = float(np.std(lengths_array))
    min_length = int(np.min(lengths_array))
    max_length = int(np.max(lengths_array))

    # Padding analysis (assume 0 is padding)
    padding_counts = [sum(1 for t in seq if t == 0) for seq in sequences]
    padding_ratios = [
        p / len(seq) if len(seq) > 0 else 0
        for p, seq in zip(padding_counts, sequences, strict=True)
    ]

    # n-gram diversity (bigrams)
    all_bigrams: set[tuple[int, int]] = set()
    for seq in sequences[:1000]:  # Sample for efficiency
        for i in range(len(seq) - 1):
            all_bigrams.add((seq[i], seq[i + 1]))

    # Repetition analysis
    repetition_ratios = []
    for seq in sequences[:1000]:
        unique = len(set(seq))
        repetition_ratios.append(1 - unique / max(len(seq), 1))

    return {
        "num_sequences": len(sequences),
        "mean_length": round(mean_length, 2),
        "std_length": round(std_length, 2),
        "min_length": min_length,
        "max_length": max_length,
        "avg_padding_ratio": round(float(np.mean(padding_ratios)), 4),
        "unique_bigrams": len(all_bigrams),
        "max_possible_bigrams": vocab_size * vocab_size,
        "bigram_coverage": round(len(all_bigrams) / (vocab_size * vocab_size) * 100, 4),
        "avg_repetition_ratio": round(float(np.mean(repetition_ratios)), 4),
    }


def run_ablation_study(
    baseline_loss: float,
    ablation_losses: dict[str, float],
    _component_names: list[str] | None = None,
) -> AblationResult:
    """Run an ablation study comparing baseline to variants.

    Args:
        baseline_loss: Loss with full dataset
        ablation_losses: Dictionary mapping ablation name to loss
        component_names: Optional names for components being ablated

    Returns:
        AblationResult with analysis
    """
    result = AblationResult()
    result.baseline_metrics = {"loss": baseline_loss, "perplexity": float(np.exp(baseline_loss))}

    # Compute metrics for each ablation
    importance_scores: dict[str, float] = {}

    for name, loss in ablation_losses.items():
        perplexity = float(np.exp(loss))
        delta = loss - baseline_loss
        relative_change = delta / (baseline_loss + 1e-8)

        result.ablation_metrics[name] = {
            "loss": round(loss, 4),
            "perplexity": round(perplexity, 2),
            "delta_loss": round(delta, 4),
            "relative_change": round(relative_change * 100, 2),
        }

        # Importance is how much removing this component hurts
        importance_scores[name] = round(max(0, delta), 4)

    result.importance_scores = dict(sorted(importance_scores.items(), key=lambda x: -x[1]))

    # Generate summary
    if importance_scores:
        most_important = max(importance_scores, key=lambda k: importance_scores[k])
        least_important = min(importance_scores, key=lambda k: importance_scores[k])
        result.summary = (
            f"Most important component: {most_important} "
            f"(+{importance_scores[most_important]:.4f} loss when removed). "
            f"Least important: {least_important} "
            f"(+{importance_scores[least_important]:.4f} loss when removed)."
        )

    return result


def analyze_class_balance(
    labels: list[int],
    num_classes: int,
) -> dict[str, Any]:
    """Analyze class balance in classification dataset.

    Args:
        labels: List of class labels
        num_classes: Number of classes

    Returns:
        Dictionary with balance analysis
    """
    labels_array = np.array(labels)
    counts = np.bincount(labels_array, minlength=num_classes)

    # Statistics
    total = len(labels)
    expected_per_class = total / num_classes

    # Imbalance metrics
    actual_distribution = counts / total
    uniform_distribution = np.ones(num_classes) / num_classes

    # KL divergence from uniform
    kl_div = float(
        np.sum(actual_distribution * np.log(actual_distribution / uniform_distribution + 1e-10))
    )

    # Gini coefficient
    sorted_counts = np.sort(counts)
    n = len(sorted_counts)
    gini = float(
        (2 * np.sum(np.arange(1, n + 1) * sorted_counts) - (n + 1) * np.sum(sorted_counts))
        / (n * np.sum(sorted_counts) + 1e-10)
    )

    # Find minority/majority classes
    minority_classes = np.where(counts < expected_per_class * 0.5)[0]
    majority_classes = np.where(counts > expected_per_class * 2)[0]

    return {
        "num_classes": num_classes,
        "total_samples": total,
        "class_counts": counts.tolist(),
        "expected_per_class": round(expected_per_class, 2),
        "min_class_count": int(np.min(counts)),
        "max_class_count": int(np.max(counts)),
        "imbalance_ratio": round(float(np.max(counts) / (np.min(counts) + 1)), 2),
        "kl_divergence_from_uniform": round(kl_div, 4),
        "gini_coefficient": round(gini, 4),
        "minority_classes": minority_classes.tolist(),
        "majority_classes": majority_classes.tolist(),
    }


def suggest_data_augmentation(
    token_frequency: dict[str, Any],
    sequence_stats: dict[str, Any],
    class_balance: dict[str, Any] | None = None,
) -> list[str]:
    """Suggest data augmentation strategies based on analysis.

    Args:
        token_frequency: Output from analyze_token_frequency
        sequence_stats: Output from compute_sequence_statistics
        class_balance: Optional output from analyze_class_balance

    Returns:
        List of augmentation suggestions
    """
    suggestions = []

    # Token frequency suggestions
    if token_frequency["vocab_coverage"] < 50:
        suggestions.append(
            f"Low vocabulary coverage ({token_frequency['vocab_coverage']}%). "
            "Consider adding more diverse text data."
        )

    if token_frequency["entropy"] < 5:
        suggestions.append(
            f"Low token entropy ({token_frequency['entropy']}). "
            "Dataset may be too repetitive. Consider token-level augmentation."
        )

    # Sequence statistics suggestions
    if sequence_stats["avg_repetition_ratio"] > 0.5:
        suggestions.append(
            f"High repetition ratio ({sequence_stats['avg_repetition_ratio']}). "
            "Consider removing duplicate sequences or using deduplication."
        )

    std_length = sequence_stats.get("std_length", 0)
    mean_length = sequence_stats.get("mean_length", 1)
    if std_length / (mean_length + 1e-8) > 0.5:
        suggestions.append("High length variance. Consider length-based stratified sampling.")

    # Class balance suggestions
    if class_balance:
        if class_balance["imbalance_ratio"] > 10:
            suggestions.append(
                f"Severe class imbalance (ratio: {class_balance['imbalance_ratio']}). "
                "Consider oversampling minority classes or using class weights."
            )

        if class_balance["gini_coefficient"] > 0.5:
            suggestions.append(
                f"High Gini coefficient ({class_balance['gini_coefficient']}). "
                "Consider SMOTE or other rebalancing techniques."
            )

    if not suggestions:
        suggestions.append("Dataset appears balanced. No major issues detected.")

    return suggestions
