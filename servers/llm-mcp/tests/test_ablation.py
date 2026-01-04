"""Tests for dataset ablation tools."""

from llm_mcp.ablation import (
    AblationResult,
    analyze_class_balance,
    analyze_token_frequency,
    compute_data_influence,
    compute_sequence_statistics,
    run_ablation_study,
    suggest_data_augmentation,
)


class TestDataInfluence:
    """Tests for data influence computation."""

    def test_compute_influence_basic(self) -> None:
        """Test basic influence computation."""
        losses = [1.0, 1.1, 0.9, 1.0, 5.0]  # One outlier
        indices = list(range(len(losses)))

        result = compute_data_influence(losses, indices, len(losses))

        assert "mean_loss" in result
        assert "std_loss" in result
        assert "high_influence_samples" in result
        assert result["total_samples"] == 5

    def test_high_influence_detection(self) -> None:
        """Test that outliers are detected as high influence."""
        # Create data with one clear outlier
        losses = [1.0] * 10 + [10.0]
        indices = list(range(len(losses)))

        result = compute_data_influence(losses, indices, len(losses))

        # The outlier (index 10) should be in high influence samples
        assert 10 in result["high_influence_samples"]

    def test_low_influence_count(self) -> None:
        """Test low influence counting."""
        # All identical losses = all low influence
        losses = [1.0] * 100
        indices = list(range(len(losses)))

        result = compute_data_influence(losses, indices, len(losses))

        # With zero std, all are at mean so low influence
        assert result["low_influence_count"] >= 0


class TestTokenFrequency:
    """Tests for token frequency analysis."""

    def test_analyze_frequency_basic(self) -> None:
        """Test basic frequency analysis."""
        tokens = [0, 1, 2, 0, 1, 0]  # 0 appears 3 times, 1 appears 2 times
        vocab_size = 100

        result = analyze_token_frequency(tokens, vocab_size)

        assert result["total_tokens"] == 6
        assert result["unique_tokens"] == 3
        assert "entropy" in result
        assert "top_tokens" in result

    def test_vocab_coverage(self) -> None:
        """Test vocabulary coverage calculation."""
        # Use all tokens in vocab
        vocab_size = 10
        tokens = list(range(vocab_size)) * 5  # Each token 5 times

        result = analyze_token_frequency(tokens, vocab_size)

        assert result["vocab_coverage"] == 100.0
        assert result["unique_tokens"] == vocab_size

    def test_entropy_calculation(self) -> None:
        """Test entropy is higher for uniform distribution."""
        vocab_size = 100

        # Uniform distribution
        uniform_tokens = list(range(vocab_size)) * 10
        uniform_result = analyze_token_frequency(uniform_tokens, vocab_size)

        # Skewed distribution (just a few tokens)
        skewed_tokens = [0] * 500 + [1] * 100 + [2] * 10
        skewed_result = analyze_token_frequency(skewed_tokens, vocab_size)

        # Uniform should have higher entropy
        assert uniform_result["entropy"] > skewed_result["entropy"]


class TestSequenceStatistics:
    """Tests for sequence statistics."""

    def test_compute_stats_basic(self) -> None:
        """Test basic sequence statistics."""
        sequences = [[1, 2, 3, 4, 5], [1, 2, 3], [1, 2, 3, 4, 5, 6, 7]]
        vocab_size = 10

        result = compute_sequence_statistics(sequences, vocab_size)

        assert result["num_sequences"] == 3
        assert result["min_length"] == 3
        assert result["max_length"] == 7
        assert "mean_length" in result
        assert "std_length" in result

    def test_padding_detection(self) -> None:
        """Test padding detection."""
        sequences = [[0, 0, 1, 2], [0, 1, 2, 3], [1, 2, 3, 4]]  # First seq has 2 pads
        vocab_size = 10

        result = compute_sequence_statistics(sequences, vocab_size)

        assert result["avg_padding_ratio"] > 0

    def test_bigram_coverage(self) -> None:
        """Test bigram diversity calculation."""
        # Repetitive sequences = low bigram diversity
        sequences = [[1, 2, 1, 2, 1, 2] for _ in range(100)]
        vocab_size = 10

        result = compute_sequence_statistics(sequences, vocab_size)

        # Only 2 unique bigrams: (1,2) and (2,1)
        assert result["unique_bigrams"] == 2


class TestAblationStudy:
    """Tests for ablation study."""

    def test_run_ablation_basic(self) -> None:
        """Test basic ablation study."""
        baseline_loss = 2.0
        ablation_losses = {
            "remove_A": 2.5,
            "remove_B": 2.2,
            "remove_C": 1.9,
        }

        result = run_ablation_study(baseline_loss, ablation_losses)

        assert isinstance(result, AblationResult)
        assert result.baseline_metrics["loss"] == baseline_loss
        assert "remove_A" in result.ablation_metrics
        assert "remove_A" in result.importance_scores

    def test_importance_ordering(self) -> None:
        """Test that importance scores are correctly ordered."""
        baseline_loss = 2.0
        ablation_losses = {
            "low_importance": 2.1,  # Small increase
            "high_importance": 3.0,  # Large increase
            "negative_impact": 1.5,  # Decrease (improvement when removed)
        }

        result = run_ablation_study(baseline_loss, ablation_losses)

        # Most important should be first
        importance_keys = list(result.importance_scores.keys())
        assert importance_keys[0] == "high_importance"

    def test_summary_generation(self) -> None:
        """Test that summary is generated."""
        baseline_loss = 2.0
        ablation_losses = {"A": 2.5, "B": 2.1}

        result = run_ablation_study(baseline_loss, ablation_losses)

        assert len(result.summary) > 0
        assert "important" in result.summary.lower()


class TestClassBalance:
    """Tests for class balance analysis."""

    def test_balanced_dataset(self) -> None:
        """Test analysis of balanced dataset."""
        labels = [0, 1, 2, 3, 4] * 100  # Perfectly balanced
        num_classes = 5

        result = analyze_class_balance(labels, num_classes)

        assert result["num_classes"] == 5
        # Allow for floating point rounding
        assert abs(result["imbalance_ratio"] - 1.0) < 0.1
        assert len(result["minority_classes"]) == 0
        assert len(result["majority_classes"]) == 0

    def test_imbalanced_dataset(self) -> None:
        """Test analysis of imbalanced dataset."""
        labels = [0] * 1000 + [1] * 10 + [2] * 10  # Highly imbalanced
        num_classes = 3

        result = analyze_class_balance(labels, num_classes)

        assert result["imbalance_ratio"] > 10
        assert 1 in result["minority_classes"] or 2 in result["minority_classes"]
        assert 0 in result["majority_classes"]

    def test_kl_divergence(self) -> None:
        """Test KL divergence from uniform."""
        # Balanced dataset should have low KL
        balanced = [0, 1, 2, 3] * 100
        balanced_result = analyze_class_balance(balanced, 4)

        # Imbalanced should have high KL
        imbalanced = [0] * 390 + [1] * 5 + [2] * 3 + [3] * 2
        imbalanced_result = analyze_class_balance(imbalanced, 4)

        assert balanced_result["kl_divergence_from_uniform"] < 0.01
        assert imbalanced_result["kl_divergence_from_uniform"] > 0.5


class TestAugmentationSuggestions:
    """Tests for augmentation suggestions."""

    def test_suggestions_for_low_coverage(self) -> None:
        """Test suggestions for low vocabulary coverage."""
        # Simulate low coverage
        token_freq = {
            "vocab_coverage": 10.0,  # Very low
            "entropy": 8.0,
        }
        seq_stats = {"avg_repetition_ratio": 0.1}

        suggestions = suggest_data_augmentation(token_freq, seq_stats)

        assert any("coverage" in s.lower() for s in suggestions)

    def test_suggestions_for_high_repetition(self) -> None:
        """Test suggestions for high repetition."""
        token_freq = {"vocab_coverage": 80.0, "entropy": 10.0}
        seq_stats = {"avg_repetition_ratio": 0.6}  # High repetition

        suggestions = suggest_data_augmentation(token_freq, seq_stats)

        assert any("repetition" in s.lower() or "duplicate" in s.lower() for s in suggestions)

    def test_healthy_dataset_suggestions(self) -> None:
        """Test suggestions for healthy dataset."""
        token_freq = {"vocab_coverage": 80.0, "entropy": 12.0}
        seq_stats = {"avg_repetition_ratio": 0.1, "std_length": 10, "mean_length": 100}

        suggestions = suggest_data_augmentation(token_freq, seq_stats)

        # Should indicate no major issues
        assert any("balanced" in s.lower() or "no major" in s.lower() for s in suggestions)
