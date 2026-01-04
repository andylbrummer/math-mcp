"""Tests for attention visualization tools."""

import numpy as np
from llm_mcp.visualization import (
    compare_attention_heads,
    compute_attention_patterns,
    compute_head_importance,
    extract_attention_summary,
    generate_attention_heatmap_data,
)

# Use numpy random generator
_rng = np.random.default_rng(42)


class TestAttentionSummary:
    """Tests for attention summary extraction."""

    def test_extract_summary_2d(self) -> None:
        """Test summary extraction from 2D attention matrix."""
        # Create a simple attention matrix
        seq_len = 10
        attention = _rng.random((seq_len, seq_len))
        attention = attention / attention.sum(axis=1, keepdims=True)

        result = extract_attention_summary(attention)

        assert "seq_length" in result
        assert result["seq_length"] == seq_len
        assert "self_attention_strength" in result
        assert "mean_entropy" in result
        assert "top_attention_pairs" in result

    def test_extract_summary_3d(self) -> None:
        """Test summary extraction from 3D attention [heads, seq, seq]."""
        n_heads, seq_len = 4, 10
        attention = _rng.random((n_heads, seq_len, seq_len))
        attention = attention / attention.sum(axis=-1, keepdims=True)

        result = extract_attention_summary(attention)

        assert result["seq_length"] == seq_len
        assert "forward_attention" in result
        assert "backward_attention" in result

    def test_extract_summary_with_tokens(self) -> None:
        """Test summary with token labels."""
        seq_len = 5
        attention = _rng.random((seq_len, seq_len))
        tokens = ["hello", "world", "this", "is", "test"]

        result = extract_attention_summary(attention, tokens=tokens)

        assert "tokens" in result
        assert result["tokens"] == tokens

    def test_extract_summary_specific_head(self) -> None:
        """Test extracting summary for specific head."""
        n_heads, seq_len = 4, 10
        attention = _rng.random((n_heads, seq_len, seq_len))

        result = extract_attention_summary(attention, head_idx=2)

        assert result["head"] == 2


class TestAttentionPatterns:
    """Tests for attention pattern detection."""

    def test_detect_local_attention(self) -> None:
        """Test detection of local attention pattern."""
        seq_len = 20
        attention = np.zeros((seq_len, seq_len))

        # Create strong local attention (diagonal band)
        for i in range(seq_len):
            for j in range(max(0, i - 3), min(seq_len, i + 4)):
                attention[i, j] = 1.0
        attention = attention / attention.sum(axis=1, keepdims=True)

        result = compute_attention_patterns(attention)

        assert "local_attention_strength" in result
        assert result["local_attention_strength"] > result["global_attention_strength"]

    def test_detect_diagonal_attention(self) -> None:
        """Test detection of strong diagonal (self-attention)."""
        seq_len = 10
        attention = np.eye(seq_len) * 0.9
        attention += np.ones((seq_len, seq_len)) * 0.01
        attention = attention / attention.sum(axis=1, keepdims=True)

        result = compute_attention_patterns(attention)

        assert "diagonal_attention" in result["detected_patterns"]

    def test_uniform_attention_detection(self) -> None:
        """Test detection of uniform attention."""
        seq_len = 10
        attention = np.ones((seq_len, seq_len)) / seq_len

        result = compute_attention_patterns(attention)

        assert "uniform_attention" in result["detected_patterns"]


class TestHeadImportance:
    """Tests for head importance computation."""

    def test_compute_importance_entropy(self) -> None:
        """Test importance computation using entropy method."""
        n_heads, seq_len = 4, 10
        attention = _rng.random((n_heads, seq_len, seq_len))
        attention = attention / attention.sum(axis=-1, keepdims=True)

        result = compute_head_importance(attention, method="entropy")

        assert "num_heads" in result
        assert result["num_heads"] == n_heads
        assert "head_rankings" in result
        assert len(result["head_rankings"]) == n_heads
        assert "most_important_head" in result
        assert "least_important_head" in result

    def test_compute_importance_variance(self) -> None:
        """Test importance computation using variance method."""
        n_heads, seq_len = 4, 10
        attention = _rng.random((n_heads, seq_len, seq_len))

        result = compute_head_importance(attention, method="variance")

        assert result["method"] == "variance"
        assert len(result["head_rankings"]) == n_heads

    def test_importance_ranking_structure(self) -> None:
        """Test structure of head rankings."""
        n_heads, seq_len = 4, 10
        attention = _rng.random((n_heads, seq_len, seq_len))

        result = compute_head_importance(attention)

        for ranking in result["head_rankings"]:
            assert "head" in ranking
            assert "score" in ranking
            assert "max_attention" in ranking
            assert "mean_attention" in ranking


class TestHeatmapGeneration:
    """Tests for attention heatmap data generation."""

    def test_generate_heatmap_basic(self) -> None:
        """Test basic heatmap generation."""
        seq_len = 10
        attention = _rng.random((seq_len, seq_len))

        result = generate_attention_heatmap_data(attention)

        assert "heatmap" in result
        assert "row_labels" in result
        assert "col_labels" in result
        assert "value_range" in result
        assert len(result["heatmap"]) == seq_len

    def test_generate_heatmap_with_tokens(self) -> None:
        """Test heatmap with token labels."""
        seq_len = 5
        attention = _rng.random((seq_len, seq_len))
        tokens = ["a", "b", "c", "d", "e"]

        result = generate_attention_heatmap_data(attention, tokens=tokens)

        assert result["row_labels"] == tokens
        assert result["col_labels"] == tokens

    def test_generate_heatmap_specific_head(self) -> None:
        """Test heatmap for specific head."""
        n_heads, seq_len = 4, 10
        attention = _rng.random((n_heads, seq_len, seq_len))

        result = generate_attention_heatmap_data(attention, head_idx=2)

        assert result["head"] == 2

    def test_heatmap_normalization(self) -> None:
        """Test that heatmap values are normalized."""
        seq_len = 10
        attention = _rng.random((seq_len, seq_len)) * 100  # Large values

        result = generate_attention_heatmap_data(attention)

        heatmap = np.array(result["heatmap"])
        assert heatmap.min() >= 0
        assert heatmap.max() <= 1


class TestHeadComparison:
    """Tests for attention head comparison."""

    def test_compare_heads_basic(self) -> None:
        """Test basic head comparison."""
        n_heads, seq_len = 4, 10
        attention = _rng.random((n_heads, seq_len, seq_len))

        result = compare_attention_heads(attention)

        assert "num_heads" in result
        assert result["num_heads"] == n_heads
        assert "similarity_matrix" in result
        assert "most_diverse_heads" in result
        assert "most_similar_heads" in result
        assert "avg_pairwise_similarity" in result

    def test_similarity_matrix_shape(self) -> None:
        """Test similarity matrix has correct shape."""
        n_heads, seq_len = 4, 10
        attention = _rng.random((n_heads, seq_len, seq_len))

        result = compare_attention_heads(attention)

        sim_matrix = result["similarity_matrix"]
        assert len(sim_matrix) == n_heads
        assert all(len(row) == n_heads for row in sim_matrix)

    def test_detect_redundant_heads(self) -> None:
        """Test detection of redundant (similar) heads."""
        n_heads, seq_len = 4, 10

        # Create nearly identical attention patterns for first two heads
        base_pattern = _rng.random((seq_len, seq_len))
        attention = _rng.random((n_heads, seq_len, seq_len))
        attention[0] = base_pattern
        attention[1] = base_pattern + _rng.random((seq_len, seq_len)) * 0.01

        result = compare_attention_heads(attention)

        # Should detect (0, 1) as redundant pair
        assert "redundant_pairs" in result
