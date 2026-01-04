"""Tests for LLM analysis tools."""

import pytest
from llm_mcp.analysis import (
    analyze_layer_norms,
    analyze_weight_distribution,
    compare_models,
    compute_flops,
    compute_model_sparsity,
    estimate_memory_usage,
)
from llm_mcp.models import GPT, GPTConfig, Mamba, MambaConfig


@pytest.fixture
def gpt_model() -> GPT:
    """Create a small GPT model for testing."""
    config = GPTConfig.from_preset("gpt-micro")
    return GPT(config)


@pytest.fixture
def mamba_model() -> Mamba:
    """Create a small Mamba model for testing."""
    config = MambaConfig(
        vocab_size=1000,
        n_layers=2,
        d_model=64,
        d_state=8,
        d_conv=2,
        max_seq_len=128,
    )
    return Mamba(config)


class TestMemoryEstimation:
    """Tests for memory estimation."""

    def test_estimate_memory_basic(self, gpt_model: GPT) -> None:
        """Test basic memory estimation."""
        result = estimate_memory_usage(gpt_model, batch_size=4, seq_length=128)

        assert "parameters_mb" in result
        assert "gradients_mb" in result
        assert "optimizer_mb" in result
        assert "activations_mb" in result
        assert "total_mb" in result
        assert "total_gb" in result

        # Total should be sum of components
        expected_total = (
            result["parameters_mb"]
            + result["gradients_mb"]
            + result["optimizer_mb"]
            + result["activations_mb"]
        )
        assert abs(result["total_mb"] - expected_total) < 0.1

    def test_mixed_precision_reduces_memory(self, gpt_model: GPT) -> None:
        """Test that mixed precision reduces memory estimate."""
        result_fp32 = estimate_memory_usage(
            gpt_model, batch_size=4, seq_length=128, mixed_precision=False
        )
        result_fp16 = estimate_memory_usage(
            gpt_model, batch_size=4, seq_length=128, mixed_precision=True
        )

        # FP16 should use less activation memory
        assert result_fp16["activations_mb"] < result_fp32["activations_mb"]


class TestFLOPsComputation:
    """Tests for FLOP computation."""

    def test_compute_flops_basic(self, gpt_model: GPT) -> None:
        """Test basic FLOP computation."""
        result = compute_flops(gpt_model, seq_length=128)

        assert "embedding_flops" in result
        assert "attention_flops_per_layer" in result
        assert "ffn_flops_per_layer" in result
        assert "total_flops" in result
        assert "total_tflops" in result

        assert result["total_flops"] > 0
        assert result["total_tflops"] >= 0

    def test_flops_scale_with_batch(self, gpt_model: GPT) -> None:
        """Test that FLOPs scale linearly with batch size."""
        result_1 = compute_flops(gpt_model, seq_length=128, batch_size=1)
        result_4 = compute_flops(gpt_model, seq_length=128, batch_size=4)

        # FLOPs should scale ~4x with batch size
        ratio = result_4["total_flops"] / result_1["total_flops"]
        assert 3.5 < ratio < 4.5


class TestWeightAnalysis:
    """Tests for weight distribution analysis."""

    def test_analyze_weights_basic(self, gpt_model: GPT) -> None:
        """Test basic weight analysis."""
        result = analyze_weight_distribution(gpt_model)

        assert "num_weight_tensors" in result
        assert "total_parameters" in result
        assert "weight_stats" in result

        assert result["num_weight_tensors"] > 0
        # total_parameters includes all params, num_parameters excludes position embeddings
        assert result["total_parameters"] >= gpt_model.num_parameters

    def test_weight_stats_structure(self, gpt_model: GPT) -> None:
        """Test weight stats have correct structure."""
        result = analyze_weight_distribution(gpt_model)

        for stat in result["weight_stats"]:
            assert "name" in stat
            assert "shape" in stat
            assert "mean" in stat
            assert "std" in stat
            assert "norm" in stat


class TestSparsityAnalysis:
    """Tests for sparsity analysis."""

    def test_compute_sparsity_basic(self, gpt_model: GPT) -> None:
        """Test basic sparsity computation."""
        result = compute_model_sparsity(gpt_model)

        assert "total_parameters" in result
        assert "zero_parameters" in result
        assert "overall_sparsity_pct" in result
        assert "near_zero_pct" in result
        assert "layer_sparsity" in result

        # New model should have ~0% zero weights
        assert result["overall_sparsity_pct"] == 0.0

    def test_sparsity_with_threshold(self, gpt_model: GPT) -> None:
        """Test sparsity with custom threshold."""
        result = compute_model_sparsity(gpt_model, threshold=1e-3)

        # Near-zero percentage should be higher with larger threshold
        assert result["near_zero_pct"] >= 0


class TestLayerNormAnalysis:
    """Tests for layer normalization analysis."""

    def test_analyze_norms_basic(self, gpt_model: GPT) -> None:
        """Test basic layer norm analysis."""
        result = analyze_layer_norms(gpt_model)

        assert "num_layer_norms" in result
        assert "layer_norms" in result

        # GPT-micro has 2 layers * 2 norms + 1 final = 5
        assert result["num_layer_norms"] == 5

    def test_layer_norm_stats_structure(self, gpt_model: GPT) -> None:
        """Test layer norm stats have correct structure."""
        result = analyze_layer_norms(gpt_model)

        for stat in result["layer_norms"]:
            assert "name" in stat
            assert "mean" in stat
            assert "std" in stat
            assert "min" in stat
            assert "max" in stat

            # Initial weights should be ones
            assert abs(stat["mean"] - 1.0) < 0.01


class TestModelComparison:
    """Tests for model comparison."""

    def test_compare_same_architecture(self, gpt_model: GPT) -> None:
        """Test comparing same architecture."""
        config_large = GPTConfig.from_preset("gpt-mini")
        model_large = GPT(config_large)

        result = compare_models(gpt_model, model_large)

        assert "model1_params" in result
        assert "model2_params" in result
        assert "param_ratio" in result

        # Large model should have more params
        assert result["model2_params"] > result["model1_params"]
        assert result["param_ratio"] < 1

    def test_compare_different_architectures(self, gpt_model: GPT, mamba_model: Mamba) -> None:
        """Test comparing different architectures."""
        result = compare_models(gpt_model, mamba_model)

        assert "model1_params" in result
        assert "model2_params" in result
        assert result["model1_params"] > 0
        assert result["model2_params"] > 0
