"""Tests for LLM MCP server."""

import ast

import pytest
from llm_mcp.server import (
    _tool_create_model,
    _tool_create_tokenizer,
    _tool_create_trainer,
    _tool_evaluate_model,
    _tool_generate_text,
    _tool_get_model_config,
    _tool_get_training_status,
    _tool_info,
    _tool_list_models,
    _tool_load_dataset,
    _tool_prepare_dataset,
    _tool_tokenize_text,
    _tool_train_step,
)


@pytest.mark.asyncio
async def test_info_overview() -> None:
    """Test info tool overview."""
    result = await _tool_info({})
    assert len(result) == 1
    assert "LLM MCP" in result[0]["text"]


@pytest.mark.asyncio
async def test_info_models() -> None:
    """Test info tool models topic."""
    result = await _tool_info({"topic": "models"})
    assert "GPT" in result[0]["text"]
    assert "Mamba" in result[0]["text"]


@pytest.mark.asyncio
async def test_create_gpt_model() -> None:
    """Test GPT model creation."""
    result = await _tool_create_model(
        {
            "architecture": "gpt",
            "preset": "gpt2-small",
        }
    )
    data = ast.literal_eval(result[0]["text"])
    assert "model_id" in data
    assert data["architecture"] == "gpt"
    assert "model://" in data["model_id"]


@pytest.mark.asyncio
async def test_create_mamba_model() -> None:
    """Test Mamba model creation."""
    result = await _tool_create_model(
        {
            "architecture": "mamba",
            "preset": "mamba-small",
        }
    )
    data = ast.literal_eval(result[0]["text"])
    assert "model_id" in data
    assert data["architecture"] == "mamba"


@pytest.mark.asyncio
async def test_create_custom_model() -> None:
    """Test custom model creation."""
    result = await _tool_create_model(
        {
            "architecture": "gpt",
            "n_layers": 6,
            "n_heads": 8,
            "d_model": 512,
            "vocab_size": 32000,
        }
    )
    data = ast.literal_eval(result[0]["text"])
    assert "model_id" in data
    assert data["config"]["n_layers"] == 6
    assert data["config"]["n_heads"] == 8


@pytest.mark.asyncio
async def test_get_model_config() -> None:
    """Test getting model configuration."""
    # Create model first
    model_result = await _tool_create_model({"architecture": "gpt", "preset": "gpt2-small"})
    model_data = ast.literal_eval(model_result[0]["text"])
    model_id = model_data["model_id"]

    # Get config
    result = await _tool_get_model_config({"model_id": model_id})
    config = ast.literal_eval(result[0]["text"])
    assert config["architecture"] == "gpt"
    assert "n_layers" in config


@pytest.mark.asyncio
async def test_list_models() -> None:
    """Test listing models."""
    # Create a model
    await _tool_create_model({"architecture": "gpt"})

    result = await _tool_list_models({})
    data = ast.literal_eval(result[0]["text"])
    assert "models" in data
    assert "count" in data
    assert data["count"] >= 1


@pytest.mark.asyncio
async def test_create_tokenizer() -> None:
    """Test tokenizer creation."""
    result = await _tool_create_tokenizer(
        {
            "tokenizer_type": "tiktoken",
            "pretrained": "gpt2",
        }
    )
    data = ast.literal_eval(result[0]["text"])
    assert "tokenizer_id" in data
    assert data["type"] == "tiktoken"


@pytest.mark.asyncio
async def test_tokenize_text() -> None:
    """Test text tokenization."""
    # Create tokenizer
    tok_result = await _tool_create_tokenizer({"tokenizer_type": "tiktoken"})
    tok_data = ast.literal_eval(tok_result[0]["text"])
    tokenizer_id = tok_data["tokenizer_id"]

    # Tokenize
    result = await _tool_tokenize_text(
        {
            "tokenizer_id": tokenizer_id,
            "text": "Hello world, this is a test.",
        }
    )
    data = ast.literal_eval(result[0]["text"])
    assert "num_tokens" in data
    assert data["num_tokens"] > 0


@pytest.mark.asyncio
async def test_load_dataset() -> None:
    """Test dataset loading."""
    result = await _tool_load_dataset(
        {
            "dataset_name": "wikitext",
            "split": "train",
        }
    )
    data = ast.literal_eval(result[0]["text"])
    assert "dataset_id" in data
    assert data["name"] == "wikitext"
    assert data["size"] > 0


@pytest.mark.asyncio
async def test_prepare_dataset() -> None:
    """Test dataset preparation."""
    # Load dataset and create tokenizer
    ds_result = await _tool_load_dataset({"dataset_name": "tinystories"})
    ds_data = ast.literal_eval(ds_result[0]["text"])
    dataset_id = ds_data["dataset_id"]

    tok_result = await _tool_create_tokenizer({"tokenizer_type": "tiktoken"})
    tok_data = ast.literal_eval(tok_result[0]["text"])
    tokenizer_id = tok_data["tokenizer_id"]

    # Prepare
    result = await _tool_prepare_dataset(
        {
            "dataset_id": dataset_id,
            "tokenizer_id": tokenizer_id,
            "max_length": 256,
            "batch_size": 4,
        }
    )
    data = ast.literal_eval(result[0]["text"])
    assert data["prepared"] is True
    assert data["max_length"] == 256
    assert data["batch_size"] == 4


@pytest.mark.asyncio
async def test_create_trainer() -> None:
    """Test trainer creation."""
    # Create model and dataset
    model_result = await _tool_create_model({"architecture": "gpt"})
    model_data = ast.literal_eval(model_result[0]["text"])
    model_id = model_data["model_id"]

    ds_result = await _tool_load_dataset({"dataset_name": "wikitext"})
    ds_data = ast.literal_eval(ds_result[0]["text"])
    dataset_id = ds_data["dataset_id"]

    # Create trainer
    result = await _tool_create_trainer(
        {
            "model_id": model_id,
            "dataset_id": dataset_id,
            "learning_rate": 1e-4,
            "max_steps": 500,
        }
    )
    data = ast.literal_eval(result[0]["text"])
    assert "experiment_id" in data
    assert data["status"] == "initialized"


@pytest.mark.asyncio
@pytest.mark.slow
async def test_train_step() -> None:
    """Test training step execution."""
    # Setup - use micro model for fast CI tests
    model_result = await _tool_create_model({"architecture": "gpt", "preset": "gpt-micro"})
    model_data = ast.literal_eval(model_result[0]["text"])
    model_id = model_data["model_id"]

    ds_result = await _tool_load_dataset({"dataset_name": "wikitext"})
    ds_data = ast.literal_eval(ds_result[0]["text"])
    dataset_id = ds_data["dataset_id"]

    trainer_result = await _tool_create_trainer(
        {
            "model_id": model_id,
            "dataset_id": dataset_id,
            "max_steps": 20,
        }
    )
    trainer_data = ast.literal_eval(trainer_result[0]["text"])
    experiment_id = trainer_data["experiment_id"]

    # Train - just a few steps for CI
    result = await _tool_train_step(
        {
            "experiment_id": experiment_id,
            "num_steps": 5,
        }
    )
    data = ast.literal_eval(result[0]["text"])
    assert data["steps_completed"] == 5
    assert data["current_step"] == 5
    assert "latest_loss" in data


@pytest.mark.asyncio
@pytest.mark.slow
async def test_get_training_status() -> None:
    """Test getting training status."""
    # Setup and train
    model_result = await _tool_create_model({"architecture": "gpt"})
    model_data = ast.literal_eval(model_result[0]["text"])
    model_id = model_data["model_id"]

    ds_result = await _tool_load_dataset({"dataset_name": "wikitext"})
    ds_data = ast.literal_eval(ds_result[0]["text"])
    dataset_id = ds_data["dataset_id"]

    trainer_result = await _tool_create_trainer(
        {
            "model_id": model_id,
            "dataset_id": dataset_id,
        }
    )
    trainer_data = ast.literal_eval(trainer_result[0]["text"])
    experiment_id = trainer_data["experiment_id"]

    await _tool_train_step({"experiment_id": experiment_id, "num_steps": 10})

    # Get status
    result = await _tool_get_training_status({"experiment_id": experiment_id})
    data = ast.literal_eval(result[0]["text"])
    assert data["current_step"] == 10
    assert "progress" in data
    assert "latest_loss" in data


@pytest.mark.asyncio
@pytest.mark.slow
async def test_evaluate_model() -> None:
    """Test model evaluation."""
    # Create model and dataset
    model_result = await _tool_create_model({"architecture": "gpt"})
    model_data = ast.literal_eval(model_result[0]["text"])
    model_id = model_data["model_id"]

    ds_result = await _tool_load_dataset({"dataset_name": "wikitext", "split": "validation"})
    ds_data = ast.literal_eval(ds_result[0]["text"])
    dataset_id = ds_data["dataset_id"]

    # Evaluate
    result = await _tool_evaluate_model(
        {
            "model_id": model_id,
            "dataset_id": dataset_id,
            "metrics": ["perplexity", "loss"],
        }
    )
    data = ast.literal_eval(result[0]["text"])
    assert "perplexity" in data
    assert "loss" in data


@pytest.mark.asyncio
@pytest.mark.slow
async def test_generate_text() -> None:
    """Test text generation."""
    # Create model and tokenizer
    model_result = await _tool_create_model({"architecture": "gpt"})
    model_data = ast.literal_eval(model_result[0]["text"])
    model_id = model_data["model_id"]

    tok_result = await _tool_create_tokenizer({"tokenizer_type": "tiktoken"})
    tok_data = ast.literal_eval(tok_result[0]["text"])
    tokenizer_id = tok_data["tokenizer_id"]

    # Generate
    result = await _tool_generate_text(
        {
            "model_id": model_id,
            "tokenizer_id": tokenizer_id,
            "prompt": "Once upon a time",
            "max_tokens": 50,
            "temperature": 0.8,
        }
    )
    data = ast.literal_eval(result[0]["text"])
    assert "prompt" in data
    assert "generated" in data
    assert data["tokens_generated"] == 50


@pytest.mark.asyncio
async def test_model_not_found() -> None:
    """Test error handling for missing model."""
    result = await _tool_get_model_config({"model_id": "model://nonexistent"})
    assert "Error" in result[0]["text"]


@pytest.mark.asyncio
@pytest.mark.slow
async def test_full_training_workflow() -> None:
    """Test complete training workflow."""
    # 1. Create model
    model_result = await _tool_create_model(
        {
            "architecture": "gpt",
            "n_layers": 4,
            "n_heads": 4,
            "d_model": 256,
        }
    )
    model_data = ast.literal_eval(model_result[0]["text"])
    model_id = model_data["model_id"]

    # 2. Create tokenizer
    tok_result = await _tool_create_tokenizer({"tokenizer_type": "tiktoken"})
    tok_data = ast.literal_eval(tok_result[0]["text"])
    tokenizer_id = tok_data["tokenizer_id"]

    # 3. Load and prepare dataset
    ds_result = await _tool_load_dataset({"dataset_name": "tinystories", "max_samples": 1000})
    ds_data = ast.literal_eval(ds_result[0]["text"])
    dataset_id = ds_data["dataset_id"]

    prep_result = await _tool_prepare_dataset(
        {
            "dataset_id": dataset_id,
            "tokenizer_id": tokenizer_id,
        }
    )
    prep_data = ast.literal_eval(prep_result[0]["text"])
    assert prep_data["prepared"] is True

    # 4. Create trainer
    trainer_result = await _tool_create_trainer(
        {
            "model_id": model_id,
            "dataset_id": dataset_id,
            "learning_rate": 3e-4,
            "max_steps": 100,
        }
    )
    trainer_data = ast.literal_eval(trainer_result[0]["text"])
    experiment_id = trainer_data["experiment_id"]

    # 5. Train
    train_result = await _tool_train_step(
        {
            "experiment_id": experiment_id,
            "num_steps": 100,
        }
    )
    train_data = ast.literal_eval(train_result[0]["text"])
    assert train_data["status"] == "completed"

    # 6. Evaluate
    eval_result = await _tool_evaluate_model(
        {
            "model_id": model_id,
            "dataset_id": dataset_id,
        }
    )
    eval_data = ast.literal_eval(eval_result[0]["text"])
    assert "perplexity" in eval_data

    # 7. Generate
    gen_result = await _tool_generate_text(
        {
            "model_id": model_id,
            "tokenizer_id": tokenizer_id,
            "prompt": "The quick brown fox",
        }
    )
    gen_data = ast.literal_eval(gen_result[0]["text"])
    assert "generated" in gen_data
