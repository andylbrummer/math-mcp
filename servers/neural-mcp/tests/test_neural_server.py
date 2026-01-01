"""Tests for Neural MCP server."""

import ast

import pytest
from neural_mcp.server import (
    _tool_define_model,
    _tool_evaluate_model,
    _tool_info,
    _tool_load_dataset,
    _tool_train_model,
)


@pytest.mark.asyncio
async def test_info() -> None:
    """Test info tool."""
    result = await _tool_info({})
    assert len(result) == 1


@pytest.mark.asyncio
async def test_define_model() -> None:
    """Test model definition."""
    result = await _tool_define_model({"architecture": "resnet18", "num_classes": 10})
    assert "model_id" in str(result[0]["text"])


@pytest.mark.asyncio
async def test_load_dataset() -> None:
    """Test dataset loading."""
    result = await _tool_load_dataset({"dataset_name": "CIFAR10", "split": "train"})
    assert "dataset_id" in str(result[0]["text"])


@pytest.mark.asyncio
async def test_train_model() -> None:
    """Test model training."""
    # Define model
    model_result = await _tool_define_model({"architecture": "resnet18", "num_classes": 10})
    model_data = ast.literal_eval(str(model_result[0]["text"]))
    model_id = model_data["model_id"]

    # Load dataset
    dataset_result = await _tool_load_dataset({"dataset_name": "CIFAR10", "split": "train"})
    dataset_data = ast.literal_eval(str(dataset_result[0]["text"]))
    dataset_id = dataset_data["dataset_id"]

    # Train
    result = await _tool_train_model({"model_id": model_id, "dataset_id": dataset_id, "epochs": 1})
    assert "experiment_id" in str(result[0]["text"])


@pytest.mark.asyncio
async def test_evaluate_model() -> None:
    """Test model evaluation."""
    # Define model and dataset
    model_result = await _tool_define_model({"architecture": "mobilenet"})
    model_data = ast.literal_eval(str(model_result[0]["text"]))
    model_id = model_data["model_id"]

    dataset_result = await _tool_load_dataset({"dataset_name": "MNIST"})
    dataset_data = ast.literal_eval(str(dataset_result[0]["text"]))
    dataset_id = dataset_data["dataset_id"]

    # Evaluate
    result = await _tool_evaluate_model({"model_id": model_id, "dataset_id": dataset_id})
    assert "accuracy" in str(result[0]["text"])
