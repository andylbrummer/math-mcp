"""Tests for new Neural MCP tools."""

import ast

import pytest
from neural_mcp.server import (
    _tool_compute_metrics,
    _tool_confusion_matrix,
    _tool_create_dataloader,
    _tool_define_model,
    _tool_export_model,
    _tool_get_model_summary,
    _tool_load_dataset,
    _tool_load_pretrained,
    _tool_plot_training_curves,
    _tool_train_model,
    _tool_tune_hyperparameters,
)


@pytest.mark.asyncio
async def test_get_model_summary() -> None:
    """Test model summary."""
    model_result = await _tool_define_model({"architecture": "resnet18"})
    data = ast.literal_eval(str(model_result[0]["text"]))
    model_id = data["model_id"]

    result = await _tool_get_model_summary({"model_id": model_id})
    assert "architecture" in str(result[0]["text"])
    assert "total_params" in str(result[0]["text"])


@pytest.mark.asyncio
async def test_create_dataloader() -> None:
    """Test dataloader creation."""
    dataset_result = await _tool_load_dataset({"dataset_name": "CIFAR10"})
    data = ast.literal_eval(str(dataset_result[0]["text"]))
    dataset_id = data["dataset_id"]

    result = await _tool_create_dataloader({"dataset_id": dataset_id, "batch_size": 64})
    assert "dataloader_id" in str(result[0]["text"])


@pytest.mark.asyncio
async def test_tune_hyperparameters() -> None:
    """Test hyperparameter tuning."""
    model_result = await _tool_define_model({"architecture": "resnet18"})
    dataset_result = await _tool_load_dataset({"dataset_name": "CIFAR10"})

    model_data = ast.literal_eval(str(model_result[0]["text"]))
    dataset_data = ast.literal_eval(str(dataset_result[0]["text"]))

    result = await _tool_tune_hyperparameters(
        {
            "model_id": model_data["model_id"],
            "dataset_id": dataset_data["dataset_id"],
            "param_grid": {"lr": [0.001, 0.01]},
            "n_trials": 5,
        }
    )
    assert "best_params" in str(result[0]["text"])


@pytest.mark.asyncio
async def test_plot_training_curves() -> None:
    """Test training curves plotting."""
    model_result = await _tool_define_model({"architecture": "mobilenet"})
    dataset_result = await _tool_load_dataset({"dataset_name": "MNIST"})

    model_data = ast.literal_eval(str(model_result[0]["text"]))
    dataset_data = ast.literal_eval(str(dataset_result[0]["text"]))

    train_result = await _tool_train_model(
        {
            "model_id": model_data["model_id"],
            "dataset_id": dataset_data["dataset_id"],
            "epochs": 1,
        }
    )
    train_data = ast.literal_eval(str(train_result[0]["text"]))
    experiment_id = train_data["experiment_id"]

    result = await _tool_plot_training_curves({"experiment_id": experiment_id})
    assert "output_path" in str(result[0]["text"])


@pytest.mark.asyncio
async def test_confusion_matrix() -> None:
    """Test confusion matrix generation."""
    model_result = await _tool_define_model({"architecture": "resnet18", "num_classes": 10})
    dataset_result = await _tool_load_dataset({"dataset_name": "CIFAR10"})

    model_data = ast.literal_eval(str(model_result[0]["text"]))
    dataset_data = ast.literal_eval(str(dataset_result[0]["text"]))

    result = await _tool_confusion_matrix(
        {
            "model_id": model_data["model_id"],
            "dataset_id": dataset_data["dataset_id"],
        }
    )
    assert "confusion_matrix" in str(result[0]["text"])


@pytest.mark.asyncio
async def test_export_model() -> None:
    """Test model export."""
    model_result = await _tool_define_model({"architecture": "resnet18"})
    data = ast.literal_eval(str(model_result[0]["text"]))
    model_id = data["model_id"]

    result = await _tool_export_model(
        {
            "model_id": model_id,
            "format": "onnx",
        }
    )
    assert "exported" in str(result[0]["text"])


@pytest.mark.asyncio
async def test_load_pretrained() -> None:
    """Test loading pretrained model."""
    result = await _tool_load_pretrained(
        {
            "model_name": "resnet50",
            "source": "torchvision",
        }
    )
    assert "model_id" in str(result[0]["text"])


@pytest.mark.asyncio
async def test_compute_metrics() -> None:
    """Test advanced metrics computation."""
    model_result = await _tool_define_model({"architecture": "mobilenet"})
    dataset_result = await _tool_load_dataset({"dataset_name": "MNIST"})

    model_data = ast.literal_eval(str(model_result[0]["text"]))
    dataset_data = ast.literal_eval(str(dataset_result[0]["text"]))

    result = await _tool_compute_metrics(
        {
            "model_id": model_data["model_id"],
            "dataset_id": dataset_data["dataset_id"],
        }
    )
    assert "accuracy" in str(result[0]["text"])
    assert "f1" in str(result[0]["text"])
