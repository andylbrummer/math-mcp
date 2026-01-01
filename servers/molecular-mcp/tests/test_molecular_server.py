"""Tests for Molecular MCP server."""

import ast

import pytest
from molecular_mcp.server import (
    _tool_add_potential,
    _tool_create_particles,
    _tool_get_trajectory,
    _tool_info,
    _tool_run_md,
)


@pytest.mark.asyncio
async def test_info() -> None:
    """Test info tool."""
    result = await _tool_info({})
    assert len(result) == 1


@pytest.mark.asyncio
async def test_create_particles() -> None:
    """Test particle creation."""
    result = await _tool_create_particles(
        {"n_particles": 100, "box_size": [10, 10, 10], "temperature": 1.0}
    )
    assert "system_id" in str(result[0]["text"])


@pytest.mark.asyncio
async def test_add_potential() -> None:
    """Test adding potential."""
    # Create system first
    create_result = await _tool_create_particles({"n_particles": 100, "box_size": [10, 10, 10]})
    data = ast.literal_eval(str(create_result[0]["text"]))
    system_id = data["system_id"]

    # Add potential
    result = await _tool_add_potential({"system_id": system_id, "potential_type": "lennard_jones"})
    assert "potential" in str(result[0]["text"])


@pytest.mark.asyncio
async def test_run_md() -> None:
    """Test MD simulation."""
    # Create system
    create_result = await _tool_create_particles({"n_particles": 100, "box_size": [10, 10, 10]})
    data = ast.literal_eval(str(create_result[0]["text"]))
    system_id = data["system_id"]

    # Run MD
    result = await _tool_run_md({"system_id": system_id, "n_steps": 100, "dt": 0.001})
    assert "trajectory_id" in str(result[0]["text"])


@pytest.mark.asyncio
async def test_get_trajectory() -> None:
    """Test trajectory retrieval."""
    # Create and run system
    create_result = await _tool_create_particles({"n_particles": 50, "box_size": [10, 10, 10]})
    data = ast.literal_eval(str(create_result[0]["text"]))
    system_id = data["system_id"]

    run_result = await _tool_run_md({"system_id": system_id, "n_steps": 50, "dt": 0.001})
    run_data = ast.literal_eval(str(run_result[0]["text"]))
    trajectory_id = run_data["trajectory_id"]

    # Get trajectory
    result = await _tool_get_trajectory({"trajectory_id": trajectory_id})
    assert "frames" in str(result[0]["text"])
