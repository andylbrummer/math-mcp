"""Tests for new Molecular MCP tools."""

import ast

import pytest
from molecular_mcp.server import (
    _tool_analyze_temperature,
    _tool_compute_msd,
    _tool_create_particles,
    _tool_detect_phase_transition,
    _tool_run_md,
    _tool_run_npt,
    _tool_run_nvt,
)


@pytest.mark.asyncio
async def test_run_nvt() -> None:
    """Test NVT ensemble simulation."""
    # Create system
    create_result = await _tool_create_particles({"n_particles": 50, "box_size": [10, 10, 10]})
    data = ast.literal_eval(str(create_result[0]["text"]))
    system_id = data["system_id"]

    # Run NVT
    result = await _tool_run_nvt({"system_id": system_id, "n_steps": 100, "temperature": 1.0})
    assert "NVT" in str(result[0]["text"])
    assert "trajectory_id" in str(result[0]["text"])


@pytest.mark.asyncio
async def test_run_npt() -> None:
    """Test NPT ensemble simulation."""
    create_result = await _tool_create_particles({"n_particles": 50, "box_size": [10, 10, 10]})
    data = ast.literal_eval(str(create_result[0]["text"]))
    system_id = data["system_id"]

    result = await _tool_run_npt(
        {"system_id": system_id, "n_steps": 100, "temperature": 1.0, "pressure": 1.0}
    )
    assert "NPT" in str(result[0]["text"])


@pytest.mark.asyncio
async def test_compute_msd() -> None:
    """Test MSD computation."""
    # Create and run system
    create_result = await _tool_create_particles({"n_particles": 50, "box_size": [10, 10, 10]})
    data = ast.literal_eval(str(create_result[0]["text"]))
    system_id = data["system_id"]

    run_result = await _tool_run_md({"system_id": system_id, "n_steps": 100, "dt": 0.001})
    run_data = ast.literal_eval(str(run_result[0]["text"]))
    trajectory_id = run_data["trajectory_id"]

    # Compute MSD
    result = await _tool_compute_msd({"trajectory_id": trajectory_id})
    assert "msd" in str(result[0]["text"])


@pytest.mark.asyncio
async def test_analyze_temperature() -> None:
    """Test temperature analysis."""
    create_result = await _tool_create_particles({"n_particles": 50, "box_size": [10, 10, 10]})
    data = ast.literal_eval(str(create_result[0]["text"]))
    system_id = data["system_id"]

    run_result = await _tool_run_md({"system_id": system_id, "n_steps": 50})
    run_data = ast.literal_eval(str(run_result[0]["text"]))
    trajectory_id = run_data["trajectory_id"]

    result = await _tool_analyze_temperature({"trajectory_id": trajectory_id})
    assert "temperature" in str(result[0]["text"])


@pytest.mark.asyncio
async def test_detect_phase_transition() -> None:
    """Test phase transition detection."""
    create_result = await _tool_create_particles({"n_particles": 50, "box_size": [10, 10, 10]})
    data = ast.literal_eval(str(create_result[0]["text"]))
    system_id = data["system_id"]

    run_result = await _tool_run_md({"system_id": system_id, "n_steps": 50})
    run_data = ast.literal_eval(str(run_result[0]["text"]))
    trajectory_id = run_data["trajectory_id"]

    result = await _tool_detect_phase_transition({"trajectory_id": trajectory_id})
    assert "phase_detected" in str(result[0]["text"])
