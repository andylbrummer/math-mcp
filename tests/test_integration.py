"""Integration tests for cross-MCP workflows."""

import ast

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_math_to_quantum_workflow() -> None:
    """Test Math→Quantum workflow: Create potential with Math, use in Quantum simulation."""
    from math_mcp.server import _tool_create_array  # noqa: PLC0415
    from quantum_mcp.server import (  # noqa: PLC0415
        _tool_create_custom_potential,
        _tool_create_gaussian_wavepacket,
        _tool_solve_schrodinger,
    )

    # Step 1: Create potential array with Math MCP
    math_result = await _tool_create_array(
        {"shape": [256], "fill_type": "function", "function": "10*exp(-(x-128)**2/100)"}
    )

    ast.literal_eval(str(math_result[0]["text"]))
    # Note: In real cross-MCP, this would be passed via array:// URI
    # For now, we test the Quantum MCP independently

    # Step 2: Create potential in Quantum MCP
    potential_result = await _tool_create_custom_potential(
        {"grid_size": [256], "function": "10*exp(-(x-128)**2/100)"}
    )
    potential_data = ast.literal_eval(str(potential_result[0]["text"]))
    potential_id = potential_data["potential_id"]

    # Step 3: Create wavepacket
    psi_result = await _tool_create_gaussian_wavepacket(
        {"grid_size": [256], "position": [64], "momentum": [2.0], "width": 5.0}
    )
    psi_data = ast.literal_eval(str(psi_result[0]["text"]))
    psi = psi_data["wavefunction"]

    # Step 4: Run simulation
    sim_result = await _tool_solve_schrodinger(
        {"potential": potential_id, "initial_state": psi, "time_steps": 50, "dt": 0.1}
    )

    # Verify simulation completed
    sim_text = str(sim_result[0]["text"])
    assert "simulation_id" in sim_text
    assert "completed" in sim_text


@pytest.mark.integration
@pytest.mark.asyncio
async def test_all_mcps_info_tools() -> None:
    """Test that all MCPs have working info tools."""
    from math_mcp.server import _tool_info as math_info  # noqa: PLC0415
    from molecular_mcp.server import _tool_info as molecular_info  # noqa: PLC0415
    from neural_mcp.server import _tool_info as neural_info  # noqa: PLC0415
    from quantum_mcp.server import _tool_info as quantum_info  # noqa: PLC0415

    # Test each MCP's info tool
    math_result = await math_info({"topic": "overview"})
    assert len(math_result) == 1
    assert "categories" in str(math_result[0]["text"]) or "symbolic" in str(math_result[0]["text"])

    quantum_result = await quantum_info({"topic": "overview"})
    assert len(quantum_result) == 1

    molecular_result = await molecular_info({})
    assert len(molecular_result) == 1

    neural_result = await neural_info({})
    assert len(neural_result) == 1


@pytest.mark.integration
def test_shared_packages_integration() -> None:
    """Test that shared packages work together."""
    import numpy as np  # noqa: PLC0415
    from compute_core.arrays import ensure_array, get_array_module  # noqa: PLC0415
    from mcp_common import GPUManager  # noqa: PLC0415

    # Test GPU manager
    gpu = GPUManager.get_instance()
    assert gpu is not None

    # Test array module selection
    arr = ensure_array([1, 2, 3, 4, 5], use_gpu=False)
    xp = get_array_module(arr)
    assert xp is np


@pytest.mark.integration
def test_all_servers_installed() -> None:
    """Test that all server modules can be imported."""
    import math_mcp.server  # noqa: PLC0415
    import molecular_mcp.server  # noqa: PLC0415
    import neural_mcp.server  # noqa: PLC0415
    import quantum_mcp.server  # noqa: PLC0415

    assert math_mcp.server.app is not None
    assert quantum_mcp.server.app is not None
    assert molecular_mcp.server.app is not None
    assert neural_mcp.server.app is not None
