"""Tests for Quantum MCP server."""

import ast

import pytest
from quantum_mcp.server import (
    _potentials,
    _tool_analyze_wavefunction,
    _tool_create_custom_potential,
    _tool_create_gaussian_wavepacket,
    _tool_create_lattice_potential,
    _tool_create_plane_wave,
    _tool_info,
    _tool_solve_schrodinger,
    _wavefunctions,
)


@pytest.mark.asyncio
async def test_info() -> None:
    """Test info tool."""
    result = await _tool_info({"topic": "overview"})
    assert len(result) == 1
    assert "categories" in str(result[0]["text"])


@pytest.mark.asyncio
async def test_create_lattice_potential_1d() -> None:
    """Test creating 1D lattice potential."""
    result = await _tool_create_lattice_potential(
        {"lattice_type": "square", "grid_size": [100], "depth": 10.0, "spacing": 10.0}
    )
    assert "potential_id" in str(result[0]["text"])


@pytest.mark.asyncio
async def test_create_lattice_potential_2d() -> None:
    """Test creating 2D lattice potential."""
    result = await _tool_create_lattice_potential(
        {"lattice_type": "square", "grid_size": [50, 50], "depth": 10.0, "spacing": 10.0}
    )
    assert "potential_id" in str(result[0]["text"])


@pytest.mark.asyncio
async def test_create_custom_potential() -> None:
    """Test creating custom potential."""
    result = await _tool_create_custom_potential(
        {"grid_size": [100], "function": "10*exp(-(x-50)**2/100)"}
    )
    assert "potential_id" in str(result[0]["text"])


@pytest.mark.asyncio
async def test_create_gaussian_wavepacket() -> None:
    """Test creating Gaussian wavepacket."""
    result = await _tool_create_gaussian_wavepacket(
        {"grid_size": [100], "position": [50], "momentum": [1.0], "width": 5.0}
    )
    assert "wavefunction_id" in str(result[0]["text"])


@pytest.mark.asyncio
async def test_create_plane_wave() -> None:
    """Test creating plane wave."""
    result = await _tool_create_plane_wave({"grid_size": [100], "momentum": [2.0]})
    assert "wavefunction_id" in str(result[0]["text"])


@pytest.mark.asyncio
async def test_solve_schrodinger_small() -> None:
    """Test solving Schrödinger equation (small, synchronous)."""
    # Create potential
    pot_result = await _tool_create_custom_potential(
        {"grid_size": [256], "function": "10*exp(-(x-128)**2/100)"}
    )
    pot_id = str(pot_result[0]["text"]).split("'potential_id': '")[1].split("'")[0]

    # Create wavepacket - now returns wavefunction_id
    psi_result = await _tool_create_gaussian_wavepacket(
        {"grid_size": [256], "position": [64], "momentum": [2.0], "width": 5.0}
    )
    psi_data = ast.literal_eval(str(psi_result[0]["text"]))
    psi_id = psi_data["wavefunction_id"]

    # Solve using wavefunction_id
    result = await _tool_solve_schrodinger(
        {"potential": pot_id, "initial_state": psi_id, "time_steps": 50, "dt": 0.1}
    )

    assert "simulation_id" in str(result[0]["text"])
    assert "completed" in str(result[0]["text"])


@pytest.mark.asyncio
async def test_analyze_wavefunction() -> None:
    """Test wavefunction analysis."""
    # Create simple wavefunction - now returns wavefunction_id
    psi_result = await _tool_create_gaussian_wavepacket(
        {"grid_size": [100], "position": [50], "momentum": [0.0], "width": 5.0}
    )
    psi_data = ast.literal_eval(str(psi_result[0]["text"]))
    psi_id = psi_data["wavefunction_id"]

    # Get the actual wavefunction from storage for analysis
    wf_key = psi_id.replace("wavefunction://", "")
    psi = _wavefunctions[wf_key].tolist()

    result = await _tool_analyze_wavefunction({"wavefunction": psi, "dx": 1.0})
    text = str(result[0]["text"])
    assert "position" in text
    assert "momentum" in text
    assert "energy" in text


@pytest.mark.asyncio
async def test_potential_cache() -> None:
    """Test that potentials are cached."""
    initial_size = len(_potentials)
    await _tool_create_lattice_potential(
        {"lattice_type": "square", "grid_size": [100], "depth": 10.0}
    )
    assert len(_potentials) == initial_size + 1
