"""Configuration loader using KDL format."""

import logging
from pathlib import Path
from typing import Any

import kdl
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class LimitsConfig(BaseModel):
    """Resource limits configuration."""

    max_array_size: int = Field(default=100_000_000, description="Maximum array elements")
    max_particles: int = Field(default=10_000_000, description="Maximum particles in MD")
    max_time_steps: int = Field(default=1_000_000, description="Maximum simulation timesteps")
    max_epochs: int = Field(default=1000, description="Maximum training epochs")


class GPUConfig(BaseModel):
    """GPU configuration."""

    memory_fraction: float = Field(default=0.8, ge=0.0, le=1.0, description="GPU memory fraction")
    backend: str = Field(default="cuda", description="GPU backend (cuda/rocm)")


class MCPConfig(BaseModel):
    """MCP server configuration."""

    server_name: str
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    gpu: GPUConfig = Field(default_factory=GPUConfig)
    extra: dict[str, Any] = Field(default_factory=dict)


def load_config(config_path: Path) -> MCPConfig:
    """Load configuration from KDL file."""
    if not config_path.exists():
        logger.warning("Config file not found: %s, using defaults", config_path)
        return MCPConfig(server_name="unknown")

    try:
        with config_path.open() as f:
            doc = kdl.parse(f.read())

        # Extract server name
        server_name = "unknown"
        for node in doc:
            if node.name == "server":
                server_name = str(node.args[0]) if node.args else "unknown"
                break

        # Extract limits
        limits_dict: dict[str, Any] = {}
        for node in doc:
            if node.name == "limits":
                for prop in node.props:
                    # Convert kebab-case to snake_case
                    key = prop.replace("-", "_")
                    limits_dict[key] = node.props[prop]

        # Extract GPU config
        gpu_dict: dict[str, Any] = {}
        for node in doc:
            if node.name == "gpu":
                for prop in node.props:
                    key = prop.replace("-", "_")
                    gpu_dict[key] = node.props[prop]

        # Build config
        config = MCPConfig(
            server_name=server_name,
            limits=LimitsConfig(**limits_dict) if limits_dict else LimitsConfig(),
            gpu=GPUConfig(**gpu_dict) if gpu_dict else GPUConfig(),
        )

        logger.info("Loaded config from %s: %s", config_path, config.server_name)
    except Exception:
        logger.exception("Failed to load config from %s", config_path)
        raise
    else:
        return config
