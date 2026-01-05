"""Mamba (State Space Model) implementation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812


@dataclass
class MambaConfig:
    """Mamba model configuration."""

    vocab_size: int = 50257
    n_layers: int = 12
    d_model: int = 768
    d_state: int = 16  # SSM state dimension
    d_conv: int = 4  # Local convolution width
    expand: int = 2  # Expansion factor for inner dimension
    max_seq_len: int = 1024
    dropout: float = 0.1
    bias: bool = False

    @property
    def d_inner(self) -> int:
        """Inner dimension after expansion."""
        return self.expand * self.d_model

    @classmethod
    def from_preset(cls, preset: str) -> MambaConfig:
        """Create config from preset name."""
        presets = {
            "mamba-small": cls(n_layers=12, d_model=768, d_state=16),
            "mamba-medium": cls(n_layers=24, d_model=1024, d_state=16),
            "mamba-large": cls(n_layers=48, d_model=1536, d_state=16),
            "mamba-mini": cls(n_layers=4, d_model=256, d_state=8),  # For testing
            "mamba-micro": cls(n_layers=2, d_model=128, d_state=4),  # For testing
        }
        if preset not in presets:
            msg = f"Unknown preset: {preset}. Available: {list(presets.keys())}"
            raise ValueError(msg)
        return presets[preset]


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class MambaBlock(nn.Module):
    """Mamba block with selective state space model."""

    def __init__(self, config: MambaConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.d_state = config.d_state
        self.d_conv = config.d_conv
        self.d_inner = config.d_inner

        # Input projection
        self.in_proj = nn.Linear(config.d_model, config.d_inner * 2, bias=config.bias)

        # Convolution (local context)
        self.conv1d = nn.Conv1d(
            in_channels=config.d_inner,
            out_channels=config.d_inner,
            kernel_size=config.d_conv,
            padding=config.d_conv - 1,
            groups=config.d_inner,
            bias=config.bias,
        )

        # SSM parameters
        self.x_proj = nn.Linear(config.d_inner, config.d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(config.d_state, config.d_inner, bias=True)

        # A and D are learnable
        a_init = torch.arange(1, config.d_state + 1, dtype=torch.float32)
        a_init = a_init.repeat(config.d_inner, 1)
        self.a_log = nn.Parameter(torch.log(a_init))
        self.d_param = nn.Parameter(torch.ones(config.d_inner))

        # Output projection
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Mamba block."""
        _batch_size, seq_len, _dim = x.shape

        # Input projection
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        # Convolution
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :seq_len]  # Causal conv
        x = x.transpose(1, 2)
        x = F.silu(x)

        # SSM computation
        y = self._ssm(x)

        # Gating
        y = y * F.silu(z)

        # Output projection
        out: torch.Tensor = self.out_proj(y)
        return out

    def _ssm(self, x: torch.Tensor) -> torch.Tensor:
        """Selective State Space Model computation."""
        batch_size, seq_len, _ = x.shape
        a_matrix = -torch.exp(self.a_log)

        # Project to B, C, delta
        x_proj = self.x_proj(x)
        delta, b_proj, c_proj = x_proj.split([1, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta.squeeze(-1)))

        # Initialize state
        h = torch.zeros(batch_size, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []

        for i in range(seq_len):
            delta_i = delta[:, i, :].unsqueeze(-1)
            b_i = b_proj[:, i, :].unsqueeze(1)
            c_i = c_proj[:, i, :]
            x_i = x[:, i, :].unsqueeze(-1)

            # Discretized A and state update
            a_bar = torch.exp(delta_i * a_matrix)
            h = a_bar * h + delta_i * b_i * x_i

            # Output
            y = torch.einsum("bn,bdn->bd", c_i, h) + self.d_param * x[:, i, :]
            ys.append(y)

        return torch.stack(ys, dim=1)


class Mamba(nn.Module):
    """Mamba Language Model."""

    def __init__(self, config: MambaConfig) -> None:
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([MambaBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.embedding.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Report parameter count
        n_params = sum(p.numel() for p in self.parameters())
        self._n_params = n_params

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def num_parameters(self) -> int:
        """Return number of parameters."""
        return self._n_params

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass."""
        x = self.embedding(idx)

        for layer in self.layers:
            x = x + layer(x)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        """Generate tokens autoregressively."""
        for _ in range(max_new_tokens):
            # Crop context if needed
            max_len = self.config.max_seq_len
            idx_cond = idx if idx.size(1) <= max_len else idx[:, -max_len:]

            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def configure_optimizers(
        self,
        weight_decay: float = 0.1,
        learning_rate: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        device_type: str = "cuda",
    ) -> torch.optim.AdamW:
        """Configure AdamW optimizer with weight decay."""
        decay_params = []
        no_decay_params = []

        for _name, param in self.named_parameters():
            if param.requires_grad:
                if param.dim() >= 2:
                    decay_params.append(param)
                else:
                    no_decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        fused_available = "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
        use_fused = fused_available and device_type == "cuda"

        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
