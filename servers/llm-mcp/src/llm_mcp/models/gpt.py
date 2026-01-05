"""GPT (Transformer Decoder) implementation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812


@dataclass
class GPTConfig:
    """GPT model configuration."""

    vocab_size: int = 50257
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    d_ff: int | None = None  # Defaults to 4 * d_model
    max_seq_len: int = 1024
    dropout: float = 0.1
    bias: bool = True
    flash_attention: bool = True  # Use flash attention if available

    def __post_init__(self) -> None:
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model

    @classmethod
    def from_preset(cls, preset: str) -> GPTConfig:
        """Create config from preset name."""
        presets = {
            "gpt2-small": cls(n_layers=12, n_heads=12, d_model=768),
            "gpt2-medium": cls(n_layers=24, n_heads=16, d_model=1024),
            "gpt2-large": cls(n_layers=36, n_heads=20, d_model=1280),
            "gpt2-xl": cls(n_layers=48, n_heads=25, d_model=1600),
            "gpt-mini": cls(n_layers=4, n_heads=4, d_model=256),  # For testing
            "gpt-micro": cls(n_layers=2, n_heads=2, d_model=128),  # For testing
        }
        if preset not in presets:
            msg = f"Unknown preset: {preset}. Available: {list(presets.keys())}"
            raise ValueError(msg)
        return presets[preset]


class LayerNorm(nn.Module):
    """Layer normalization with optional bias."""

    def __init__(self, ndim: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.d_model % config.n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads
        self.flash_attention = config.flash_attention

        # Key, Query, Value projections
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask
        self.mask: torch.Tensor
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(config.max_seq_len, config.max_seq_len), diagonal=1).bool(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.size()

        # Compute Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Try flash attention if available
        if self.flash_attention and hasattr(F, "scaled_dot_product_attention"):
            dropout_p = self.dropout.p if self.training else 0.0
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True
            )
        else:
            # Manual attention computation
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att = att.masked_fill(self.mask[:seq_len, :seq_len], float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.dropout(att)
            y = att @ v

        # Reshape back
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        # Output projection
        out: torch.Tensor = self.resid_dropout(self.c_proj(y))
        return out


class MLP(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.d_ff is not None
        self.c_fc = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.c_proj = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x, approximate="tanh")
        x = self.c_proj(x)
        out: torch.Tensor = self.dropout(x)
        return out


class TransformerBlock(nn.Module):
    """Transformer decoder block."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.d_model, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.d_model, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        out: torch.Tensor = x + self.mlp(self.ln_2(x))
        return out


class GPT(nn.Module):
    """GPT Language Model."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.d_model),
                "wpe": nn.Embedding(config.max_seq_len, config.d_model),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)]),
                "ln_f": LayerNorm(config.d_model, bias=config.bias),
            }
        )
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.transformer["wte"].weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Report parameter count (exclude position embeddings)
        n_params = sum(p.numel() for p in self.parameters())
        wpe = cast("nn.Embedding", self.transformer["wpe"])
        self._n_params = n_params - wpe.weight.numel()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def num_parameters(self) -> int:
        """Return number of parameters (excluding position embeddings)."""
        return self._n_params

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass."""
        device = idx.device
        _batch_size, seq_len = idx.size()

        if self.config.max_seq_len < seq_len:
            msg = f"Sequence length {seq_len} exceeds max {self.config.max_seq_len}"
            raise ValueError(msg)

        # Token + position embeddings
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device)
        tok_emb = self.transformer["wte"](idx)
        pos_emb = self.transformer["wpe"](pos)
        x = self.transformer["drop"](tok_emb + pos_emb)

        # Transformer blocks
        blocks = cast("nn.ModuleList", self.transformer["h"])
        for block in blocks:
            x = block(x)

        x = self.transformer["ln_f"](x)
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

        # Use fused AdamW if available
        fused_available = "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
        use_fused = fused_available and device_type == "cuda"

        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
