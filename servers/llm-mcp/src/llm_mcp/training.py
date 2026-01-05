"""Training infrastructure for LLM models."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""

    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    max_steps: int = 1000
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    mixed_precision: bool = True
    gradient_checkpointing: bool = False
    max_grad_norm: float = 1.0
    use_gpu: bool = False
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.use_gpu and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""

    loss: list[float] = field(default_factory=list)
    learning_rate: list[float] = field(default_factory=list)
    grad_norm: list[float] = field(default_factory=list)
    step: int = 0
    epoch: int = 0

    def update(self, loss: float, lr: float, grad_norm: float = 0.0) -> None:
        """Update metrics."""
        self.loss.append(loss)
        self.learning_rate.append(lr)
        self.grad_norm.append(grad_norm)
        self.step += 1

    def summary(self) -> dict[str, Any]:
        """Get metrics summary."""
        if not self.loss:
            return {"step": self.step, "epoch": self.epoch}
        return {
            "step": self.step,
            "epoch": self.epoch,
            "latest_loss": self.loss[-1],
            "min_loss": min(self.loss),
            "avg_loss_last_100": sum(self.loss[-100:]) / min(len(self.loss), 100),
            "avg_grad_norm": sum(self.grad_norm[-100:]) / max(len(self.grad_norm[-100:]), 1),
        }


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    warmup_steps: int,
    max_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Create learning rate scheduler."""
    if scheduler_type == "constant":
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    if scheduler_type == "linear":

        def linear_schedule(step: int) -> float:
            if step < warmup_steps:
                return step / warmup_steps
            return max(0.0, 1.0 - (step - warmup_steps) / (max_steps - warmup_steps))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, linear_schedule)

    if scheduler_type in ("cosine", "warmup_cosine"):

        def cosine_schedule(step: int) -> float:
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_schedule)

    msg = f"Unknown scheduler: {scheduler_type}"
    raise ValueError(msg)


class DataBatcher:
    """Simple data batcher for language model training."""

    def __init__(
        self,
        data: torch.Tensor,
        batch_size: int,
        seq_length: int,
        device: str = "cpu",
    ) -> None:
        self.data = data
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.device = device
        self.pos = 0

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pos + self.batch_size * self.seq_length + 1 > len(self.data):
            self.pos = 0
            raise StopIteration

        # Get batch
        batch_data = self.data[self.pos : self.pos + self.batch_size * self.seq_length + 1]
        x = batch_data[:-1].view(self.batch_size, self.seq_length)
        y = batch_data[1:].view(self.batch_size, self.seq_length)

        self.pos += self.batch_size * self.seq_length

        return x.to(self.device), y.to(self.device)

    def __len__(self) -> int:
        return len(self.data) // (self.batch_size * self.seq_length)


class Trainer:
    """Training loop for language models."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
    ) -> None:
        self.model = model.to(config.device)
        self.config = config
        self.metrics = TrainingMetrics()

        # Configure optimizer
        if hasattr(model, "configure_optimizers"):
            self.optimizer: torch.optim.Optimizer = model.configure_optimizers(  # type: ignore[operator]
                weight_decay=config.weight_decay,
                learning_rate=config.learning_rate,
                device_type=config.device,
            )
        else:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )

        # Configure scheduler
        self.scheduler = get_lr_scheduler(
            self.optimizer,
            config.scheduler,
            config.warmup_steps,
            config.max_steps,
        )

        # Mixed precision
        self.scaler = torch.amp.GradScaler() if config.mixed_precision else None
        self.autocast_dtype = torch.float16 if config.device == "cuda" else torch.bfloat16

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> float:
        """Execute a single training step."""
        self.model.train()

        # Mixed precision forward pass
        if self.config.mixed_precision:
            with torch.autocast(device_type=self.config.device, dtype=self.autocast_dtype):
                _logits, loss = self.model(x, targets=y)
        else:
            _logits, loss = self.model(x, targets=y)

        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return float(loss.item()) * self.config.gradient_accumulation_steps

    def optimizer_step(self) -> float:
        """Execute optimizer step with gradient clipping."""
        # Unscale gradients for clipping
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)

        # Clip gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
        )

        # Optimizer step
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()

        return float(grad_norm)

    def train(
        self,
        data_batcher: DataBatcher,
        num_steps: int | None = None,
    ) -> TrainingMetrics:
        """Run training loop."""
        max_steps = num_steps or self.config.max_steps
        accumulated_loss = 0.0
        accum_step = 0

        for step, (x, y) in enumerate(data_batcher):
            if step >= max_steps:
                break

            # Training step
            loss = self.train_step(x, y)
            accumulated_loss += loss
            accum_step += 1

            # Gradient accumulation
            if accum_step >= self.config.gradient_accumulation_steps:
                grad_norm = self.optimizer_step()
                avg_loss = accumulated_loss / self.config.gradient_accumulation_steps

                # Update metrics
                current_lr = self.scheduler.get_last_lr()[0]
                self.metrics.update(avg_loss, current_lr, grad_norm)

                # Reset accumulation
                accumulated_loss = 0.0
                accum_step = 0

                if step % 100 == 0:
                    logger.info("Step %d: loss=%.4f, lr=%.2e", step, avg_loss, current_lr)

        return self.metrics

    @torch.no_grad()
    def evaluate(
        self,
        data_batcher: DataBatcher,
        max_batches: int = 50,
    ) -> dict[str, float]:
        """Evaluate model on data."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for i, (x, y) in enumerate(data_batcher):
            if i >= max_batches:
                break

            with torch.autocast(device_type=self.config.device, dtype=self.autocast_dtype):
                _, loss = self.model(x, targets=y)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        perplexity = math.exp(avg_loss)

        return {
            "loss": round(avg_loss, 4),
            "perplexity": round(perplexity, 2),
            "num_batches": num_batches,
        }


def create_synthetic_data(
    num_tokens: int,
    vocab_size: int = 50257,
    seed: int = 42,
) -> torch.Tensor:
    """Create synthetic data for testing training loop."""
    torch.manual_seed(seed)
    return torch.randint(0, vocab_size, (num_tokens,))


def estimate_memory_usage(
    model: nn.Module,
    batch_size: int,
    seq_length: int,
    mixed_precision: bool = True,
) -> dict[str, float]:
    """Estimate memory usage for training."""
    # Parameter memory
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    # Gradient memory (same as parameters)
    grad_bytes = param_bytes

    # Optimizer state (AdamW: 2 states per parameter)
    optimizer_bytes = param_bytes * 2

    # Activation memory (rough estimate)
    n_params = sum(p.numel() for p in model.parameters())
    bytes_per_param = 2 if mixed_precision else 4
    activation_bytes = batch_size * seq_length * (n_params // 1000) * bytes_per_param

    total_bytes = param_bytes + grad_bytes + optimizer_bytes + activation_bytes

    return {
        "parameters_mb": param_bytes / 1024**2,
        "gradients_mb": grad_bytes / 1024**2,
        "optimizer_mb": optimizer_bytes / 1024**2,
        "activations_mb": activation_bytes / 1024**2,
        "total_mb": total_bytes / 1024**2,
        "total_gb": total_bytes / 1024**3,
    }
