"""Deep-learning trainer for ``SequenceModel`` subclasses.

Design constraints:

* One sample == one full 1000-step sequence. The trainer sees
  ``(B, T, F)`` -> ``(B, T, K)`` and applies the loss only on
  ``mask == True`` positions (excludes warm-up).

* Optimiser: AdamW. Cosine LR schedule with warm-up. Mixed-precision
  (``torch.cuda.amp``) when CUDA is available.

* Validation: every ``eval_every`` epochs, run inference on the held-out
  set, compute weighted Pearson (numpy reference for parity with the
  scorer), record metrics. Save the best checkpoint by weighted Pearson.

* Logging: rich-printed table at end of each epoch + a CSV history.

The trainer never touches files on its own — it returns the trained
model and a `metrics_history` list; persistence is the caller's job (so
the same trainer can be invoked from a notebook or a CLI).
"""

from __future__ import annotations

import math
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from rich.console import Console
from torch.utils.data import DataLoader

from data import SequenceDataset, pad_collate
from losses import build_loss
from metrics import summary, weighted_pearson_per_target
from models.base import SequenceModel

CONSOLE = Console()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TrainerConfig:
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    grad_clip: float | None = 1.0
    eval_every: int = 1
    num_workers: int = 0
    pin_memory: bool = True
    device: str = "auto"           # auto | cuda | cpu
    amp: bool = True
    seed: int = 0
    loss_name: str = "weighted_pearson"
    loss_kwargs: dict[str, Any] = field(default_factory=dict)
    early_stopping_patience: int = 5

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    def __init__(
        self,
        model: SequenceModel,
        config: TrainerConfig,
        train_ds: SequenceDataset,
        val_ds: SequenceDataset | None = None,
    ):
        self.model = model
        self.config = config
        self.device = config.resolve_device()
        self.model.to(self.device)
        self.train_ds = train_ds
        self.val_ds = val_ds

        torch.manual_seed(config.seed)

        self.train_loader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory and self.device.type == "cuda",
            collate_fn=pad_collate,
            drop_last=True,
        )
        self.val_loader = (
            None
            if val_ds is None
            else DataLoader(
                val_ds,
                batch_size=max(1, config.batch_size),
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory and self.device.type == "cuda",
                collate_fn=pad_collate,
            )
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        total_steps = max(1, len(self.train_loader) * max(1, config.epochs))
        warmup_steps = max(1, len(self.train_loader) * max(0, config.warmup_epochs))

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        self.loss_fn = build_loss(config.loss_name, **config.loss_kwargs).to(self.device)

        self.amp = config.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)

        self.best_score: float = -float("inf")
        self.best_state: dict | None = None
        self.history: list[dict] = []
        self._epochs_since_improvement = 0

    # ------------------------------------------------------------------
    # Training loop.
    # ------------------------------------------------------------------
    def fit(self) -> dict:
        for epoch in range(1, self.config.epochs + 1):
            t0 = time.time()
            train_loss = self._train_one_epoch(epoch)
            val_info: dict[str, Any] = {}
            if self.val_loader is not None and (epoch % self.config.eval_every == 0):
                val_info = self.evaluate(self.val_loader)
            elapsed = time.time() - t0

            self.history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    "elapsed_s": float(elapsed),
                    **{f"val_{k}": v for k, v in val_info.items() if k != "per_target"},
                    "val_per_target": val_info.get("per_target", {}),
                    "lr": float(self.optimizer.param_groups[0]["lr"]),
                }
            )

            score = val_info.get("weighted_pearson", -float("inf"))
            improved = score > self.best_score
            if improved:
                self.best_score = score
                self.best_state = {
                    k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                }
                self._epochs_since_improvement = 0
            else:
                self._epochs_since_improvement += 1

            CONSOLE.print(
                f"[bold]epoch {epoch:>3d}[/] | loss {train_loss:.5f}"
                f" | val_corr {score:+.4f}"
                f" | best {self.best_score:+.4f}"
                f" | lr {self.optimizer.param_groups[0]['lr']:.2e}"
                f" | {elapsed:.1f}s"
                + ("  [green]*[/]" if improved else "")
            )

            if (
                self.config.early_stopping_patience > 0
                and self._epochs_since_improvement >= self.config.early_stopping_patience
            ):
                CONSOLE.print(
                    f"[yellow]early stopping at epoch {epoch} "
                    f"(no improvement in {self._epochs_since_improvement} epochs)[/]"
                )
                break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        return {"best_score": self.best_score, "history": self.history}

    # ------------------------------------------------------------------
    # Internals.
    # ------------------------------------------------------------------
    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        n_samples = 0
        loss_sum = 0.0
        amp_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=torch.float16)
            if self.amp
            else nullcontext()
        )
        for batch in self.train_loader:
            features = batch["features"].to(self.device, non_blocking=True)
            targets = batch["targets"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                preds = self.model(features)
                loss = self.loss_fn(targets, preds, mask=mask)

            if self.amp:
                self.scaler.scale(loss).backward()
                if self.config.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.config.grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    )
                self.optimizer.step()

            self.scheduler.step()
            bs = features.size(0)
            n_samples += bs
            loss_sum += float(loss.item()) * bs

        return loss_sum / max(1, n_samples)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        self.model.eval()
        all_preds = []
        all_targets = []
        for batch in loader:
            features = batch["features"].to(self.device, non_blocking=True)
            targets = batch["targets"]
            mask = batch["mask"].bool()
            preds = self.model(features).cpu()
            # Keep only scored positions; targets/mask are still on CPU.
            for b in range(features.size(0)):
                m = mask[b]
                if m.any():
                    all_preds.append(preds[b][m].numpy())
                    all_targets.append(targets[b][m].numpy())
        if not all_preds:
            return {"weighted_pearson": 0.0, "per_target": {}, "n_samples": 0}
        y_pred = np.concatenate(all_preds, axis=0)
        y_true = np.concatenate(all_targets, axis=0)
        return summary(y_true, y_pred)


# ---------------------------------------------------------------------------
# Convenience function used by the CLI.
# ---------------------------------------------------------------------------


def train_sequence_model(
    model: SequenceModel,
    train_ds: SequenceDataset,
    val_ds: SequenceDataset | None,
    config: TrainerConfig,
) -> dict:
    return Trainer(model, config, train_ds, val_ds).fit()
