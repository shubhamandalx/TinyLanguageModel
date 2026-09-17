"""
trainer.py
==========

The main training engine, wrapped in a small `Trainer` class, plus
the learning-rate schedule and checkpoint save/load functions used
by both `train.py` and `generate.py`.
"""

import math
import time
from dataclasses import asdict
from typing import Tuple

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from config import GPTConfig
from model import ModernGPT
from optimizer import Muon, build_optimizers


# ====================================================================
# Learning-rate schedule: linear warmup, then cosine decay
# ====================================================================


def get_lr_multiplier(step: int, warmup_steps: int, max_steps: int) -> float:
    """
    Returns a multiplier in [0, 1] to scale the base learning rate:

        step < warmup_steps  -> linear ramp from ~0 to 1
        otherwise             -> cosine decay from 1 to 0
    """

    if step < warmup_steps:
        return (step + 1) / warmup_steps

    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)

    return 0.5 * (1.0 + math.cos(math.pi * progress))


def set_learning_rate(optimizer: torch.optim.Optimizer, base_lr: float, multiplier: float) -> float:
    """
    Apply `multiplier` to `base_lr` and set it on every parameter
    group of `optimizer`. Returns the resulting learning rate.
    """

    new_lr = base_lr * multiplier
    for group in optimizer.param_groups:
        group["lr"] = new_lr

    return new_lr


# ====================================================================
# Checkpointing
# ====================================================================


def save_checkpoint(
    model: ModernGPT,
    config: GPTConfig,
    tokenizer_name: str,
    path: str,
) -> None:
    """
    Save model weights + config + tokenizer name to `path`.

    Saving the config and tokenizer name alongside the weights
    means `load_checkpoint` can reconstruct the exact model
    architecture without the caller needing to know it in advance.
    """

    cpu_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

    checkpoint = {
        "model_state_dict": cpu_state_dict,
        "gpt_config": asdict(config),
        "tokenizer_name": tokenizer_name,
    }

    torch.save(checkpoint, path)
    print(f"Checkpoint saved to: {path}")


def load_checkpoint(path: str, device: torch.device):
    """
    Load a checkpoint produced by `save_checkpoint`.

    Returns: (model, config, tokenizer)
    """

    checkpoint = torch.load(path, map_location=device)

    config = GPTConfig(**checkpoint["gpt_config"])
    tokenizer = tiktoken.get_encoding(checkpoint["tokenizer_name"])

    model = ModernGPT(config, tokenizer.n_vocab)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    return model, config, tokenizer


# ====================================================================
# Trainer
# ====================================================================


class Trainer:
    """
    Owns the optimizers, the training loop, validation, and
    checkpoint saving for a `ModernGPT` model.

    Preserves the original training algorithm:
        - gradient accumulation
        - mixed-precision FP16 on CUDA
        - gradient clipping
        - Muon step for matrix weights + AdamW step for the rest
        - linear-warmup / cosine-decay learning-rate schedule
    """

    def __init__(
        self,
        model: ModernGPT,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: GPTConfig,
        device: torch.device,
        tokenizer_name: str,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.tokenizer_name = tokenizer_name

        self.muon_optimizer: Muon
        self.adamw_optimizer: AdamW
        self.muon_optimizer, self.adamw_optimizer = build_optimizers(model, config)

        self.use_amp = device.type == "cuda"
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

        self.optimizer_step = 0

    def train(self) -> ModernGPT:
        """
        Run the main pretraining loop for `config.max_steps`
        optimizer updates.
        """

        config = self.config

        print("AMP enabled:", self.use_amp)
        print("Gradient accumulation:", config.gradient_accumulation_steps)
        print("Effective batch size:", config.batch_size * config.gradient_accumulation_steps)

        train_iter = iter(self.train_loader)
        self.model.train()

        start_time = time.time()

        while self.optimizer_step < config.max_steps:
            mean_loss, mean_aux, train_iter = self._run_optimizer_step(train_iter)

            self.optimizer_step += 1

            if self.optimizer_step % config.print_every == 0 or self.optimizer_step == 1:
                self._log_progress(mean_loss, mean_aux, start_time)

        print("\nPretraining finished.")
        return self.model

    def _run_optimizer_step(self, train_iter):
        """
        One full optimizer update: `gradient_accumulation_steps`
        micro-batches of forward/backward, followed by a single
        Muon + AdamW step.
        """

        config = self.config

        self.muon_optimizer.zero_grad(set_to_none=True)
        self.adamw_optimizer.zero_grad(set_to_none=True)

        accumulated_loss = 0.0
        accumulated_aux = 0.0

        for _ in range(config.gradient_accumulation_steps):
            try:
                input_ids, targets = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                input_ids, targets = next(train_iter)

            input_ids = input_ids.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=torch.float16):
                logits, moe_aux_loss = self.model(input_ids)

                language_loss = F.cross_entropy(
                    logits.reshape(-1, self.model.vocab_size), targets.reshape(-1)
                )

                total_loss = language_loss + config.moe_aux_loss_weight * moe_aux_loss
                total_loss = total_loss / config.gradient_accumulation_steps

            self.scaler.scale(total_loss).backward()

            accumulated_loss += language_loss.item()
            accumulated_aux += moe_aux_loss.item()

        if self.use_amp:
            self.scaler.unscale_(self.muon_optimizer)
            self.scaler.unscale_(self.adamw_optimizer)

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.grad_clip)

        multiplier = get_lr_multiplier(self.optimizer_step, config.warmup_steps, config.max_steps)
        self.current_muon_lr = set_learning_rate(self.muon_optimizer, config.muon_lr, multiplier)
        self.current_adamw_lr = set_learning_rate(self.adamw_optimizer, config.adamw_lr, multiplier)

        self.scaler.step(self.muon_optimizer)
        self.scaler.step(self.adamw_optimizer)
        self.scaler.update()

        mean_loss = accumulated_loss / config.gradient_accumulation_steps
        mean_aux = accumulated_aux / config.gradient_accumulation_steps

        return mean_loss, mean_aux, train_iter

    def _log_progress(self, mean_loss: float, mean_aux: float, start_time: float) -> None:
        elapsed = time.time() - start_time
        steps_per_sec = self.optimizer_step / max(elapsed, 1e-6)

        print(f"\nStep {self.optimizer_step:5d}/{self.config.max_steps}")
        print(f"Language loss : {mean_loss:.4f}")
        print(f"MoE aux loss  : {mean_aux:.4f}")
        print(f"Muon LR       : {self.current_muon_lr:.6f}")
        print(f"AdamW LR      : {self.current_adamw_lr:.6f}")
        print(f"Steps/sec     : {steps_per_sec:.2f}")

        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.device) / 1024 ** 3
            reserved = torch.cuda.memory_reserved(self.device) / 1024 ** 3
            print(f"VRAM allocated: {allocated:.2f} GB")
            print(f"VRAM reserved : {reserved:.2f} GB")

    @torch.no_grad()
    def evaluate(self, max_batches: int = 50) -> float:
        """
        Compute average validation language-model loss over at
        most `max_batches` batches.
        """

        print("\nRunning validation...")

        self.model.eval()

        total_loss = 0.0
        batches = 0

        for input_ids, targets in self.val_loader:
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)

            logits, _ = self.model(input_ids)

            loss = F.cross_entropy(
                logits.reshape(-1, self.model.vocab_size), targets.reshape(-1)
            )

            total_loss += loss.item()
            batches += 1

            if batches >= max_batches:
                break

        average_loss = total_loss / max(batches, 1)
        print(f"Validation loss: {average_loss:.4f}")

        self.model.train()

        return average_loss

    def save_checkpoint(self, path: str = None) -> None:
        """
        Save the current model + config + tokenizer name to
        `path` (defaults to `config.checkpoint_path`).
        """

        save_checkpoint(
            self.model,
            self.config,
            self.tokenizer_name,
            path or self.config.checkpoint_path,
        )
