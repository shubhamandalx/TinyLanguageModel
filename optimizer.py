"""
optimizer.py
============

Two optimizers are used together during training:

    Muon  — for 2D matrix weights (attention/MoE projections)
    AdamW — for everything else (embeddings, norms, biases)

`build_optimizers(model, config)` inspects the model's parameters,
splits them into the right groups, and returns both optimizer
instances.
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from config import GPTConfig


class Muon(torch.optim.Optimizer):
    """
    Educational implementation of the core idea behind Muon.

    Where AdamW maintains moving averages of gradients and squared
    gradients, Muon uses momentum followed by an approximate
    orthogonalization of the update (via Newton-Schulz iteration).
    It is designed specifically for 2D matrix-shaped parameters.

    NOTE: this is a simplified, educational version. Production
    Muon implementations contain additional optimizations.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        ns_steps: int = 5,
    ) -> None:
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @staticmethod
    def newton_schulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
        """
        Approximate an orthogonalized version of matrix `G` using
        `steps` Newton-Schulz iterations.
        """

        transposed = False
        if G.shape[0] < G.shape[1]:
            G = G.transpose(0, 1)
            transposed = True

        X = G / (G.norm() + 1e-7)

        # Polynomial coefficients from a common Newton-Schulz approximation.
        a, b, c = 3.4445, -4.7750, 2.0315

        for _ in range(steps):
            A = X @ X.transpose(-1, -2)
            X = a * X + (b * A + c * (A @ A)) @ X

        if transposed:
            X = X.transpose(0, 1)

        return X

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Muon is designed for matrix parameters; skip
                # anything else as a safety guard.
                if p.ndim != 2:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                momentum_buffer = state["momentum_buffer"]
                momentum_buffer.mul_(momentum).add_(grad)

                update = self.newton_schulz(momentum_buffer, steps=ns_steps)

                if weight_decay != 0:
                    p.mul_(1.0 - lr * weight_decay)

                p.add_(update, alpha=-lr)

        return loss


def build_optimizers(
    model: nn.Module, config: GPTConfig
) -> Tuple[Muon, torch.optim.AdamW]:
    """
    Split model parameters between Muon and AdamW, then construct
    both optimizers.

    Rule of thumb:
        2D matrix weights (excluding the embedding)  -> Muon
        embeddings, norms, biases, everything else   -> AdamW
    """

    muon_params: List[torch.nn.Parameter] = []
    adamw_decay_params: List[torch.nn.Parameter] = []
    adamw_no_decay_params: List[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_embedding = "token_embedding.weight" in name

        if param.ndim == 2 and not is_embedding:
            muon_params.append(param)
        elif param.ndim == 1 or name.endswith(".bias"):
            # Biases and norm scales generally don't receive weight decay.
            adamw_no_decay_params.append(param)
        else:
            adamw_decay_params.append(param)

    print("Optimizer parameter groups:")
    print("  Muon matrices :", sum(p.numel() for p in muon_params))
    print("  AdamW decay   :", sum(p.numel() for p in adamw_decay_params))
    print("  AdamW no-decay:", sum(p.numel() for p in adamw_no_decay_params))

    muon_optimizer = Muon(
        muon_params,
        lr=config.muon_lr,
        momentum=0.95,
        weight_decay=config.muon_weight_decay,
    )

    adamw_optimizer = torch.optim.AdamW(
        [
            {"params": adamw_decay_params, "weight_decay": config.adamw_weight_decay},
            {"params": adamw_no_decay_params, "weight_decay": 0.0},
        ],
        lr=config.adamw_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    return muon_optimizer, adamw_optimizer
