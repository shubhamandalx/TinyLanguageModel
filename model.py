"""
model.py
========

Complete Modern Mini LLM architecture in a single file, in
top-to-bottom order:

    1. RMSNorm
    2. Rotary Positional Embeddings (RoPE) + apply_rope helper
    3. Grouped Query Attention (GQA)
    4. SwiGLU expert
    5. Mixture of Experts (MoE) with top-k routing
    6. Transformer block
    7. ModernGPT (the full model)
    8. Parameter-count utilities

Every class here is a standard `nn.Module`; the only thing tying
them together is `GPTConfig` (see config.py).
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from config import GPTConfig


# ====================================================================
# 1. RMSNorm
# ====================================================================


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization — a common, cheaper replacement
    for LayerNorm in modern LLMs.

    LayerNorm normalizes using mean + variance. RMSNorm normalizes
    using only the root mean square:

        x / RMS(x)

    followed by a learned per-dimension scale. The normalization
    itself is computed in float32 for numerical stability, even
    when the model runs in fp16.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        original_dtype = x.dtype

        x_float = x.float()
        rms = torch.mean(x_float * x_float, dim=-1, keepdim=True)
        x_float = x_float * torch.rsqrt(rms + self.eps)

        return x_float.to(original_dtype) * self.weight


# ====================================================================
# 2. Rotary Positional Embeddings (RoPE)
# ====================================================================


class RotaryEmbedding(nn.Module):
    """
    Precomputes sin/cos rotation caches for a given head dimension
    and maximum sequence length.

    RoPE encodes token position by rotating the query and key
    vectors, rather than adding a separate learned positional
    embedding — this is the approach used by LLaMA-family models.
    """

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0) -> None:
        super().__init__()

        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        positions = torch.arange(max_seq_len, dtype=torch.float32)

        # Outer product: position * frequency -> [max_seq_len, head_dim / 2]
        freqs = torch.outer(positions, inv_freq)

        self.register_buffer("cos_cache", freqs.cos(), persistent=False)
        self.register_buffer("sin_cache", freqs.sin(), persistent=False)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        x shape: [batch, heads, sequence, head_dim]
        """

        seq_len = x.size(-2)

        cos = self.cos_cache[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cache[:seq_len].unsqueeze(0).unsqueeze(0)

        return cos, sin


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """
    Apply the rotary transformation to `x` by splitting its final
    dimension into even/odd pairs and performing a 2D rotation on
    each pair, then interleaving the results back together.
    """

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos

    x_out = torch.stack([rotated_even, rotated_odd], dim=-1)

    return x_out.flatten(-2)


# ====================================================================
# 3. Grouped Query Attention (GQA)
# ====================================================================


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention: many query heads share a smaller
    number of key/value heads.

    Example: 6 query heads, 2 KV heads -> every KV head is shared
    by 3 query heads. This shrinks the KV cache during inference
    compared to standard multi-head attention, where Q/K/V head
    counts are all equal.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()

        assert config.n_heads % config.n_kv_heads == 0, (
            "n_heads must be divisible by n_kv_heads"
        )

        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.num_query_groups = config.n_heads // config.n_kv_heads

        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)

        # NOTE: only n_kv_heads instead of n_heads — this is the GQA saving.
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)

        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len)

        self.dropout = config.dropout

    def forward(self, x: Tensor) -> Tensor:
        """
        x: [batch, sequence, d_model]
        """

        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # [B, T, C] -> [B, heads, T, head_dim]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(q)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Repeat each KV head so it lines up with its group of Q heads.
        # e.g. K0 K1 -> K0 K0 K0 K1 K1 K1
        k = k.repeat_interleave(self.num_query_groups, dim=1)
        v = v.repeat_interleave(self.num_query_groups, dim=1)

        dropout_p = self.dropout if self.training else 0.0

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.o_proj(y)


# ====================================================================
# 4. SwiGLU expert
# ====================================================================


class SwiGLUExpert(nn.Module):
    """
    One feed-forward expert using the SwiGLU activation:

        SiLU(gate(x)) * up(x)  ->  down projection

    This is the feed-forward design used by LLaMA-family models.
    """

    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()

        self.gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.up = nn.Linear(d_model, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate = F.silu(self.gate(x))
        up = self.up(x)
        return self.down(gate * up)


# ====================================================================
# 5. Mixture of Experts (MoE)
# ====================================================================


class MixtureOfExperts(nn.Module):
    """
    Sparse Mixture of Experts feed-forward layer.

    A learned router assigns each token to its top-k experts (out
    of n_experts total). Only the selected experts run for each
    token, so total parameter count can be large while the active
    compute per token stays small.

    Also returns an auxiliary load-balancing loss: without it, the
    router can collapse to routing almost everything to a single
    "good" expert.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()

        self.n_experts = config.n_experts
        self.top_k = config.top_k_experts

        self.router = nn.Linear(config.d_model, config.n_experts, bias=False)

        self.experts = nn.ModuleList(
            [
                SwiGLUExpert(config.d_model, config.expert_hidden_dim)
                for _ in range(config.n_experts)
            ]
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        x: [B, T, D]

        Returns: (output [B, T, D], aux_loss [scalar])
        """

        B, T, D = x.shape

        router_logits = self.router(x)                      # [B, T, n_experts]
        router_probs = F.softmax(router_logits, dim=-1)

        topk_probs, topk_indices = torch.topk(router_probs, k=self.top_k, dim=-1)

        # Normalize only the selected experts' probabilities so
        # they sum to 1 for each token.
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        flat_x = x.reshape(-1, D)
        flat_indices = topk_indices.reshape(-1, self.top_k)
        flat_probs = topk_probs.reshape(-1, self.top_k)

        output = torch.zeros_like(flat_x)

        # Educational routing loop. Production MoE implementations
        # use highly optimized batched routing kernels instead.
        for expert_id, expert in enumerate(self.experts):

            token_positions, topk_position = torch.where(flat_indices == expert_id)

            if token_positions.numel() == 0:
                continue

            expert_input = flat_x[token_positions]
            expert_output = expert(expert_input)

            routing_weight = flat_probs[token_positions, topk_position].unsqueeze(-1)
            expert_output = expert_output * routing_weight

            # A token can be routed to more than one expert;
            # index_add_ accumulates their weighted contributions.
            output.index_add_(0, token_positions, expert_output)

        # --- Auxiliary load-balancing loss -----------------------------
        importance = router_probs.mean(dim=(0, 1))
        load = F.one_hot(topk_indices, num_classes=self.n_experts).float().mean(dim=(0, 1))
        aux_loss = self.n_experts * torch.sum(importance * load)

        return output.view(B, T, D), aux_loss


# ====================================================================
# 6. Transformer block
# ====================================================================


class TransformerBlock(nn.Module):
    """
    One decoder block using the modern pre-norm structure:

        x -> RMSNorm -> Attention -> (+ residual)
          -> RMSNorm -> MoE       -> (+ residual)
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()

        self.attn_norm = RMSNorm(config.d_model)
        self.attention = GroupedQueryAttention(config)

        self.moe_norm = RMSNorm(config.d_model)
        self.moe = MixtureOfExperts(config)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = x + self.attention(self.attn_norm(x))

        moe_output, aux_loss = self.moe(self.moe_norm(x))
        x = x + moe_output

        return x, aux_loss


# ====================================================================
# 7. ModernGPT — the complete model
# ====================================================================


class ModernGPT(nn.Module):
    """
    Complete decoder-only Transformer language model.

        token IDs -> Embedding -> N x TransformerBlock -> RMSNorm
        -> LM Head (tied to the embedding) -> logits over vocab
    """

    def __init__(self, config: GPTConfig, vocab_size: int) -> None:
        super().__init__()

        self.config = config
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, config.d_model)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        self.final_norm = RMSNorm(config.d_model)

        self.lm_head = nn.Linear(config.d_model, vocab_size, bias=False)

        # Weight tying: reuse the embedding matrix as the output
        # projection instead of learning a second copy of it.
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: Tensor) -> Tuple[Tensor, Tensor]:
        """
        input_ids: [B, T]

        Returns: (logits [B, T, vocab_size], total_aux_loss [scalar])
        """

        x = self.token_embedding(input_ids)

        total_aux_loss = 0.0

        for block in self.blocks:
            x, aux_loss = block(x)
            total_aux_loss = total_aux_loss + aux_loss

        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits, total_aux_loss


# ====================================================================
# 8. Parameter-count utilities
# ====================================================================


def count_parameters(model: ModernGPT) -> int:
    """
    Print and return the total number of trainable parameters.
    """

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Trainable parameters: {total:,} ({total / 1e6:.2f}M)")
    print(f"Approx FP16 parameter memory: {total * 2 / (1024 ** 2):.2f} MB")

    return total


def print_parameter_breakdown(model: ModernGPT) -> None:
    """
    Print parameter counts grouped by major architectural
    component (embedding, attention, MoE, norms).
    """

    groups = {
        "Token Embedding": model.token_embedding,
        "Attention": nn.ModuleList([b.attention for b in model.blocks]),
        "MoE": nn.ModuleList([b.moe for b in model.blocks]),
        "Norms": nn.ModuleList(
            [b.attn_norm for b in model.blocks]
            + [b.moe_norm for b in model.blocks]
            + [model.final_norm]
        ),
    }

    print("\nParameter breakdown:")
    for name, module in groups.items():
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name:<16}: {params:,} ({params / 1e6:.2f}M)")
