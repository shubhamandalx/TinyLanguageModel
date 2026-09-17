from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from config import GPTConfig



class RMSNorm(nn.Module):
    

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



class RotaryEmbedding(nn.Module):
    

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
        

        seq_len = x.size(-2)

        cos = self.cos_cache[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cache[:seq_len].unsqueeze(0).unsqueeze(0)

        return cos, sin


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
   

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos

    x_out = torch.stack([rotated_even, rotated_odd], dim=-1)

    return x_out.flatten(-2)




class GroupedQueryAttention(nn.Module):
   

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

       
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)

        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len)

        self.dropout = config.dropout

    def forward(self, x: Tensor) -> Tensor:
        

        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(q)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        k = k.repeat_interleave(self.num_query_groups, dim=1)
        v = v.repeat_interleave(self.num_query_groups, dim=1)

        dropout_p = self.dropout if self.training else 0.0

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.o_proj(y)




class SwiGLUExpert(nn.Module):
   

    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()

        self.gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.up = nn.Linear(d_model, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate = F.silu(self.gate(x))
        up = self.up(x)
        return self.down(gate * up)



class MixtureOfExperts(nn.Module):

    

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
        

        B, T, D = x.shape

        router_logits = self.router(x)                      # [B, T, n_experts]
        router_probs = F.softmax(router_logits, dim=-1)

        topk_probs, topk_indices = torch.topk(router_probs, k=self.top_k, dim=-1)

       
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        flat_x = x.reshape(-1, D)
        flat_indices = topk_indices.reshape(-1, self.top_k)
        flat_probs = topk_probs.reshape(-1, self.top_k)

        output = torch.zeros_like(flat_x)

        
        for expert_id, expert in enumerate(self.experts):

            token_positions, topk_position = torch.where(flat_indices == expert_id)

            if token_positions.numel() == 0:
                continue

            expert_input = flat_x[token_positions]
            expert_output = expert(expert_input)

            routing_weight = flat_probs[token_positions, topk_position].unsqueeze(-1)
            expert_output = expert_output * routing_weight

            
            output.index_add_(0, token_positions, expert_output)

        
        importance = router_probs.mean(dim=(0, 1))
        load = F.one_hot(topk_indices, num_classes=self.n_experts).float().mean(dim=(0, 1))
        aux_loss = self.n_experts * torch.sum(importance * load)

        return output.view(B, T, D), aux_loss




class TransformerBlock(nn.Module):
  

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



class ModernGPT(nn.Module):
   

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
      

        x = self.token_embedding(input_ids)

        total_aux_loss = 0.0

        for block in self.blocks:
            x, aux_loss = block(x)
            total_aux_loss = total_aux_loss + aux_loss

        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits, total_aux_loss



def count_parameters(model: ModernGPT) -> int:
   

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Trainable parameters: {total:,} ({total / 1e6:.2f}M)")
    print(f"Approx FP16 parameter memory: {total * 2 / (1024 ** 2):.2f} MB")

    return total


def print_parameter_breakdown(model: ModernGPT) -> None:
    

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
