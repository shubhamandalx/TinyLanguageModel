"""
====================================================================
EDUCATIONAL MODERN MINI-LLM
====================================================================

This is a small decoder-only Transformer trained from scratch.

Modern components included:

    1. tiktoken tokenizer
    2. RMSNorm
    3. RoPE (Rotary Positional Embeddings)
    4. Grouped Query Attention (GQA)
    5. SwiGLU feed-forward networks
    6. Mixture of Experts (MoE)
    7. Top-k expert routing
    8. Muon optimizer for 2D matrix weights
    9. AdamW for embeddings / norms / other parameters
   10. Mixed precision FP16 on CUDA
   11. Gradient accumulation
   12. Cosine learning-rate schedule
   13. Gradient clipping
   14. Weight tying between token embedding and LM head
   15. Autoregressive text generation
   16. torch.save checkpointing

NOT included yet:

    - Quantization
    - QLoRA
    - KV cache for generation
    - FlashAttention-specific custom implementation
    - distributed training
    - instruction tuning

Those can be added after the pretrained model is saved.

Hardware target:

    RTX 2070 ~ 8 GB VRAM should handle this configuration comfortably.
    The code is intentionally conservative so that you can lower
    batch size / sequence length if memory becomes an issue.

Expected parameter count:

    Roughly ~50-65M parameters depending on the exact configuration.

====================================================================
"""


# ================================================================
# IMPORTS
# ================================================================

import math
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import tiktoken


# ================================================================
# 1. GLOBAL CONFIGURATION
# ================================================================
#
# Put the important hyperparameters here.
#
# This is intentionally at the very beginning so that when you
# revisit this file later, you immediately know what model you built.
#
# In real LLM repositories you will see something very similar:
#
#   hidden size
#   number of layers
#   number of attention heads
#   context length
#   vocabulary size
#   etc.
#


@dataclass
class GPTConfig:

    # ------------------------------------------------------------
    # TOKENIZER
    # ------------------------------------------------------------

    # tiktoken vocabulary.
    #
    # cl100k_base is the tokenizer used by several OpenAI models.
    #
    # It has around 100k tokens, which means the embedding layer
    # will be one of the largest parts of our model.
    #
    tokenizer_name: str = "cl100k_base"

    # ------------------------------------------------------------
    # TRANSFORMER SIZE
    # ------------------------------------------------------------

    # Number of Transformer blocks.
    #
    # More layers = deeper model.
    #
    n_layers: int = 8

    # Hidden representation size.
    #
    # Every token becomes a vector of this size.
    #
    d_model: int = 384

    # Total number of QUERY heads.
    #
    # d_model must be divisible by n_heads.
    #
    n_heads: int = 6

    # Number of KEY/VALUE heads.
    #
    # This is smaller than n_heads.
    #
    # This is what makes this Grouped Query Attention.
    #
    # Example:
    #
    #   Q heads  = 6
    #   KV heads = 2
    #
    # Every KV head is shared by 3 Q heads.
    #
    n_kv_heads: int = 2

    # ------------------------------------------------------------
    # CONTEXT WINDOW
    # ------------------------------------------------------------

    # Maximum number of tokens processed at once.
    #
    # 256 is deliberately small for educational / consumer-GPU
    # training.
    #
    max_seq_len: int = 256

    # ------------------------------------------------------------
    # MOE
    # ------------------------------------------------------------

    # Number of experts in each MoE layer.
    #
    # Instead of one huge feed-forward network, we have multiple
    # smaller expert networks.
    #
    n_experts: int = 4

    # How many experts each token actually visits.
    #
    # Top-2 means:
    #
    #   4 experts exist
    #   but each token activates only 2
    #
    # This is the important idea behind sparse MoE.
    #
    top_k_experts: int = 2

    # Hidden size INSIDE each expert.
    #
    # This is intentionally moderate so the model stays small.
    #
    expert_hidden_dim: int = 512

    # ------------------------------------------------------------
    # REGULARIZATION
    # ------------------------------------------------------------

    dropout: float = 0.0

    # ------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------

    # Micro batch size.
    #
    # This is the actual batch sent to GPU at a time.
    #
    batch_size: int = 4

    # We accumulate gradients across multiple micro batches.
    #
    # Effective batch size becomes:
    #
    #     batch_size * gradient_accumulation_steps
    #
    gradient_accumulation_steps: int = 8

    # Number of optimizer updates.
    #
    # Increase this for better training.
    #
    max_steps: int = 2500

    # How many training steps should be spent warming up LR.
    #
    warmup_steps: int = 200

    # Learning rate for AdamW.
    #
    adamw_lr: float = 3e-4

    # Learning rate for Muon.
    #
    # Muon normally operates at a noticeably different LR scale
    # than AdamW.
    #
    muon_lr: float = 2e-2

    # AdamW weight decay.
    #
    adamw_weight_decay: float = 0.1

    # Muon weight decay.
    #
    muon_weight_decay: float = 0.01

    # Gradient clipping.
    grad_clip: float = 1.0

    # ------------------------------------------------------------
    # MOE AUXILIARY LOSS
    # ------------------------------------------------------------

    # Small coefficient encouraging better load balancing between
    # experts.
    #
    moe_aux_loss_weight: float = 0.01

    # ------------------------------------------------------------
    # PRINTING / CHECKPOINTING
    # ------------------------------------------------------------

    print_every: int = 50

    checkpoint_path: str = "modern_mini_llm.pt"

    # ------------------------------------------------------------
    # DATASET
    # ------------------------------------------------------------

    input_file: str = "input.txt"

    # ------------------------------------------------------------
    # GENERATION
    # ------------------------------------------------------------

    generation_temperature: float = 0.8

    generation_top_k: int = 40

    # Generate enough tokens to see a couple of lines.
    generation_max_new_tokens: int = 60


# ================================================================
# 2. CREATE CONFIG
# ================================================================

gpt_config = GPTConfig()


# ================================================================
# 3. DEVICE SETUP
# ================================================================

def get_device():

    """
    Decide whether CUDA is available.

    If CUDA exists, use the NVIDIA GPU.

    Otherwise fall back to CPU.

    For your RTX 2070 this should normally print:

        cuda
    """

    if torch.cuda.is_available():

        device = torch.device("cuda")

        # Print GPU information so you know exactly where
        # training is happening.
        print("=" * 70)
        print("CUDA detected")
        print("GPU:", torch.cuda.get_device_name(0))
        print("=" * 70)

        # This allows PyTorch to choose efficient matrix
        # multiplication algorithms where appropriate.
        torch.set_float32_matmul_precision("high")

    else:

        device = torch.device("cpu")

        print("=" * 70)
        print("CUDA not available -> using CPU")
        print("=" * 70)

    return device


# ================================================================
# 4. TOKENIZER
# ================================================================

def load_tokenizer(config):

    """
    Load the tiktoken tokenizer.

    Important idea:

        Text
          ↓
        tokenizer
          ↓
        token IDs
          ↓
        neural network

    Example:

        "Hello world"

    might become something conceptually like:

        [9906, 1917]

    These integers are what the model actually sees.
    """

    print("\n[1/7] Loading tokenizer...")

    tokenizer = tiktoken.get_encoding(config.tokenizer_name)

    print("Tokenizer:", config.tokenizer_name)
    print("Vocabulary size:", tokenizer.n_vocab)

    return tokenizer


# ================================================================
# 5. LOAD AND TOKENIZE DATASET
# ================================================================

def load_tokens(config, tokenizer):

    """
    Read input.txt and convert the complete text into token IDs.

    We assume input.txt lives in the SAME directory as this script.

    Example:

        input.txt

            The little dog went outside.
            The sun was shining brightly.
            ...

    """

    print("\n[2/7] Reading input.txt...")

    with open(config.input_file, "r", encoding="utf-8") as f:

        text = f.read()

    print("Characters in dataset:", len(text))

    # ------------------------------------------------------------
    # Tokenize the entire document.
    #
    # allowed_special=set()
    #
    # means we don't allow special tokens to accidentally appear
    # inside normal text.
    # ------------------------------------------------------------

    token_ids = tokenizer.encode(
        text,
        allowed_special={'<|endoftext|>'}
    )

    print("Number of tokens:", len(token_ids))

    # ------------------------------------------------------------
    # Convert Python list -> PyTorch tensor.
    #
    # int64 / long is required because token IDs are used as
    # indices into the embedding matrix.
    # ------------------------------------------------------------

    tokens = torch.tensor(
        token_ids,
        dtype=torch.long
    )

    return tokens


# ================================================================
# 6. TRAIN / VALIDATION SPLIT
# ================================================================

def split_dataset(tokens):

    """
    Split the token sequence into:

        95% training
         5% validation

    We do this based on token positions rather than randomly
    shuffling individual tokens.

    Why?

    Because randomly mixing everything could cause information
    leakage between train and validation data.
    """

    split = int(len(tokens) * 0.95)

    train_tokens = tokens[:split]

    val_tokens = tokens[split:]

    print("\nDataset split")
    print("Train tokens:", len(train_tokens))
    print("Validation tokens:", len(val_tokens))

    return train_tokens, val_tokens


# ================================================================
# 7. LANGUAGE MODEL DATASET
# ================================================================

class LanguageModelDataset(Dataset):

    """
    Converts one long token stream into many training examples.

    Suppose the token sequence is:

        A B C D E F G H

    and seq_len = 4

    Input:

        A B C D

    Target:

        B C D E

    The model learns:

        given A -> predict B
        given A B -> predict C
        given A B C -> predict D
        given A B C D -> predict E
    """

    def __init__(self, tokens, seq_len):

        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):

        # We need seq_len + 1 tokens because target is shifted
        # by one position.
        return len(self.tokens) - self.seq_len - 1

    def __getitem__(self, index):

        # Input sequence.
        x = self.tokens[
            index:
            index + self.seq_len
        ]

        # Same sequence shifted by one token.
        y = self.tokens[
            index + 1:
            index + self.seq_len + 1
        ]

        return x, y


# ================================================================
# 8. RMSNorm
# ================================================================

class RMSNorm(nn.Module):

    """
    RMSNorm = Root Mean Square Normalization.

    This is a common replacement for LayerNorm in modern LLMs.

    LayerNorm:

        normalize using mean + variance

    RMSNorm:

        normalize using root mean square

    Formula conceptually:

        x / RMS(x)

    followed by a learned scale parameter.

    We keep the calculation in float32 for numerical stability.
    """

    def __init__(self, dim, eps=1e-6):

        super().__init__()

        self.eps = eps

        # One learnable scale value per hidden dimension.
        self.weight = nn.Parameter(
            torch.ones(dim)
        )

    def forward(self, x):

        # Save original dtype.
        original_dtype = x.dtype

        # Do normalization in FP32 for stability.
        x_float = x.float()

        # Mean of squared values.
        rms = torch.mean(
            x_float * x_float,
            dim=-1,
            keepdim=True
        )

        # Divide by sqrt(mean square + epsilon).
        x_float = x_float * torch.rsqrt(
            rms + self.eps
        )

        # Return to original dtype.
        return (
            x_float.to(original_dtype)
            * self.weight
        )


# ================================================================
# 9. ROTARY POSITIONAL EMBEDDINGS - RoPE
# ================================================================

class RotaryEmbedding(nn.Module):

    """
    RoPE gives the model information about token positions.

    Traditional Transformer:

        token embedding
        +
        learned positional embedding

    RoPE works differently.

    It rotates the QUERY and KEY vectors according to position.

    This means positional information becomes part of attention
    itself.

    Modern LLMs such as LLaMA-family models use this idea.
    """

    def __init__(
        self,
        head_dim,
        max_seq_len,
        theta=10000.0
    ):

        super().__init__()

        # Frequencies for each pair of dimensions.

        inv_freq = 1.0 / (
            theta ** (
                torch.arange(
                    0,
                    head_dim,
                    2,
                    dtype=torch.float32
                ) / head_dim
            )
        )

        # Register as buffer instead of Parameter.
        #
        # Buffer:
        #     moves with model to GPU
        #     but is NOT trained.
        self.register_buffer(
            "inv_freq",
            inv_freq,
            persistent=False
        )

        # Position indices:
        #
        # 0, 1, 2, 3, ...
        positions = torch.arange(
            max_seq_len,
            dtype=torch.float32
        )

        # Outer product:
        #
        # position * frequency
        #
        # Shape:
        #
        # [max_seq_len, head_dim / 2]
        freqs = torch.outer(
            positions,
            inv_freq
        )

        # Precompute sin/cos.
        self.register_buffer(
            "cos_cache",
            freqs.cos(),
            persistent=False
        )

        self.register_buffer(
            "sin_cache",
            freqs.sin(),
            persistent=False
        )

    def forward(self, x):

        """
        x shape:

            [batch, heads, sequence, head_dim]

        """

        seq_len = x.size(-2)

        cos = self.cos_cache[
            :seq_len
        ].unsqueeze(0).unsqueeze(0)

        sin = self.sin_cache[
            :seq_len
        ].unsqueeze(0).unsqueeze(0)

        return cos, sin


def apply_rope(x, cos, sin):

    """
    Apply rotary transformation.

    We split the final dimension into:

        even dimensions
        odd dimensions

    Then perform a 2D rotation.

    """

    x_even = x[..., 0::2]

    x_odd = x[..., 1::2]

    rotated_even = (
        x_even * cos
        - x_odd * sin
    )

    rotated_odd = (
        x_even * sin
        + x_odd * cos
    )

    # Interleave them back together.

    x_out = torch.stack(
        [rotated_even, rotated_odd],
        dim=-1
    )

    return x_out.flatten(-2)


# ================================================================
# 10. GROUPED QUERY ATTENTION
# ================================================================

class GroupedQueryAttention(nn.Module):

    """
    GQA = Grouped Query Attention.

    Standard multi-head attention:

        Q heads = K heads = V heads

    GQA:

        many Q heads
        fewer KV heads

    Example:

        Q = 6 heads
        K = 2 heads
        V = 2 heads

    So:

        3 Q heads share each KV head.

    Why?

        KV cache becomes significantly smaller during inference.

    This is widely used in modern LLM architectures.
    """

    def __init__(self, config):

        super().__init__()

        assert (
            config.n_heads % config.n_kv_heads == 0
        ), "n_heads must be divisible by n_kv_heads"

        self.n_heads = config.n_heads

        self.n_kv_heads = config.n_kv_heads

        self.head_dim = (
            config.d_model // config.n_heads
        )

        # Number of Q heads sharing one KV head.
        self.num_query_groups = (
            config.n_heads // config.n_kv_heads
        )

        # --------------------------------------------------------
        # Query projection
        #
        # d_model -> n_heads * head_dim
        #
        # Since:
        #
        # n_heads * head_dim = d_model
        # --------------------------------------------------------

        self.q_proj = nn.Linear(
            config.d_model,
            config.n_heads * self.head_dim,
            bias=False
        )

        # --------------------------------------------------------
        # Key projection
        #
        # IMPORTANT:
        #
        # Only n_kv_heads instead of n_heads.
        #
        # That's the GQA saving.
        # --------------------------------------------------------

        self.k_proj = nn.Linear(
            config.d_model,
            config.n_kv_heads * self.head_dim,
            bias=False
        )

        # Value projection.
        self.v_proj = nn.Linear(
            config.d_model,
            config.n_kv_heads * self.head_dim,
            bias=False
        )

        # Final attention output projection.
        self.o_proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=False
        )

        # Rotary positional embedding.
        self.rope = RotaryEmbedding(
            self.head_dim,
            config.max_seq_len
        )

        self.dropout = config.dropout

    def forward(self, x):

        """
        x:

            [batch, sequence, d_model]
        """

        B, T, C = x.shape

        # --------------------------------------------------------
        # Project hidden state into Q, K, V.
        # --------------------------------------------------------

        q = self.q_proj(x)

        k = self.k_proj(x)

        v = self.v_proj(x)

        # --------------------------------------------------------
        # Reshape.
        #
        # Q:
        #
        # [B, T, C]
        #
        # ->
        #
        # [B, heads, T, head_dim]
        # --------------------------------------------------------

        q = q.view(
            B,
            T,
            self.n_heads,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            B,
            T,
            self.n_kv_heads,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            B,
            T,
            self.n_kv_heads,
            self.head_dim
        ).transpose(1, 2)

        # --------------------------------------------------------
        # RoPE
        # --------------------------------------------------------

        cos, sin = self.rope(q)

        q = apply_rope(
            q,
            cos,
            sin
        )

        k = apply_rope(
            k,
            cos,
            sin
        )

        # --------------------------------------------------------
        # GQA
        #
        # We have:
        #
        # Q = 6 heads
        # K = 2 heads
        #
        # Repeat each K/V head 3 times.
        #
        # Example:
        #
        # K0 K1
        #
        # becomes:
        #
        # K0 K0 K0 K1 K1 K1
        # --------------------------------------------------------

        k = k.repeat_interleave(
            self.num_query_groups,
            dim=1
        )

        v = v.repeat_interleave(
            self.num_query_groups,
            dim=1
        )

        # --------------------------------------------------------
        # Causal Self Attention
        #
        # PyTorch provides a highly optimized implementation.
        #
        # scaled_dot_product_attention internally can select
        # efficient attention kernels.
        #
        # This saves us from writing:
        #
        #       Q @ K^T
        #       / sqrt(d)
        #       softmax(...)
        #       @ V
        #
        # manually.
        # --------------------------------------------------------

        dropout_p = (
            self.dropout
            if self.training
            else 0.0
        )

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=True
        )

        # --------------------------------------------------------
        # Back to:
        #
        # [B, T, C]
        # --------------------------------------------------------

        y = y.transpose(1, 2).contiguous()

        y = y.view(
            B,
            T,
            C
        )

        # Final projection.
        y = self.o_proj(y)

        return y


# ================================================================
# 11. SWIGLU EXPERT
# ================================================================

class SwiGLUExpert(nn.Module):

    """
    One feed-forward expert.

    The activation is SwiGLU:

        SiLU(gate(x)) * up(x)

    followed by a down projection.

    Conceptually:

        x
        |
        +-------------------+
        |                   |
        gate                up
        |                   |
       SiLU                 |
        |                   |
        +------ multiply ---+
                  |
               down
                  |
                output

    """

    def __init__(
        self,
        d_model,
        hidden_dim
    ):

        super().__init__()

        # Gate branch.
        self.gate = nn.Linear(
            d_model,
            hidden_dim,
            bias=False
        )

        # Up branch.
        self.up = nn.Linear(
            d_model,
            hidden_dim,
            bias=False
        )

        # Project back to model dimension.
        self.down = nn.Linear(
            hidden_dim,
            d_model,
            bias=False
        )

    def forward(self, x):

        # SiLU = Swish.
        gate = F.silu(
            self.gate(x)
        )

        up = self.up(x)

        # The GLU gating operation.
        x = gate * up

        # Project back to d_model.
        x = self.down(x)

        return x


# ================================================================
# 12. MIXTURE OF EXPERTS
# ================================================================

class MixtureOfExperts(nn.Module):

    """
    Sparse Mixture of Experts.

    Instead of one feed-forward network:

        x -> FFN -> output

    we have:

        x
         |
       router
       / | \ \
      E0 E1 E2 E3
       \ | /
        combine

    The router decides which experts should process each token.

    With:

        n_experts = 4
        top_k = 2

    each token uses only TWO experts.

    This means:

        total parameters can be large

    while:

        active compute per token stays much smaller.

    This is one of the key ideas behind modern sparse MoE models.
    """

    def __init__(self, config):

        super().__init__()

        self.n_experts = config.n_experts

        self.top_k = config.top_k_experts

        # Router:
        #
        # hidden state -> one score for each expert
        #
        # [d_model] -> [n_experts]
        self.router = nn.Linear(
            config.d_model,
            config.n_experts,
            bias=False
        )

        # Create the experts.
        self.experts = nn.ModuleList(
            [
                SwiGLUExpert(
                    config.d_model,
                    config.expert_hidden_dim
                )
                for _ in range(config.n_experts)
            ]
        )

    def forward(self, x):

        """
        x:

            [B, T, D]

        """

        B, T, D = x.shape

        # --------------------------------------------------------
        # Router scores
        # --------------------------------------------------------

        router_logits = self.router(x)

        # Shape:
        #
        # [B, T, number_of_experts]

        # Convert scores to probabilities.
        router_probs = F.softmax(
            router_logits,
            dim=-1
        )

        # --------------------------------------------------------
        # Top-k routing
        # --------------------------------------------------------

        topk_probs, topk_indices = torch.topk(
            router_probs,
            k=self.top_k,
            dim=-1
        )

        # --------------------------------------------------------
        # Normalize only the selected experts.
        #
        # Example:
        #
        # selected:
        #
        # expert 1 = 0.7
        # expert 3 = 0.2
        #
        # normalized:
        #
        # expert 1 = 0.777...
        # expert 3 = 0.222...
        # --------------------------------------------------------

        topk_probs = (
            topk_probs
            / topk_probs.sum(
                dim=-1,
                keepdim=True
            )
        )

        # --------------------------------------------------------
        # Flatten tokens.
        #
        # [B,T,D]
        #
        # ->
        #
        # [B*T,D]
        # --------------------------------------------------------

        flat_x = x.reshape(
            -1,
            D
        )

        flat_indices = topk_indices.reshape(
            -1,
            self.top_k
        )

        flat_probs = topk_probs.reshape(
            -1,
            self.top_k
        )

        # Output buffer.
        output = torch.zeros_like(
            flat_x
        )

        # --------------------------------------------------------
        # Process each expert.
        #
        # This loop is educational.
        #
        # Production MoE implementations use highly optimized
        # routing kernels instead.
        # --------------------------------------------------------

        for expert_id, expert in enumerate(
            self.experts
        ):

            # Find all token/expert assignments where this
            # particular expert was selected.

            token_positions, topk_position = torch.where(
                flat_indices == expert_id
            )

            # No tokens assigned to this expert.
            if token_positions.numel() == 0:
                continue

            # Select tokens for this expert.
            expert_input = flat_x[
                token_positions
            ]

            # Run the expert.
            expert_output = expert(
                expert_input
            )

            # Routing weight for every token.
            routing_weight = flat_probs[
                token_positions,
                topk_position
            ]

            routing_weight = routing_weight.unsqueeze(-1)

            # Multiply expert output by router probability.
            expert_output = (
                expert_output
                * routing_weight
            )

            # Multiple experts can contribute to the same token.
            #
            # index_add accumulates them correctly.
            output.index_add_(
                0,
                token_positions,
                expert_output
            )

        # --------------------------------------------------------
        # Auxiliary load-balancing loss
        # --------------------------------------------------------
        #
        # Without this, the router may discover:
        #
        #   "Expert 0 is good"
        #
        # and send almost everything there.
        #
        # We want the experts to receive reasonably balanced load.
        #

        importance = router_probs.mean(
            dim=(0, 1)
        )

        # Which experts were selected.
        load = F.one_hot(
            topk_indices,
            num_classes=self.n_experts
        ).float()

        load = load.mean(
            dim=(0, 1)
        )

        # Simplified balancing objective.
        aux_loss = (
            self.n_experts
            * torch.sum(
                importance * load
            )
        )

        # Reshape back to original shape.
        output = output.view(
            B,
            T,
            D
        )

        return output, aux_loss


# ================================================================
# 13. TRANSFORMER BLOCK
# ================================================================

class TransformerBlock(nn.Module):

    """
    One complete decoder block.

    Modern pre-norm structure:

            x
            |
        RMSNorm
            |
       Attention
            |
           + <---- residual
            |
        RMSNorm
            |
          MoE
            |
           + <---- residual
            |
          output

    """

    def __init__(self, config):

        super().__init__()

        # Normalization before attention.
        self.attn_norm = RMSNorm(
            config.d_model
        )

        # Grouped Query Attention.
        self.attention = GroupedQueryAttention(
            config
        )

        # Normalization before MoE.
        self.moe_norm = RMSNorm(
            config.d_model
        )

        # Mixture of Experts.
        self.moe = MixtureOfExperts(
            config
        )

    def forward(self, x):

        # --------------------------------------------------------
        # ATTENTION SUB-LAYER
        # --------------------------------------------------------

        # Pre-normalize.
        normalized = self.attn_norm(x)

        # Attention.
        attention_output = self.attention(
            normalized
        )

        # Residual connection.
        x = x + attention_output

        # --------------------------------------------------------
        # MOE SUB-LAYER
        # --------------------------------------------------------

        normalized = self.moe_norm(x)

        moe_output, aux_loss = self.moe(
            normalized
        )

        # Residual connection.
        x = x + moe_output

        return x, aux_loss


# ================================================================
# 14. COMPLETE GPT MODEL
# ================================================================

class ModernGPT(nn.Module):

    """
    Complete decoder-only language model.

    High-level architecture:

        token IDs
           |
        Embedding
           |
        Block 1
           |
        Block 2
           |
          ...
           |
        Block N
           |
        RMSNorm
           |
        LM Head
           |
        logits over vocabulary
    """

    def __init__(
        self,
        config,
        vocab_size
    ):

        super().__init__()

        self.config = config

        self.vocab_size = vocab_size

        # --------------------------------------------------------
        # TOKEN EMBEDDING
        # --------------------------------------------------------
        #
        # Converts integer token IDs into dense vectors.
        #
        # [token_id]
        #
        # ->
        #
        # [d_model vector]
        #

        self.token_embedding = nn.Embedding(
            vocab_size,
            config.d_model
        )

        # --------------------------------------------------------
        # TRANSFORMER BLOCKS
        # --------------------------------------------------------

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.n_layers)
            ]
        )

        # Final RMSNorm.
        self.final_norm = RMSNorm(
            config.d_model
        )

        # --------------------------------------------------------
        # LANGUAGE MODEL HEAD
        # --------------------------------------------------------
        #
        # Converts hidden vectors into a probability score for
        # every token in the vocabulary.
        #
        # [d_model]
        #
        # ->
        #
        # [vocab_size]
        #

        self.lm_head = nn.Linear(
            config.d_model,
            vocab_size,
            bias=False
        )

        # --------------------------------------------------------
        # WEIGHT TYING
        # --------------------------------------------------------
        #
        # Instead of having TWO copies of the vocabulary matrix:
        #
        # embedding
        # lm_head
        #
        # we reuse the same weights.
        #
        # This saves a large amount of parameters.
        #

        self.lm_head.weight = (
            self.token_embedding.weight
        )

        # Initialize weights.
        self.apply(self._init_weights)

    def _init_weights(self, module):

        """
        Simple Transformer-style initialization.

        """

        if isinstance(module, nn.Linear):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:
                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(module, nn.Embedding):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def forward(self, input_ids):

        """
        input_ids:

            [B,T]

        returns:

            logits:
                [B,T,vocab_size]

            auxiliary MoE loss
        """

        # --------------------------------------------------------
        # TOKEN IDs -> EMBEDDINGS
        # --------------------------------------------------------

        x = self.token_embedding(
            input_ids
        )

        total_aux_loss = 0.0

        # --------------------------------------------------------
        # PASS THROUGH ALL TRANSFORMER BLOCKS
        # --------------------------------------------------------

        for block in self.blocks:

            x, aux_loss = block(x)

            total_aux_loss = (
                total_aux_loss
                + aux_loss
            )

        # --------------------------------------------------------
        # FINAL NORMALIZATION
        # --------------------------------------------------------

        x = self.final_norm(x)

        # --------------------------------------------------------
        # PROJECT TO VOCABULARY
        # --------------------------------------------------------

        logits = self.lm_head(x)

        return (
            logits,
            total_aux_loss
        )


# ================================================================
# 15. PARAMETER COUNT
# ================================================================

def count_parameters(model):

    """
    Print total number of trainable parameters.

    This is something you should get used to checking for every
    model.

    Example:

        60M parameters

    means roughly sixty million learned numbers.
    """

    total = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\n" + "=" * 70)
    print("MODEL SIZE")
    print("=" * 70)

    print(
        f"Trainable parameters: "
        f"{total:,}"
    )

    print(
        f"Trainable parameters: "
        f"{total / 1e6:.2f} Million"
    )

    print(
        f"Approx FP16 parameter memory: "
        f"{total * 2 / (1024 ** 2):.2f} MB"
    )

    print("=" * 70)

    return total


# ================================================================
# 16. MODEL COMPONENT PARAMETER COUNT
# ================================================================

def print_parameter_breakdown(model):

    """
    Print parameters by major component.

    This helps you understand which parts of an LLM actually
    consume the most parameters.
    """

    print("\nParameter breakdown:")

    groups = {
        "Token Embedding": model.token_embedding,
        "Attention": nn.ModuleList(
            [
                block.attention
                for block in model.blocks
            ]
        ),
        "MoE": nn.ModuleList(
            [
                block.moe
                for block in model.blocks
            ]
        ),
        "Norms": nn.ModuleList(
            [
                block.attn_norm
                for block in model.blocks
            ]
            + [
                block.moe_norm
                for block in model.blocks
            ]
            + [model.final_norm]
        )
    }

    for name, module in groups.items():

        params = sum(
            p.numel()
            for p in module.parameters()
        )

        print(
            f"{name:<20}: "
            f"{params:,} "
            f"({params / 1e6:.2f}M)"
        )


# ================================================================
# 17. MUON OPTIMIZER
# ================================================================

class Muon(torch.optim.Optimizer):

    """
    EDUCATIONAL implementation of the basic idea behind Muon.

    Muon is especially interesting for matrix-shaped weights.

    AdamW:

        maintains moving averages of gradients and squared
        gradients.

    Muon:

        uses momentum + orthogonalization of the update.

    Here we use Muon for 2D matrix parameters and AdamW for:

        embeddings
        RMSNorm parameters
        other non-2D parameters

    IMPORTANT:

        This is an educational implementation.

        Production Muon implementations contain additional
        optimizations and details.
    """

    def __init__(
        self,
        params,
        lr=0.02,
        momentum=0.95,
        weight_decay=0.01,
        ns_steps=5
    ):

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_steps=ns_steps
        )

        super().__init__(
            params,
            defaults
        )

    @staticmethod
    def newton_schulz(
        G,
        steps=5
    ):

        """
        Approximate an orthogonalized version of a matrix.

        Newton-Schulz iterations are used here to make the update
        behave more like an orthogonal matrix transformation.
        """

        # --------------------------------------------------------
        # Make sure the larger dimension is the row dimension.
        #
        # This simplifies the matrix multiplication below.
        # --------------------------------------------------------

        transposed = False

        if G.shape[0] < G.shape[1]:

            G = G.transpose(0, 1)

            transposed = True

        # Normalize initial matrix.
        X = G / (
            G.norm() + 1e-7
        )

        # Polynomial coefficients used in a common
        # Newton-Schulz approximation.
        a = 3.4445
        b = -4.7750
        c = 2.0315

        for _ in range(steps):

            A = X @ X.transpose(
                -1,
                -2
            )

            X = (
                a * X
                + (
                    b * A
                    + c * (A @ A)
                ) @ X
            )

        if transposed:

            X = X.transpose(
                0,
                1
            )

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

            weight_decay = group[
                "weight_decay"
            ]

            ns_steps = group[
                "ns_steps"
            ]

            for p in group["params"]:

                if p.grad is None:
                    continue

                # Muon is designed for matrix parameters.
                #
                # Our optimizer group should already contain 2D
                # matrices, but keep this guard for safety.
                if p.ndim != 2:
                    continue

                grad = p.grad

                # ------------------------------------------------
                # Momentum buffer
                # ------------------------------------------------

                state = self.state[p]

                if len(state) == 0:

                    state[
                        "momentum_buffer"
                    ] = torch.zeros_like(
                        p
                    )

                momentum_buffer = state[
                    "momentum_buffer"
                ]

                momentum_buffer.mul_(
                    momentum
                ).add_(
                    grad
                )

                # ------------------------------------------------
                # Orthogonalize update.
                # ------------------------------------------------

                update = self.newton_schulz(
                    momentum_buffer,
                    steps=ns_steps
                )

                # ------------------------------------------------
                # Weight decay
                # ------------------------------------------------

                if weight_decay != 0:

                    p.mul_(
                        1.0
                        - lr * weight_decay
                    )

                # ------------------------------------------------
                # Parameter update
                # ------------------------------------------------

                p.add_(
                    update,
                    alpha=-lr
                )

        return loss


# ================================================================
# 18. BUILD MUON + ADAMW OPTIMIZERS
# ================================================================

def build_optimizers(model, config):

    """
    Split parameters between Muon and AdamW.

    Rough rule:

        2D matrix weights -> Muon

        everything else -> AdamW

    However, embeddings are intentionally kept on AdamW because
    they are not conventional dense projection matrices.
    """

    muon_params = []

    adamw_decay_params = []

    adamw_no_decay_params = []

    for name, param in model.named_parameters():

        if not param.requires_grad:
            continue

        # Embedding should stay with AdamW.
        is_embedding = (
            "token_embedding.weight"
            in name
        )

        if param.ndim == 2 and not is_embedding:

            muon_params.append(param)

        else:

            # Biases and norm parameters generally don't receive
            # weight decay.
            if (
                param.ndim == 1
                or name.endswith(".bias")
            ):

                adamw_no_decay_params.append(
                    param
                )

            else:

                adamw_decay_params.append(
                    param
                )

    print("\nOptimizer parameter groups")

    print(
        "Muon matrices:",
        sum(
            p.numel()
            for p in muon_params
        )
    )

    print(
        "AdamW decay:",
        sum(
            p.numel()
            for p in adamw_decay_params
        )
    )

    print(
        "AdamW no-decay:",
        sum(
            p.numel()
            for p in adamw_no_decay_params
        )
    )

    # ------------------------------------------------------------
    # Muon optimizer.
    # ------------------------------------------------------------

    muon_optimizer = Muon(
        muon_params,
        lr=config.muon_lr,
        momentum=0.95,
        weight_decay=config.muon_weight_decay
    )

    # ------------------------------------------------------------
    # AdamW optimizer.
    # ------------------------------------------------------------

    adamw_optimizer = torch.optim.AdamW(
        [
            {
                "params": adamw_decay_params,
                "weight_decay":
                    config.adamw_weight_decay
            },
            {
                "params": adamw_no_decay_params,
                "weight_decay": 0.0
            }
        ],
        lr=config.adamw_lr,
        betas=(0.9, 0.95),
        eps=1e-8
    )

    return (
        muon_optimizer,
        adamw_optimizer
    )


# ================================================================
# 19. LEARNING-RATE SCHEDULER
# ================================================================

def get_lr_multiplier(
    step,
    warmup_steps,
    max_steps
):

    """
    Learning-rate schedule:

            ^
            |       ______
            |      /      \
            |     /        \
            |____/          \____
                warmup       cosine decay

    """

    # ------------------------------------------------------------
    # WARMUP
    # ------------------------------------------------------------

    if step < warmup_steps:

        return (
            step + 1
        ) / warmup_steps

    # ------------------------------------------------------------
    # COSINE DECAY
    # ------------------------------------------------------------

    progress = (
        step - warmup_steps
    ) / max(
        1,
        max_steps - warmup_steps
    )

    progress = min(
        max(progress, 0.0),
        1.0
    )

    return (
        0.5
        * (
            1.0
            + math.cos(
                math.pi * progress
            )
        )
    )


def set_learning_rate(
    optimizer,
    base_lr,
    multiplier
):

    """
    Update learning rate for every optimizer parameter group.
    """

    new_lr = (
        base_lr
        * multiplier
    )

    for group in optimizer.param_groups:

        group["lr"] = new_lr

    return new_lr


# ================================================================
# 20. TEXT GENERATION
# ================================================================

@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    device,
    config
):

    """
    Autoregressive text generation.

    Suppose prompt is:

        "The little girl"

    The model does:

        predict next token
        append token
        predict next token
        append token
        ...

    """

    model.eval()

    # ------------------------------------------------------------
    # Convert prompt -> token IDs.
    # ------------------------------------------------------------

    tokens = tokenizer.encode(
        prompt,
        allowed_special=set()
    )

    input_ids = torch.tensor(
        [tokens],
        dtype=torch.long,
        device=device
    )

    # ------------------------------------------------------------
    # Generate one token at a time.
    #
    # NOTE:
    #
    # This educational version recomputes the entire context.
    #
    # A production inference implementation would use a KV cache.
    #
    # We'll add that when we upgrade this model.
    # ------------------------------------------------------------

    for _ in range(
        config.generation_max_new_tokens
    ):

        # Limit to maximum context size.
        input_for_model = input_ids[
            :, -config.max_seq_len:
        ]

        # Forward pass.
        logits, _ = model(
            input_for_model
        )

        # Only need predictions for the final token.
        next_token_logits = logits[
            :, -1, :
        ]

        # --------------------------------------------------------
        # TEMPERATURE
        # --------------------------------------------------------
        #
        # Lower:
        #     more deterministic
        #
        # Higher:
        #     more random
        #

        next_token_logits = (
            next_token_logits
            / config.generation_temperature
        )

        # --------------------------------------------------------
        # TOP-K SAMPLING
        # --------------------------------------------------------
        #
        # Only allow the model to choose among its strongest K
        # candidates.
        #

        if config.generation_top_k is not None:

            values, _ = torch.topk(
                next_token_logits,
                min(
                    config.generation_top_k,
                    next_token_logits.size(-1)
                )
            )

            # Everything below the K-th highest score becomes
            # negative infinity.
            threshold = values[
                :, -1
            ].unsqueeze(-1)

            next_token_logits = torch.where(
                next_token_logits
                < threshold,
                torch.full_like(
                    next_token_logits,
                    float("-inf")
                ),
                next_token_logits
            )

        # --------------------------------------------------------
        # Convert logits -> probability distribution.
        # --------------------------------------------------------

        probs = F.softmax(
            next_token_logits,
            dim=-1
        )

        # --------------------------------------------------------
        # Randomly sample ONE token.
        # --------------------------------------------------------

        next_token = torch.multinomial(
            probs,
            num_samples=1
        )

        # --------------------------------------------------------
        # Append token to context.
        # --------------------------------------------------------

        input_ids = torch.cat(
            [
                input_ids,
                next_token
            ],
            dim=1
        )

    # ------------------------------------------------------------
    # Decode token IDs -> human-readable text.
    # ------------------------------------------------------------

    generated_text = tokenizer.decode(
        input_ids[0].tolist()
    )

    model.train()

    return generated_text


# ================================================================
# 21. TRAINING FUNCTION
# ================================================================

def train_model(
    model,
    train_loader,
    val_loader,
    tokenizer,
    device,
    config
):

    """
    Main pretraining loop.

    High-level flow:

        dataset
            ↓
        tokens
            ↓
        model
            ↓
        logits
            ↓
        cross entropy
            ↓
        backward()
            ↓
        gradient accumulation
            ↓
        gradient clipping
            ↓
        Muon + AdamW
            ↓
        next step
    """

    print("\n[5/7] Building optimizers...")

    (
        muon_optimizer,
        adamw_optimizer
    ) = build_optimizers(
        model,
        config
    )

    # ------------------------------------------------------------
    # AMP scaler.
    #
    # FP16 can make training considerably faster and use less VRAM
    # on NVIDIA GPUs.
    # ------------------------------------------------------------

    use_amp = device.type == "cuda"

    scaler = torch.amp.GradScaler(
        enabled=use_amp
    )

    print("\n[6/7] Starting pretraining...")
    print("AMP enabled:", use_amp)
    print(
        "Gradient accumulation:",
        config.gradient_accumulation_steps
    )
    print(
        "Effective batch size:",
        config.batch_size
        * config.gradient_accumulation_steps
    )

    # ------------------------------------------------------------
    # Create iterators manually.
    #
    # We want training to run for "max_steps" rather than
    # exactly one pass through the DataLoader.
    # ------------------------------------------------------------

    train_iter = iter(train_loader)

    model.train()

    optimizer_step = 0

    running_loss = 0.0

    running_aux_loss = 0.0

    start_time = time.time()

    # ------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------

    while optimizer_step < config.max_steps:

        # Zero gradients at the beginning of an optimizer step.
        #
        # IMPORTANT:
        #
        # We do NOT zero gradients every micro batch because we
        # are accumulating them.
        #

        muon_optimizer.zero_grad(
            set_to_none=True
        )

        adamw_optimizer.zero_grad(
            set_to_none=True
        )

        accumulated_loss = 0.0

        accumulated_aux = 0.0

        # --------------------------------------------------------
        # Gradient accumulation loop.
        # --------------------------------------------------------

        for micro_step in range(
            config.gradient_accumulation_steps
        ):

            # ----------------------------------------------------
            # Get next batch.
            # ----------------------------------------------------

            try:

                input_ids, targets = next(
                    train_iter
                )

            except StopIteration:

                # If dataset ended, restart it.
                train_iter = iter(
                    train_loader
                )

                input_ids, targets = next(
                    train_iter
                )

            input_ids = input_ids.to(
                device,
                non_blocking=True
            )

            targets = targets.to(
                device,
                non_blocking=True
            )

            # ----------------------------------------------------
            # Forward + loss
            # ----------------------------------------------------

            with torch.cuda.amp.autocast(
                enabled=use_amp,
                dtype=torch.float16
            ):

                logits, moe_aux_loss = model(
                    input_ids
                )

                # ------------------------------------------------
                # Language modeling loss.
                #
                # logits:
                #
                # [B,T,V]
                #
                # targets:
                #
                # [B,T]
                #
                # Cross entropy expects:
                #
                # [N,C]
                #
                # so flatten B and T.
                # ------------------------------------------------

                language_loss = F.cross_entropy(
                    logits.reshape(
                        -1,
                        model.vocab_size
                    ),
                    targets.reshape(-1)
                )

                # ------------------------------------------------
                # Combine language loss with MoE balancing loss.
                # ------------------------------------------------

                total_loss = (
                    language_loss
                    + config.moe_aux_loss_weight
                    * moe_aux_loss
                )

                # Divide loss before backward because several
                # micro-batches will contribute to one optimizer
                # update.
                total_loss = (
                    total_loss
                    / config.gradient_accumulation_steps
                )

            # ----------------------------------------------------
            # Backpropagation.
            # ----------------------------------------------------

            scaler.scale(
                total_loss
            ).backward()

            accumulated_loss += (
                language_loss.item()
            )

            accumulated_aux += (
                moe_aux_loss.item()
            )

        # ========================================================
        # NOW PERFORM ONE REAL OPTIMIZER UPDATE
        # ========================================================

        # --------------------------------------------------------
        # Unscale gradients before clipping.
        #
        # Both optimizers need to be unscaled.
        # --------------------------------------------------------

        if use_amp:

            scaler.unscale_(
                muon_optimizer
            )

            scaler.unscale_(
                adamw_optimizer
            )

        # --------------------------------------------------------
        # Gradient clipping
        #
        # This prevents a giant gradient from destroying the
        # weights.
        # --------------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.grad_clip
        )

        # --------------------------------------------------------
        # Learning-rate schedule
        # --------------------------------------------------------

        multiplier = get_lr_multiplier(
            optimizer_step,
            config.warmup_steps,
            config.max_steps
        )

        current_muon_lr = set_learning_rate(
            muon_optimizer,
            config.muon_lr,
            multiplier
        )

        current_adamw_lr = set_learning_rate(
            adamw_optimizer,
            config.adamw_lr,
            multiplier
        )

        # --------------------------------------------------------
        # Update parameters.
        #
        # We have TWO optimizers:
        #
        #     Muon
        #     AdamW
        # --------------------------------------------------------

        scaler.step(
            muon_optimizer
        )

        scaler.step(
            adamw_optimizer
        )

        scaler.update()

        # --------------------------------------------------------
        # Statistics.
        # --------------------------------------------------------

        mean_loss = (
            accumulated_loss
            / config.gradient_accumulation_steps
        )

        mean_aux = (
            accumulated_aux
            / config.gradient_accumulation_steps
        )

        running_loss += mean_loss

        running_aux_loss += mean_aux

        optimizer_step += 1

        # --------------------------------------------------------
        # PRINT PROGRESS
        # --------------------------------------------------------

        if (
            optimizer_step % config.print_every == 0
            or optimizer_step == 1
        ):

            elapsed = (
                time.time()
                - start_time
            )

            steps_per_sec = (
                optimizer_step
                / max(elapsed, 1e-6)
            )

            print(
                f"\n"
                f"Step {optimizer_step:5d}/"
                f"{config.max_steps}"
            )

            print(
                f"Language loss : "
                f"{mean_loss:.4f}"
            )

            print(
                f"MoE aux loss  : "
                f"{mean_aux:.4f}"
            )

            print(
                f"Muon LR       : "
                f"{current_muon_lr:.6f}"
            )

            print(
                f"AdamW LR      : "
                f"{current_adamw_lr:.6f}"
            )

            print(
                f"Steps/sec     : "
                f"{steps_per_sec:.2f}"
            )

            # ----------------------------------------------------
            # Print GPU memory.
            # ----------------------------------------------------

            if device.type == "cuda":

                allocated = (
                    torch.cuda.memory_allocated(
                        device
                    )
                    / 1024**3
                )

                reserved = (
                    torch.cuda.memory_reserved(
                        device
                    )
                    / 1024**3
                )

                print(
                    f"VRAM allocated: "
                    f"{allocated:.2f} GB"
                )

                print(
                    f"VRAM reserved : "
                    f"{reserved:.2f} GB"
                )

    print("\nPretraining finished.")

    return model


# ================================================================
# 22. VALIDATION LOSS
# ================================================================

@torch.no_grad()
def evaluate_model(
    model,
    val_loader,
    device,
    max_batches=50
):

    """
    Calculate validation language-model loss.

    This gives us a basic answer to:

        "Is the model improving on unseen text?"
    """

    print("\nRunning validation...")

    model.eval()

    total_loss = 0.0

    batches = 0

    for input_ids, targets in val_loader:

        input_ids = input_ids.to(
            device
        )

        targets = targets.to(
            device
        )

        logits, moe_aux_loss = model(
            input_ids
        )

        loss = F.cross_entropy(
            logits.reshape(
                -1,
                model.vocab_size
            ),
            targets.reshape(-1)
        )

        total_loss += loss.item()

        batches += 1

        if batches >= max_batches:
            break

    average_loss = (
        total_loss
        / max(batches, 1)
    )

    print(
        f"Validation loss: "
        f"{average_loss:.4f}"
    )

    model.train()

    return average_loss


# ================================================================
# 23. SAVE MODEL
# ================================================================

def save_checkpoint(
    model,
    config,
    tokenizer,
    path
):

    """
    Save everything needed to reconstruct the pretrained model.

    We save:

        model weights
        GPT configuration
        tokenizer name

    Later we can load this checkpoint and perform:

        quantization
        QLoRA
        fine-tuning
        inference
    """

    print("\nSaving model...")

    # ------------------------------------------------------------
    # Move state dict tensors to CPU before saving.
    #
    # This makes the checkpoint more portable.
    # ------------------------------------------------------------

    cpu_state_dict = {
        key: value.cpu()
        for key, value
        in model.state_dict().items()
    }

    checkpoint = {

        # Learned parameters.
        "model_state_dict":
            cpu_state_dict,

        # Configuration.
        "gpt_config":
            asdict(config),

        # Tokenizer information.
        "tokenizer_name":
            tokenizer.name,

    }

    torch.save(
        checkpoint,
        path
    )

    print(
        f"Model saved to: {path}"
    )


# ================================================================
# 24. LOAD CHECKPOINT
# ================================================================

def load_checkpoint(
    path,
    device
):

    """
    This function is not needed for today's training run.

    It is included so you can understand what happens later
    when we load the pretrained model for quantization / QLoRA.
    """

    checkpoint = torch.load(
        path,
        map_location=device
    )

    config = GPTConfig(
        **checkpoint["gpt_config"]
    )

    tokenizer = tiktoken.get_encoding(
        checkpoint["tokenizer_name"]
    )

    model = ModernGPT(
        config,
        tokenizer.n_vocab
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)

    return (
        model,
        config,
        tokenizer
    )


# ================================================================
# 25. MAIN FUNCTION
# ================================================================

def main():

    """
    Everything executes sequentially from here.

    This is useful because you can mentally follow the complete
    pipeline:

        configuration
            ↓
        device
            ↓
        tokenizer
            ↓
        dataset
            ↓
        model
            ↓
        parameter count
            ↓
        optimizer
            ↓
        pretraining
            ↓
        validation
            ↓
        text generation
            ↓
        saving checkpoint
    """

    print("\n")
    print("=" * 70)
    print("MODERN MINI LLM - FROM SCRATCH")
    print("=" * 70)

    # ------------------------------------------------------------
    # STEP 0
    # ------------------------------------------------------------

    print("\n[0/7] Configuration")

    print(
        asdict(gpt_config)
    )

    # ------------------------------------------------------------
    # STEP 1
    # DEVICE
    # ------------------------------------------------------------

    device = get_device()

    # ------------------------------------------------------------
    # STEP 2
    # TOKENIZER
    # ------------------------------------------------------------

    tokenizer = load_tokenizer(
        gpt_config
    )

    # ------------------------------------------------------------
    # STEP 3
    # DATASET
    # ------------------------------------------------------------

    tokens = load_tokens(
        gpt_config,
        tokenizer
    )

    train_tokens, val_tokens = (
        split_dataset(tokens)
    )

    # ------------------------------------------------------------
    # Build PyTorch datasets.
    # ------------------------------------------------------------

    train_dataset = LanguageModelDataset(
        train_tokens,
        gpt_config.max_seq_len
    )

    val_dataset = LanguageModelDataset(
        val_tokens,
        gpt_config.max_seq_len
    )

    print(
        "Number of training samples:",
        len(train_dataset)
    )

    print(
        "Number of validation samples:",
        len(val_dataset)
    )

    # ------------------------------------------------------------
    # DataLoader
    #
    # num_workers=0 is intentional.
    #
    # On a machine with only 4 GB system RAM, spawning many worker
    # processes is not helpful.
    # ------------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=gpt_config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        )
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=gpt_config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        )
    )

    # ------------------------------------------------------------
    # STEP 4
    # MODEL
    # ------------------------------------------------------------

    print("\n[3/7] Building model...")

    model = ModernGPT(
        gpt_config,
        tokenizer.n_vocab
    )

    # Move model to GPU.
    model.to(device)

    # ------------------------------------------------------------
    # IMPORTANT MODEL INFORMATION
    # ------------------------------------------------------------

    print("\nArchitecture:")
    print(
        f"Layers          : "
        f"{gpt_config.n_layers}"
    )

    print(
        f"d_model         : "
        f"{gpt_config.d_model}"
    )

    print(
        f"Attention heads : "
        f"{gpt_config.n_heads}"
    )

    print(
        f"KV heads        : "
        f"{gpt_config.n_kv_heads}"
    )

    print(
        f"Experts         : "
        f"{gpt_config.n_experts}"
    )

    print(
        f"Active experts  : "
        f"{gpt_config.top_k_experts}"
    )

    print(
        f"Context length  : "
        f"{gpt_config.max_seq_len}"
    )

    print(
        f"Vocabulary      : "
        f"{tokenizer.n_vocab}"
    )

    # ------------------------------------------------------------
    # PARAMETER COUNT
    # ------------------------------------------------------------

    total_params = count_parameters(
        model
    )

    print_parameter_breakdown(
        model
    )

    # ------------------------------------------------------------
    # STEP 5
    # INITIAL TEST
    # ------------------------------------------------------------

    print("\n[4/7] Initial random-model test")

    prompt = (
        "The little girl"
    )

    print("\nPrompt:")
    print(prompt)

    print("\nRandom model output:")
    print(
        generate(
            model,
            tokenizer,
            prompt,
            device,
            gpt_config
        )
    )

    # This output will be nonsense because the model has not
    # learned anything yet.
    #
    # That is completely normal.

    # ------------------------------------------------------------
    # STEP 6
    # TRAIN
    # ------------------------------------------------------------

    model = train_model(
        model,
        train_loader,
        val_loader,
        tokenizer,
        device,
        gpt_config
    )

    # ------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------

    validation_loss = evaluate_model(
        model,
        val_loader,
        device
    )

    # ------------------------------------------------------------
    # STEP 7
    # GENERATION AFTER TRAINING
    # ------------------------------------------------------------

    print("\n[7/7] Testing trained model")

    # You can change this prompt.
    #
    # With a TinyStories-like dataset, prompts such as:
    #
    #   "Once upon a time"
    #
    #   "The little boy"
    #
    #   "One day the girl"
    #
    # are much better tests than completely unrelated text.

    test_prompt = (
        "Once upon a time"
    )

    print("\n" + "=" * 70)

    print("PROMPT:")
    print(test_prompt)

    print("\nGENERATED TEXT:")
    print("-" * 70)

    generated = generate(
        model,
        tokenizer,
        test_prompt,
        device,
        gpt_config
    )

    print(generated)

    print("=" * 70)

    # ------------------------------------------------------------
    # SAVE CHECKPOINT
    # ------------------------------------------------------------

    save_checkpoint(
        model,
        gpt_config,
        tokenizer,
        gpt_config.checkpoint_path
    )

    # ------------------------------------------------------------
    # FINAL INFORMATION
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Parameters : "
        f"{total_params / 1e6:.2f}M"
    )

    print(
        f"Validation : "
        f"{validation_loss:.4f}"
    )

    print(
        f"Checkpoint : "
        f"{gpt_config.checkpoint_path}"
    )

    print("\nNext stage:")
    print(
        "Load this checkpoint -> quantize -> QLoRA fine-tuning"
    )

    print("=" * 70)


# ================================================================
# 26. PYTHON ENTRY POINT
# ================================================================

if __name__ == "__main__":

    main()