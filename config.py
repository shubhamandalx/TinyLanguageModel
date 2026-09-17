"""
config.py
=========

All hyperparameters and settings for Modern Mini LLM live in a
single place: the `GPTConfig` dataclass below.

If you want to change the model size, training length, MoE
routing, or generation behavior, this is the only file you should
need to edit.
"""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    """
    Central configuration object for the model, optimizers,
    training loop, dataset, and text generation.

    Every other module in this project (model.py, dataset.py,
    optimizer.py, trainer.py, generate.py) receives this object
    and reads whatever fields it needs from it.
    """

    # ------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------

    # tiktoken vocabulary. cl100k_base is the tokenizer used by
    # several OpenAI models (~100k tokens).
    tokenizer_name: str = "cl100k_base"

    # ------------------------------------------------------------
    # Model / Transformer size
    # ------------------------------------------------------------

    # Number of Transformer blocks.
    n_layers: int = 8

    # Hidden representation size. Every token becomes a vector of
    # this size.
    d_model: int = 384

    # Total number of QUERY heads. d_model must be divisible by
    # n_heads.
    n_heads: int = 6

    # Number of KEY/VALUE heads (must be smaller than n_heads).
    # This is what makes attention "grouped query" attention:
    # every KV head is shared by (n_heads // n_kv_heads) Q heads.
    n_kv_heads: int = 2

    # Maximum number of tokens processed at once.
    max_seq_len: int = 256

    # ------------------------------------------------------------
    # Mixture of Experts (MoE)
    # ------------------------------------------------------------

    # Number of experts in each MoE layer.
    n_experts: int = 4

    # How many experts each token actually visits (top-k routing).
    top_k_experts: int = 2

    # Hidden size inside each expert's SwiGLU feed-forward network.
    expert_hidden_dim: int = 512

    # Coefficient for the MoE load-balancing auxiliary loss.
    moe_aux_loss_weight: float = 0.01

    # ------------------------------------------------------------
    # Regularization
    # ------------------------------------------------------------

    dropout: float = 0.0

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------

    # Micro batch size actually sent to the GPU at a time.
    batch_size: int = 4

    # Gradients are accumulated across this many micro-batches
    # before each optimizer step. Effective batch size becomes
    # batch_size * gradient_accumulation_steps.
    gradient_accumulation_steps: int = 8

    # Number of optimizer updates to run.
    max_steps: int = 2500

    # Number of steps spent linearly warming up the learning rate.
    warmup_steps: int = 200

    # Gradient clipping (max global norm).
    grad_clip: float = 1.0

    # ------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------

    # AdamW handles embeddings, norms, biases, and any other
    # non-matrix parameters.
    adamw_lr: float = 3e-4
    adamw_weight_decay: float = 0.1

    # Muon handles 2D matrix weights (attention/MoE projections).
    # Muon typically operates at a noticeably different LR scale
    # than AdamW.
    muon_lr: float = 2e-2
    muon_weight_decay: float = 0.01

    # ------------------------------------------------------------
    # Checkpointing / logging
    # ------------------------------------------------------------

    checkpoint_path: str = "checkpoints/modern_mini_llm.pt"

    print_every: int = 50

    # ------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------

    input_file: str = "data/input.txt"

    # ------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------

    generation_temperature: float = 0.8
    generation_top_k: int = 40
    generation_max_new_tokens: int = 60
