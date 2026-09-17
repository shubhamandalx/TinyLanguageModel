from dataclasses import dataclass


@dataclass
class GPTConfig:
    
    tokenizer_name: str = "cl100k_base"
    n_layers: int = 8
    d_model: int = 384
    n_heads: int = 6
    n_kv_heads: int = 2
    max_seq_len: int = 256
    n_experts: int = 4
    top_k_experts: int = 2
    expert_hidden_dim: int = 512
    moe_aux_loss_weight: float = 0.01
    dropout: float = 0.0
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_steps: int = 2500
    warmup_steps: int = 200
    grad_clip: float = 1.0
    adamw_lr: float = 3e-4
    adamw_weight_decay: float = 0.1
    muon_lr: float = 2e-2
    muon_weight_decay: float = 0.01
    checkpoint_path: str = "checkpoints/modern_mini_llm.pt"
    print_every: int = 50
    input_file: str = "data/input.txt"
    generation_temperature: float = 0.8
    generation_top_k: int = 40
    generation_max_new_tokens: int = 60
