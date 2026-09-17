"""
train.py
========

Entry point that orchestrates a full pretraining run, start to
finish. This file intentionally contains no model or training-loop
implementation details — just the sequence of steps:

    1. load configuration
    2. select device
    3. create tokenizer
    4. create train/validation DataLoaders
    5. create model
    6. print model information
    7. create Trainer
    8. train
    9. evaluate
    10. save checkpoint
    11. run a short generation test

Run with:

    python train.py
"""

import os
from dataclasses import asdict

import torch

from config import GPTConfig
from tokenizer import get_tokenizer
from dataset import create_dataloaders
from model import ModernGPT, count_parameters, print_parameter_breakdown
from trainer import Trainer
from generate import generate


def main() -> None:
    # 1. Configuration
    config = GPTConfig()
    print("Configuration:")
    print(asdict(config))

    # 2. Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nDevice:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.set_float32_matmul_precision("high")

    # 3. Tokenizer
    tokenizer = get_tokenizer(config)

    # 4. DataLoaders
    train_loader, val_loader = create_dataloaders(config, tokenizer, device)

    # 5. Model
    print("\nBuilding model...")
    model = ModernGPT(config, tokenizer.n_vocab)
    model.to(device)

    # 6. Model information
    print("\nArchitecture:")
    print(f"Layers          : {config.n_layers}")
    print(f"d_model         : {config.d_model}")
    print(f"Attention heads : {config.n_heads}")
    print(f"KV heads        : {config.n_kv_heads}")
    print(f"Experts         : {config.n_experts}")
    print(f"Active experts  : {config.top_k_experts}")
    print(f"Context length  : {config.max_seq_len}")
    print(f"Vocabulary      : {tokenizer.n_vocab}")

    total_params = count_parameters(model)
    print_parameter_breakdown(model)

    # Sanity check: generation from the untrained (random) model.
    # This output is expected to be nonsense.
    print("\nRandom-model sample (before training):")
    print(generate(model, tokenizer, "The little girl", device, config))

    # 7. Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        tokenizer_name=config.tokenizer_name,
    )

    # 8. Train
    model = trainer.train()

    # 9. Evaluate
    validation_loss = trainer.evaluate()

    # 10. Save checkpoint
    os.makedirs(os.path.dirname(config.checkpoint_path) or ".", exist_ok=True)
    trainer.save_checkpoint()

    # 11. Generation test after training
    print("\nSample after training:")
    print("-" * 70)
    print(generate(model, tokenizer, "Once upon a time", device, config))
    print("-" * 70)

    print("\nTraining complete.")
    print(f"Parameters : {total_params / 1e6:.2f}M")
    print(f"Validation : {validation_loss:.4f}")
    print(f"Checkpoint : {config.checkpoint_path}")


if __name__ == "__main__":
    main()
