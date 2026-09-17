import argparse

import torch
import torch.nn.functional as F
from torch import Tensor

from config import GPTConfig
from model import ModernGPT
from tokenizer import decode, encode
from trainer import load_checkpoint


@torch.no_grad()
def generate(
    model: ModernGPT,
    tokenizer,
    prompt: str,
    device: torch.device,
    config: GPTConfig,
) -> str:
    

    model.eval()

    tokens = encode(tokenizer, prompt, allowed_special=())
    input_ids: Tensor = torch.tensor([tokens], dtype=torch.long, device=device)

    for _ in range(config.generation_max_new_tokens):
        input_for_model = input_ids[:, -config.max_seq_len:]

        logits, _ = model(input_for_model)
        next_token_logits = logits[:, -1, :]

        
        next_token_logits = next_token_logits / config.generation_temperature

      
        if config.generation_top_k is not None:
            k = min(config.generation_top_k, next_token_logits.size(-1))
            values, _ = torch.topk(next_token_logits, k)
            threshold = values[:, -1].unsqueeze(-1)
            next_token_logits = torch.where(
                next_token_logits < threshold,
                torch.full_like(next_token_logits, float("-inf")),
                next_token_logits,
            )

        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_token], dim=1)

    return decode(tokenizer, input_ids[0].tolist())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a Modern Mini LLM checkpoint")

    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Text prompt to continue")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/modern_mini_llm.pt",
        help="Path to a checkpoint produced by train.py",
    )
    parser.add_argument("--temperature", type=float, default=None, help="Override generation temperature")
    parser.add_argument("--top-k", type=int, default=None, help="Override generation top-k")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override number of tokens to generate")

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    print(f"Loading checkpoint: {args.checkpoint}")
    model, config, tokenizer = load_checkpoint(args.checkpoint, device)

    if args.temperature is not None:
        config.generation_temperature = args.temperature
    if args.top_k is not None:
        config.generation_top_k = args.top_k
    if args.max_new_tokens is not None:
        config.generation_max_new_tokens = args.max_new_tokens

    print("\nPrompt:")
    print(args.prompt)

    text = generate(model, tokenizer, args.prompt, device, config)

    print("\nGenerated text:")
    print("-" * 70)
    print(text)
    print("-" * 70)


if __name__ == "__main__":
    main()
