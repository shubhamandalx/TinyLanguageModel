"""
generate.py
===========

Autoregressive text generation from a trained checkpoint.

Usage:

    python generate.py --prompt "Once upon a time"
    python generate.py --prompt "The little girl" --checkpoint checkpoints/modern_mini_llm.pt \\
        --temperature 0.8 --top-k 40 --max-new-tokens 100
"""

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
    stop_at_eot: bool = True,
) -> str:
    """
    Autoregressive generation: repeatedly predict the next token,
    append it to the context, and repeat.

    Corpora such as TinyStories place the special token
    "<|endoftext|>" after every story, so the model learns it as a
    natural "this story is finished" signal. If `stop_at_eot` is
    True (the default), generation stops the moment that token is
    produced instead of silently continuing past it and sampling
    what would effectively be the start of a new, unrelated story
    right after it.

    NOTE: this recomputes attention over the entire growing context
    on every step (no KV cache). See the README's roadmap section
    for future-work notes.
    """

    model.eval()

    tokens = encode(tokenizer, prompt, allowed_special=())
    input_ids: Tensor = torch.tensor([tokens], dtype=torch.long, device=device)

    eot_id = encode(tokenizer, "<|endoftext|>", allowed_special=("<|endoftext|>",))[0]

    for _ in range(config.generation_max_new_tokens):
        input_for_model = input_ids[:, -config.max_seq_len:]

        logits, _ = model(input_for_model)
        next_token_logits = logits[:, -1, :]

        # Temperature: lower = more deterministic, higher = more random.
        next_token_logits = next_token_logits / config.generation_temperature

        # Top-k sampling: only consider the k highest-scoring tokens.
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

        if stop_at_eot and next_token.item() == eot_id:
            break

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
    parser.add_argument(
        "--no-stop-at-eot",
        action="store_true",
        help="Keep generating past the <|endoftext|> marker instead of stopping there "
             "(useful if you deliberately want the model to keep going into a new story)",
    )

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

    text = generate(
        model, tokenizer, args.prompt, device, config,
        stop_at_eot=not args.no_stop_at_eot,
    )

    ended_naturally = text.endswith("<|endoftext|>")
    if ended_naturally:
        text = text[: -len("<|endoftext|>")].rstrip()

    print("\nGenerated text:")
    print("-" * 70)
    print(text)
    print("-" * 70)
    if ended_naturally:
        print("(model reached its own end-of-story marker)")


if __name__ == "__main__":
    main()
