"""
tokenizer.py
============

Thin, independent wrapper around `tiktoken`.

The rest of the project (dataset.py, generate.py, train.py) only
ever calls `get_tokenizer()`, `encode()`, and `decode()` — nothing
downstream needs to know that tiktoken is the library underneath.
This keeps the tokenizer swappable later without touching other
files.
"""

from typing import Iterable, List

import tiktoken
from tiktoken import Encoding

from config import GPTConfig


def get_tokenizer(config: GPTConfig) -> Encoding:
    """
    Load the tiktoken tokenizer named in `config.tokenizer_name`.

    Text -> tokenizer -> token IDs -> neural network.

    Example: "Hello world" becomes something conceptually like
    [9906, 1917]. These integers are what the model actually sees.
    """

    print("Loading tokenizer:", config.tokenizer_name)

    tokenizer = tiktoken.get_encoding(config.tokenizer_name)

    print("Vocabulary size:", tokenizer.n_vocab)

    return tokenizer


def encode(
    tokenizer: Encoding,
    text: str,
    allowed_special: Iterable[str] = ("<|endoftext|>",),
) -> List[int]:
    """
    Encode a string into a list of token IDs.

    `allowed_special` controls which special tokens (like the
    end-of-text marker) are allowed to appear literally in the
    input text rather than being rejected.
    """

    return tokenizer.encode(text, allowed_special=set(allowed_special))


def decode(tokenizer: Encoding, token_ids: List[int]) -> str:
    """
    Decode a list of token IDs back into a human-readable string.
    """

    return tokenizer.decode(token_ids)
