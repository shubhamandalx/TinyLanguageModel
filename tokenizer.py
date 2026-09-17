from typing import Iterable, List

import tiktoken
from tiktoken import Encoding

from config import GPTConfig


def get_tokenizer(config: GPTConfig) -> Encoding:
    

    print("Loading tokenizer:", config.tokenizer_name)

    tokenizer = tiktoken.get_encoding(config.tokenizer_name)

    print("Vocabulary size:", tokenizer.n_vocab)

    return tokenizer


def encode(
    tokenizer: Encoding,
    text: str,
    allowed_special: Iterable[str] = ("<|endoftext|>",),
) -> List[int]:
    

    return tokenizer.encode(text, allowed_special=set(allowed_special))


def decode(tokenizer: Encoding, token_ids: List[int]) -> str:
    
    return tokenizer.decode(token_ids)
