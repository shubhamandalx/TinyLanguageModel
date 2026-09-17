from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tiktoken import Encoding

from config import GPTConfig
from tokenizer import encode


def load_tokens(config: GPTConfig, tokenizer: Encoding) -> Tensor:
  

    with open(config.input_file, "r", encoding="utf-8") as f:
        text = f.read()

    print("Characters in dataset:", len(text))

    token_ids = encode(tokenizer, text, allowed_special=("<|endoftext|>",))

    print("Number of tokens:", len(token_ids))

    return torch.tensor(token_ids, dtype=torch.long)


def split_dataset(tokens: Tensor, val_fraction: float = 0.05) -> Tuple[Tensor, Tensor]:
   

    split = int(len(tokens) * (1.0 - val_fraction))

    train_tokens = tokens[:split]
    val_tokens = tokens[split:]

    print("Train tokens:", len(train_tokens))
    print("Validation tokens:", len(val_tokens))

    return train_tokens, val_tokens


class LanguageModelDataset(Dataset):
   

    def __init__(self, tokens: Tensor, seq_len: int) -> None:
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self) -> int:
        
        return len(self.tokens) - self.seq_len - 1

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        x = self.tokens[index: index + self.seq_len]
        y = self.tokens[index + 1: index + self.seq_len + 1]
        return x, y


def create_dataloaders(
    config: GPTConfig,
    tokenizer: Encoding,
    device: torch.device,
) -> Tuple[DataLoader, DataLoader]:
    

    tokens = load_tokens(config, tokenizer)

    train_tokens, val_tokens = split_dataset(tokens)

    train_dataset = LanguageModelDataset(train_tokens, config.max_seq_len)
    val_dataset = LanguageModelDataset(val_tokens, config.max_seq_len)

    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader
