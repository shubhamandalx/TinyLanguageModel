# TinyLanguageModel

A small decoder-only Transformer language model implemented from
scratch in PyTorch and trained using a custom pretraining
pipeline.

## Features

- PyTorch (no Hugging Face Transformers dependency)
- Decoder-only Transformer
- RMSNorm
- RoPE (Rotary Positional Embeddings)
- Grouped Query Attention (GQA)
- SwiGLU feed-forward networks
- Mixture of Experts (MoE) with top-k expert routing and a load-balancing auxiliary loss
- Muon optimizer for 2D matrix weights
- AdamW for embeddings, norms, and biases
- Mixed-precision (FP16) training on CUDA
- Gradient accumulation
- Cosine learning-rate schedule with linear warmup
- Gradient clipping
- Weight tying between the token embedding and the LM head
- Autoregressive text generation (temperature + top-k sampling)
- Checkpointing (`torch.save` / `torch.load`)

## Architecture

```
Text
  ↓
Tokenizer (tiktoken)
  ↓
Token IDs
  ↓
Embedding
  ↓
Transformer Blocks × n_layers
  ↓
RMSNorm
  ↓
LM Head (tied to Embedding)
  ↓
Logits
  ↓
Next-token prediction
```

Each Transformer block uses a **pre-norm** structure:

```
x
 └─ RMSNorm → Grouped Query Attention (+ RoPE) → + residual
 └─ RMSNorm → Mixture of Experts (SwiGLU experts) → + residual
```

Attention uses fewer key/value heads than query heads (GQA), with
each KV head shared by a group of query heads. The feed-forward
sub-layer is a sparse Mixture of Experts: a small router sends each
token to its top-`k` experts (out of `n_experts` total), so the
model has more total parameters than it actually computes per
token.

## Repository Structure

```
modern-mini-llm/
├── config.py        # All hyperparameters (GPTConfig dataclass)
├── tokenizer.py      # tiktoken wrapper: get_tokenizer / encode / decode
├── dataset.py        # Reading, tokenizing, splitting data; DataLoader creation
├── model.py          # Full model architecture (RMSNorm, RoPE, GQA, MoE, ModernGPT)
├── optimizer.py       # Muon optimizer + build_optimizers()
├── trainer.py         # Trainer class: training loop, validation, checkpointing
├── generate.py        # Autoregressive generation + CLI
├── train.py           # Entry point that wires everything together
├── data/              # Put your training corpus here (data/input.txt)
├── checkpoints/        # Saved model checkpoints land here
├── requirements.txt
└── .gitignore
```

A quick tour, in the order you'd read them to understand the
project:

- **`config.py`** — every hyperparameter, in one dataclass. Start here.
- **`tokenizer.py`** — thin wrapper around `tiktoken`; independent of the model.
- **`dataset.py`** — turns `data/input.txt` into `(input, target)` training batches.
- **`model.py`** — the entire network, RMSNorm through the final LM head.
- **`optimizer.py`** — splits parameters between Muon (matrices) and AdamW (everything else).
- **`trainer.py`** — the `Trainer` class: runs the training loop, validation, and checkpoint saving.
- **`generate.py`** — loads a checkpoint and samples text from a prompt.
- **`train.py`** — the script you actually run; orchestrates all of the above.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Dataset

Place your training corpus at:

```
data/input.txt
```

A single, plain-text, UTF-8 file. See `data/README.md`. The
dataset itself is not committed to this repository — bring your
own corpus (for example, something like TinyStories works well for
a model this size).

## Training

```bash
python train.py
```

This will: print the configuration, pick a device, build the
tokenizer and dataset, build the model, print parameter counts,
sample from the untrained model as a sanity check, run pretraining
for `config.max_steps` optimizer steps, run validation, save a
checkpoint to `checkpoints/modern_mini_llm.pt`, and sample from the
trained model.

## Generation

```bash
python generate.py --prompt "Once upon a time"
```

Optional flags:

```bash
python generate.py \
    --prompt "The little girl" \
    --checkpoint checkpoints/modern_mini_llm.pt \
    --temperature 0.8 \
    --top-k 40 \
    --max-new-tokens 100
```

## Hardware

This project was designed and tested against a target of a single
consumer NVIDIA GPU with **~8 GB VRAM** (e.g. an RTX 2070), using
the default configuration (`d_model=384`, `n_layers=8`,
`max_seq_len=256`, `batch_size=4` with 8-step gradient
accumulation). It also runs on CPU (much more slowly — CPU is fine
for smoke-testing the pipeline, not for a full pretraining run).
No other hardware configurations have been tested; if you hit
out-of-memory errors, lower `batch_size`, `max_seq_len`, `d_model`,
or `n_layers` in `config.py`.

## Configuration

All model, training, MoE, optimizer, checkpoint, dataset, and
generation settings live in `config.py` (the `GPTConfig`
dataclass). Edit the defaults there, or override individual fields
in your own script before constructing the model/trainer.

## Limitations

- No KV cache: generation recomputes attention over the full
  growing context on every new token, so it gets slower as the
  output grows.
- No quantization or QLoRA support.
- No custom fused/FlashAttention kernel — relies on PyTorch's
  `scaled_dot_product_attention`, which may or may not dispatch to
  a fused kernel depending on your PyTorch/CUDA build.
- Single-GPU / single-process only; no distributed training.
- Pretraining only — no instruction-tuning stage.
- The MoE expert-routing loop in `model.py` is written for
  readability, not throughput; it is not a production-grade batched
  MoE kernel.

## Roadmap

Planned future work (not currently implemented):

- KV cache for faster generation
- Quantization (e.g. int8 / int4 weight-only)
- QLoRA fine-tuning on top of a quantized checkpoint
- A custom FlashAttention-style fused kernel
- Distributed / multi-GPU training
- An instruction-tuning stage on top of the pretrained base model
