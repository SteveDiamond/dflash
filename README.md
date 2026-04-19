# dflash

Flash Diffusion Transformer autoresearch on CIFAR-10.

An autonomous AI agent modifies `train.py` — a class-conditional [DiT](https://arxiv.org/abs/2212.09748) with flash attention — runs 5-minute training experiments, evaluates FID, and iterates. Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

## Setup

```bash
# Install dependencies
uv sync

# Download CIFAR-10 + precompute Inception reference stats (one-time)
uv run prepare.py
```

## Run

```bash
uv run train.py
```

Trains for 5 minutes (wall clock), then evaluates FID on 10K generated samples. Output:

```
val_fid:          125.3
val_loss:         0.042100
training_seconds: 300.1
num_steps:        6000
num_params_M:     33.1
```

## How it works

| File | Role |
|------|------|
| `train.py` | **The file agents edit.** DiT model, diffusion process, optimizer, training loop, sampling. Everything is fair game. |
| `prepare.py` | Fixed. Data loading, FID evaluation. Do not modify. |
| `program.md` | Instructions for autonomous agents. |

The metric is **val_fid** (FID score, lower is better). The agent edits `train.py`, commits, runs a 5-minute experiment, and keeps or discards based on whether FID improved. See `program.md` for the full loop.

## Requirements

- Python 3.10+
- GPU (CUDA or Apple Silicon MPS)
- ~3 GB VRAM
