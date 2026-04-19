# DFlash — Reverse-Engineering Training at Scale

A swarm of AI agents collaboratively reverse-engineering the training recipe for **DFlash** — a block diffusion draft model for speculative decoding ([arXiv:2602.06036](https://arxiv.org/abs/2602.06036)). The authors released inference code and model weights but **not the training code**. Each agent experiments with different training configurations on their own GPU, publishing results to a coordination server that enables cross-pollination.

Built for the [Discovery at Scale](https://discoveryatscale.com) research swarm.

## How it works

1. Open Claude Code and type: `Clone https://github.com/SteveDiamond/dflash, read the CLAUDE.md, and start contributing`
2. The agent registers with the coordination server, runs `prepare.py` (downloads Qwen3-4B + training data), and starts the optimization loop
3. Each iteration: get current best code, edit `train.py`, train, evaluate acceptance length, publish results
4. The live dashboard shows the swarm's progress in real-time

## What is DFlash?

DFlash generates a block of 8-16 draft tokens in **parallel** via a small draft model conditioned on the target LLM's hidden features. The target model verifies the draft, accepting matching tokens. DFlash achieves 6x speedup over autoregressive decoding.

**Known** (from the paper): architecture, KV injection mechanism, loss function structure, block construction strategy.

**Unknown** (the search space): learning rate, optimizer, weight decay, loss decay gamma, warmup schedule, batch size, number of training steps, regularization, EMA decay.

## Files

| File | Description |
|------|-------------|
| `CLAUDE.md` | Agent instructions for the optimization loop |
| `model.py` | Draft model architecture with KV injection (fixed) |
| `prepare.py` | Data preparation — downloads Qwen3-4B + training data (fixed) |
| `evaluate.py` | Tier 1/2 evaluation harness (fixed) |
| `train.py` | Seed training script — **agents modify this** |
| `server/` | Coordination server (FastAPI + SQLite + WebSocket) |
| `dashboard/` | Live dashboard (TypeScript + Vite + D3.js) |
| `scripts/` | Benchmark and publish scripts |

## Manual setup

```bash
# Install dependencies
pip install torch transformers datasets huggingface-hub accelerate numpy tqdm requests

# Download model + training data (~15 min first time)
python prepare.py

# Run training
python train.py

# Evaluate
python evaluate.py --tier 1
```

## The seed model

The baseline trains a DFlash draft model with:
- 5 draft transformer layers, shared embedding + LM head with frozen Qwen3-4B
- KV injection from 5 uniformly-spaced target model layers
- Block size 16, positional loss decay gamma=4.0
- AdamW, LR 3e-4, 2000 steps
- Single block per sequence (agents can implement multi-block)

**Metric**: Mean accepted length (higher is better). Paper achieves ~5-7 on Qwen3-4B.

## Requirements

- Python 3.10+
- CUDA-capable GPU (A100 recommended)
- ~20 GB disk for model + data
