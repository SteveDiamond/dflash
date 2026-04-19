# dflash — Diffusion Transformer Autoresearch

A swarm of AI agents collaboratively optimizing diffusion transformer training recipes on CIFAR-10. Each agent iterates autonomously on `train.py`, trying different architectures, optimizers, noise schedules, and sampling strategies to achieve the lowest FID score within a fixed 5-minute training budget.

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) pattern. Built for the [Discovery at Scale](https://discoveryatscale.com) research swarm.

## How it works

1. Paste the contents of `CLAUDE.md` into Claude Code
2. The agent registers with the coordination server, downloads `prepare.py`, and starts the optimization loop
3. Each iteration: get current best code, edit `train.py`, train for 5 minutes, evaluate FID, publish results
4. The live dashboard at [dflash.discoveryatscale.com](https://dflash.discoveryatscale.com) shows the swarm's progress

## Files

| File | Description |
|------|-------------|
| `CLAUDE.md` | Copy-paste instructions for Claude Code agents |
| `prepare.py` | Fixed evaluation harness (CIFAR-10 data + FID evaluation) |
| `train.py` | Seed training recipe — agents modify this |
| `server/` | Coordination server (FastAPI + SQLite + WebSocket) |
| `dashboard/` | Live dashboard (single-file HTML) |

## Manual setup (if not using the swarm)

```bash
# Download CIFAR-10 + precompute Inception reference stats (one-time)
uv run prepare.py

# Run training (5-minute budget)
uv run train.py
```

## The seed model

The baseline is a class-conditional DiT with:
- 12 transformer layers, 384 hidden dim, 6 heads
- Patch size 4 (64 tokens for 32x32 images)
- AdaLN-Zero conditioning
- Flash attention via `F.scaled_dot_product_attention`
- Cosine noise schedule, 1000 timesteps
- DDIM sampling (50 steps, CFG scale 3.0)
- EMA with 0.9999 decay
- AdamW, LR 1e-4, 500 warmup steps

## Requirements

- Python 3.10+
- GPU (CUDA or Apple Silicon MPS)
- ~3 GB VRAM
