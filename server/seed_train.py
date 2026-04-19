# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.0",
#     "torchvision>=0.15",
#     "scipy>=1.10",
#     "numpy>=1.24",
# ]
# ///
"""
dflash: Flash Diffusion Transformer autoresearch. Single-GPU, single-file.
Train a class-conditional DiT on CIFAR-10 with flash attention.
Usage: uv run train.py
"""

import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import contextlib
import gc
import math
import time
from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (TIME_BUDGET, IMAGE_SIZE, NUM_CHANNELS, NUM_CLASSES,
                     NUM_EVAL_SAMPLES, download_data, make_dataloader, evaluate_fid)

# ---------------------------------------------------------------------------
# Diffusion Transformer
# ---------------------------------------------------------------------------

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, dim, freq_dim=256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    @staticmethod
    def sinusoidal(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t):
        return self.mlp(self.sinusoidal(t, self.freq_dim))


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(num_classes + int(dropout > 0), dim)
        self.num_classes = num_classes
        self.dropout = dropout

    def forward(self, labels, train):
        if self.dropout > 0 and train:
            drop = torch.rand(labels.shape[0], device=labels.device) < self.dropout
            labels = torch.where(drop, self.num_classes, labels)
        return self.embedding(labels)


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_dim, dim),
        )
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, c):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaLN(c).chunk(6, dim=1)
        x = x + gate_a.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_a, scale_a))
        x = x + gate_m.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_m, scale_m))
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim, patch_size, out_channels):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_channels)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x, c):
        shift, scale = self.adaLN(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm(x), shift, scale))


@dataclass
class DiTConfig:
    image_size: int = 32
    patch_size: int = 4
    in_channels: int = 3
    hidden_size: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    class_dropout: float = 0.1
    num_classes: int = 10


class DiT(nn.Module):
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config
        self.out_channels = config.in_channels
        num_patches = (config.image_size // config.patch_size) ** 2

        self.x_embedder = nn.Linear(
            config.patch_size ** 2 * config.in_channels, config.hidden_size
        )
        self.t_embedder = TimestepEmbedder(config.hidden_size)
        self.y_embedder = LabelEmbedder(
            config.num_classes, config.hidden_size, config.class_dropout
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, config.hidden_size))
        self.blocks = nn.ModuleList([
            DiTBlock(config.hidden_size, config.num_heads, config.mlp_ratio)
            for _ in range(config.depth)
        ])
        self.final_layer = FinalLayer(
            config.hidden_size, config.patch_size, self.out_channels
        )
        self._init_weights()

    def _init_weights(self):
        def _basic(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(_basic)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.y_embedder.embedding.weight, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN[-1].weight)
            nn.init.zeros_(block.adaLN[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def patchify(self, x):
        B, C, H, W = x.shape
        p = self.config.patch_size
        return x.reshape(B, C, H // p, p, W // p, p).permute(0, 2, 4, 1, 3, 5).reshape(B, -1, C * p * p)

    def unpatchify(self, x):
        p = self.config.patch_size
        h = w = self.config.image_size // p
        C = self.out_channels
        return x.reshape(-1, h, w, C, p, p).permute(0, 3, 1, 4, 2, 5).reshape(-1, C, h * p, w * p)

    def forward(self, x, t, y):
        x = self.x_embedder(self.patchify(x)) + self.pos_embed
        c = self.t_embedder(t) + self.y_embedder(y, self.training)
        for block in self.blocks:
            x = block(x, c)
        return self.unpatchify(self.final_layer(x, c))

# ---------------------------------------------------------------------------
# Diffusion
# ---------------------------------------------------------------------------

def cosine_schedule(T, s=0.008):
    t = torch.linspace(0, T, T + 1)
    alpha_bar = torch.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(0.0001, 0.9999)


class Diffusion:
    def __init__(self, T=1000, device="cuda"):
        self.T = T
        betas = cosine_schedule(T).to(device)
        alpha = 1 - betas
        self.alpha_bar = torch.cumprod(alpha, 0)
        self.sqrt_ab = self.alpha_bar.sqrt()
        self.sqrt_1m_ab = (1 - self.alpha_bar).sqrt()

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.sqrt_ab[t][:, None, None, None]
        b = self.sqrt_1m_ab[t][:, None, None, None]
        return a * x0 + b * noise

    def loss(self, model, x0, t, y):
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        return F.mse_loss(model(xt, t, y), noise)

    @torch.no_grad()
    def sample_ddim(self, model, n, steps=50, cfg=3.0, device="cuda"):
        x = torch.randn(n, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device=device)
        labels = torch.randint(0, NUM_CLASSES, (n,), device=device)
        ts = torch.linspace(self.T - 1, 0, steps, device=device).long()

        if cfg > 1:
            null_labels = torch.full_like(labels, NUM_CLASSES)
            y_combined = torch.cat([labels, null_labels])

        for i, t in enumerate(ts):
            tb = t.expand(n)
            if cfg > 1:
                eps = model(torch.cat([x, x]), torch.cat([tb, tb]), y_combined)
                cond, uncond = eps.chunk(2)
                eps = uncond + cfg * (cond - uncond)
            else:
                eps = model(x, tb, labels)

            ab = self.alpha_bar[t]
            ab_prev = self.alpha_bar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=device)

            x0_pred = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
            x = ab_prev.sqrt() * x0_pred + (1 - ab_prev).sqrt() * eps

        return x

# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            for sp, mp in zip(self.shadow.parameters(), model.parameters()):
                sp.lerp_(mp, 1 - self.decay)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these freely)
# ---------------------------------------------------------------------------

# Architecture
DEPTH = 12               # transformer layers
HIDDEN_SIZE = 384         # embedding dimension
NUM_HEADS = 6             # attention heads
PATCH_SIZE = 4            # image patch size (32/4 = 8x8 = 64 tokens)
MLP_RATIO = 4.0           # MLP hidden dim multiplier
CLASS_DROPOUT = 0.1       # label dropout for classifier-free guidance

# Optimization
LR = 1e-4                 # peak learning rate
BETAS = (0.9, 0.999)      # Adam betas
WEIGHT_DECAY = 0.0        # AdamW weight decay
WARMUP_STEPS = 500        # LR warmup steps
GRAD_CLIP = 1.0           # max gradient norm (0 to disable)
BATCH_SIZE = 128          # images per step

# Diffusion
DIFF_STEPS = 1000         # total diffusion timesteps

# Sampling
DDIM_STEPS = 50           # DDIM sampling steps
CFG_SCALE = 3.0           # classifier-free guidance scale

# EMA
EMA_DECAY = 0.9999        # exponential moving average decay

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Device: {device}")

if device.type == "cuda":
    torch.set_float32_matmul_precision("high")

use_amp = device.type == "cuda"
amp_ctx = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
           if use_amp else contextlib.nullcontext())

download_data()

config = DiTConfig(
    image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, in_channels=NUM_CHANNELS,
    hidden_size=HIDDEN_SIZE, depth=DEPTH, num_heads=NUM_HEADS,
    mlp_ratio=MLP_RATIO, class_dropout=CLASS_DROPOUT, num_classes=NUM_CLASSES,
)
model = DiT(config).to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_params:,} ({num_params / 1e6:.1f}M)")
print(f"Config: depth={DEPTH} hidden={HIDDEN_SIZE} heads={NUM_HEADS} patch={PATCH_SIZE}")

diffusion = Diffusion(DIFF_STEPS, device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=BETAS,
                              weight_decay=WEIGHT_DECAY)
ema = EMA(model, EMA_DECAY)

loader = make_dataloader(BATCH_SIZE, "train")
images, labels = next(loader)

print(f"Time budget: {TIME_BUDGET}s")
print(f"Batch size: {BATCH_SIZE}")


def lr_schedule(step):
    if step < WARMUP_STEPS:
        return LR * step / max(WARMUP_STEPS, 1)
    return LR

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_train_start = time.time()
total_time = 0
step = 0
smooth_loss = 0

while True:
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()

    x = images.to(device, non_blocking=True)
    y = labels.to(device, non_blocking=True)
    t = torch.randint(0, DIFF_STEPS, (x.shape[0],), device=device)

    with amp_ctx:
        loss = diffusion.loss(model, x, t, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    if GRAD_CLIP > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

    lr = lr_schedule(step)
    for g in optimizer.param_groups:
        g["lr"] = lr

    optimizer.step()
    ema.update(model)

    images, labels = next(loader)
    lv = loss.item()

    if math.isnan(lv) or lv > 100:
        print("FAIL")
        exit(1)

    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0

    if step > 5:
        total_time += dt

    ema_beta = 0.9
    smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * lv
    dloss = smooth_loss / (1 - ema_beta ** (step + 1))
    pct = min(total_time / TIME_BUDGET, 1) * 100
    rem = max(0, TIME_BUDGET - total_time)

    if step % 10 == 0:
        print(f"\rstep {step:05d} ({pct:.1f}%) | loss: {dloss:.6f} | lr: {lr:.2e} "
              f"| dt: {dt * 1000:.0f}ms | remaining: {rem:.0f}s    ",
              end="", flush=True)

    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()

    step += 1
    if step > 5 and total_time >= TIME_BUDGET:
        break

print()
total_images = step * BATCH_SIZE

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

print("Generating samples for FID evaluation...")
ema.shadow.eval()

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)


def sample_fn(n):
    with torch.no_grad(), amp_ctx:
        return diffusion.sample_ddim(ema.shadow, n, DDIM_STEPS, CFG_SCALE, device)


val_fid = evaluate_fid(sample_fn, NUM_EVAL_SAMPLES, device)

# Validation loss
model.eval()
val_loader = make_dataloader(BATCH_SIZE, "val")
vl = []
for _ in range(20):
    vi, vy = next(val_loader)
    vi, vy = vi.to(device), vy.to(device)
    vt = torch.randint(0, DIFF_STEPS, (vi.shape[0],), device=device)
    with torch.no_grad(), amp_ctx:
        vl.append(diffusion.loss(ema.shadow, vi, vt, vy).item())
val_loss = sum(vl) / len(vl)

# Summary
t_end = time.time()
peak_vram = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0

print("---")
print(f"val_fid:          {val_fid:.1f}")
print(f"val_loss:         {val_loss:.6f}")
print(f"training_seconds: {total_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram:.1f}")
print(f"total_images_K:   {total_images / 1000:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
