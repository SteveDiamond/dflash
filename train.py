"""
DFlash draft model training script.
THIS IS THE FILE YOU MODIFY — experiment with hyperparameters, training
strategies, loss functions, and optimization to maximize acceptance length.

Usage: uv run train.py
"""

import os
import gc
import json
import math
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from model import (
    DFlashConfig,
    DFlashDraftModel,
    extract_context_features,
    load_target_model,
    compute_position_weights,
)
from prepare import CACHE_DIR, TARGET_MODEL, TRAIN_DATA_PATH, EVAL_DATA_PATH

# ---------------------------------------------------------------------------
# Hyperparameters (edit these freely)
# ---------------------------------------------------------------------------

# Architecture
NUM_DRAFT_LAYERS = 5          # number of draft transformer layers
NUM_TARGET_FEATURES = 5       # number of target model layers to extract features from
BLOCK_SIZE = 16               # tokens per block (paper uses 16 for Qwen, 10 for LLaMA)

# Optimization
LR = 3e-4                    # peak learning rate
OPTIMIZER = "adamw"           # optimizer: "adamw", "adam", "sgd"
BETAS = (0.9, 0.999)         # Adam betas
WEIGHT_DECAY = 0.0            # weight decay
WARMUP_STEPS = 100            # LR warmup steps
LR_SCHEDULE = "constant"     # "constant", "cosine", "linear"
GRAD_CLIP = 1.0               # max gradient norm (0 to disable)
BATCH_SIZE = 4                # sequences per step (limited by VRAM)

# Loss
GAMMA = 4.0                  # positional weight decay (w_k = exp(-(k-1)/gamma))
LABEL_SMOOTHING = 0.0         # label smoothing for cross-entropy

# Training
NUM_STEPS = 2000              # total training steps
BLOCKS_PER_SEQ = 1            # training blocks sampled per sequence
SEED = 42                     # random seed

# EMA
USE_EMA = False               # use exponential moving average
EMA_DECAY = 0.999             # EMA decay rate

# Mask token
MASK_TOKEN_ID = 0             # token ID used for masked positions

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

torch.manual_seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    torch.set_float32_matmul_precision("high")
    torch.cuda.manual_seed(SEED)

use_amp = device.type == "cuda"
amp_dtype = torch.bfloat16

# ---------------------------------------------------------------------------
# Load target model (frozen)
# ---------------------------------------------------------------------------

print(f"Loading target model: {TARGET_MODEL}")
target_model, tokenizer = load_target_model(
    TARGET_MODEL, device=device, dtype=amp_dtype,
)
target_config = target_model.config
print(f"Target: {target_config.num_hidden_layers} layers, "
      f"hidden_size={target_config.hidden_size}, "
      f"vocab_size={target_config.vocab_size}")

# ---------------------------------------------------------------------------
# Create draft model
# ---------------------------------------------------------------------------

draft_config = DFlashConfig.from_target(
    target_config,
    num_draft_layers=NUM_DRAFT_LAYERS,
    num_target_features=NUM_TARGET_FEATURES,
    block_size=BLOCK_SIZE,
)
draft_model = DFlashDraftModel(draft_config).to(device=device, dtype=amp_dtype)

num_params = sum(p.numel() for p in draft_model.parameters() if p.requires_grad)
print(f"Draft model: {NUM_DRAFT_LAYERS} layers, {num_params / 1e6:.1f}M trainable parameters")
print(f"Block size: {BLOCK_SIZE}, Target features from layers: {draft_model.target_layer_ids}")

# EMA
ema_model = None
if USE_EMA:
    from copy import deepcopy
    ema_model = deepcopy(draft_model)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

# ---------------------------------------------------------------------------
# Load training data
# ---------------------------------------------------------------------------

print(f"Loading training data from {TRAIN_DATA_PATH}")
train_data = torch.load(TRAIN_DATA_PATH, weights_only=True)
all_input_ids = train_data["input_ids"]
all_prompt_lens = train_data["prompt_lens"]
print(f"Training sequences: {len(all_input_ids)}")

# Shared embedding and LM head from target (frozen)
embed_fn = target_model.model.embed_tokens
lm_head = target_model.lm_head

# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

if OPTIMIZER == "adamw":
    optimizer = torch.optim.AdamW(draft_model.parameters(), lr=LR, betas=BETAS,
                                  weight_decay=WEIGHT_DECAY)
elif OPTIMIZER == "adam":
    optimizer = torch.optim.Adam(draft_model.parameters(), lr=LR, betas=BETAS)
elif OPTIMIZER == "sgd":
    optimizer = torch.optim.SGD(draft_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
else:
    raise ValueError(f"Unknown optimizer: {OPTIMIZER}")

position_weights = compute_position_weights(BLOCK_SIZE, GAMMA, device)

# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_lr(step: int) -> float:
    if step < WARMUP_STEPS:
        return LR * step / max(WARMUP_STEPS, 1)
    if LR_SCHEDULE == "constant":
        return LR
    elif LR_SCHEDULE == "cosine":
        progress = (step - WARMUP_STEPS) / max(NUM_STEPS - WARMUP_STEPS, 1)
        return LR * 0.5 * (1 + math.cos(math.pi * progress))
    elif LR_SCHEDULE == "linear":
        progress = (step - WARMUP_STEPS) / max(NUM_STEPS - WARMUP_STEPS, 1)
        return LR * (1 - progress)
    return LR


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

print(f"\nStarting training: {NUM_STEPS} steps, batch_size={BATCH_SIZE}")
print(f"Optimizer: {OPTIMIZER}, LR: {LR}, Schedule: {LR_SCHEDULE}")
print(f"Gamma (loss decay): {GAMMA}, Label smoothing: {LABEL_SMOOTHING}")
print("-" * 60)

draft_model.train()
t_start = time.time()
smooth_loss = 0.0
best_loss = float("inf")
losses = []
data_indices = list(range(len(all_input_ids)))
pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

for step in range(NUM_STEPS):
    t0 = time.time()

    # --- Sample batch + per-sample anchor ---
    # Sample anchors up front so we can pad, run one target forward for the
    # whole batch, and one draft forward for the whole batch. Constraint:
    # abs_anchor = prompt_len + anchor_pos >= BLOCK_SIZE, so the ctx slice
    # [abs_anchor - BLOCK_SIZE : abs_anchor] always fits.
    seqs = []  # list of (input_ids, prompt_len, anchor_pos)
    for idx in random.choices(data_indices, k=BATCH_SIZE):
        iids = all_input_ids[idx]
        plen = all_prompt_lens[idx]
        if len(iids) - plen < BLOCK_SIZE + 1:
            continue
        min_anchor_pos = max(0, BLOCK_SIZE - plen)
        max_anchor_pos = len(iids) - plen - BLOCK_SIZE
        if max_anchor_pos < min_anchor_pos:
            continue
        anchor_pos = int(torch.randint(min_anchor_pos, max_anchor_pos + 1, (1,)).item())
        seqs.append((iids, plen, anchor_pos))

    if not seqs:
        continue

    bsz = len(seqs)
    max_len = max(iids.size(0) for iids, _, _ in seqs)

    # --- Pad into a single tensor ---
    padded = torch.full((bsz, max_len), pad_token_id, dtype=torch.long, device=device)
    attn_mask = torch.zeros((bsz, max_len), dtype=torch.long, device=device)
    for i, (iids, _, _) in enumerate(seqs):
        L = iids.size(0)
        padded[i, :L] = iids.to(device)
        attn_mask[i, :L] = 1

    # --- One batched target forward ---
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        target_out = target_model(
            padded,
            attention_mask=attn_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        target_features = extract_context_features(
            target_out.hidden_states, draft_model.target_layer_ids,
        )  # (bsz, max_len, num_features * hidden_size)

    # --- Assemble per-sample blocks, ctx, and positions ---
    ctx_dim = target_features.size(-1)
    block_ids_batch = torch.full((bsz, BLOCK_SIZE), MASK_TOKEN_ID,
                                  dtype=torch.long, device=device)
    labels_batch = torch.zeros(bsz, BLOCK_SIZE - 1, dtype=torch.long, device=device)
    ctx_batch = torch.empty(bsz, BLOCK_SIZE, ctx_dim,
                             dtype=target_features.dtype, device=device)
    pos_batch = torch.empty(bsz, 2 * BLOCK_SIZE, dtype=torch.long, device=device)

    for i, (iids, plen, anchor_pos) in enumerate(seqs):
        abs_anchor = plen + anchor_pos
        iids_dev = iids.to(device)
        block_ids_batch[i, 0] = iids_dev[abs_anchor]
        labels_batch[i] = iids_dev[abs_anchor + 1 : abs_anchor + BLOCK_SIZE]
        ctx_batch[i] = target_features[i, abs_anchor - BLOCK_SIZE : abs_anchor, :]
        pos_batch[i, :BLOCK_SIZE] = torch.arange(
            abs_anchor - BLOCK_SIZE, abs_anchor, device=device,
        )
        pos_batch[i, BLOCK_SIZE:] = torch.arange(
            abs_anchor, abs_anchor + BLOCK_SIZE, device=device,
        )

    # --- One batched draft forward + loss ---
    with torch.no_grad():
        noise_emb = embed_fn(block_ids_batch)

    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        draft_hidden = draft_model(
            noise_embedding=noise_emb,
            target_hidden=ctx_batch,
            position_ids=pos_batch,
        )
        logits = lm_head(draft_hidden[:, 1:, :])  # (bsz, B-1, vocab)

        loss_per_pos = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels_batch.reshape(-1),
            reduction="none",
            label_smoothing=LABEL_SMOOTHING,
        ).view(bsz, BLOCK_SIZE - 1)

        weights = position_weights  # (B-1,)
        per_sample_loss = (loss_per_pos * weights).sum(dim=1) / weights.sum()
        # Divide by nominal BATCH_SIZE (not bsz) so steps where some samples
        # were dropped scale the gradient the same way the old loop did.
        batch_loss = per_sample_loss.sum() / BATCH_SIZE

    # --- Backward + optimize ---
    optimizer.zero_grad(set_to_none=True)
    batch_loss.backward()

    if GRAD_CLIP > 0:
        torch.nn.utils.clip_grad_norm_(draft_model.parameters(), GRAD_CLIP)

    lr = get_lr(step)
    for g in optimizer.param_groups:
        g["lr"] = lr
    optimizer.step()

    # --- EMA update ---
    if ema_model is not None:
        with torch.no_grad():
            for sp, mp in zip(ema_model.parameters(), draft_model.parameters()):
                sp.lerp_(mp, 1 - EMA_DECAY)

    # --- Logging ---
    lv = batch_loss.item()
    smooth_loss = 0.9 * smooth_loss + 0.1 * lv if step > 0 else lv
    losses.append(lv)
    dt = time.time() - t0

    if step % 10 == 0:
        elapsed = time.time() - t_start
        print(f"step {step:05d} | loss: {smooth_loss:.4f} | lr: {lr:.2e} | "
              f"dt: {dt*1000:.0f}ms | elapsed: {elapsed:.0f}s")

    if smooth_loss < best_loss and step > 20:
        best_loss = smooth_loss

    if step == 0:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Save final checkpoint
# ---------------------------------------------------------------------------

t_end = time.time()
training_seconds = t_end - t_start

save_model = ema_model if ema_model is not None else draft_model
torch.save({
    "step": step,
    "model_state": save_model.state_dict(),
    "config": draft_config,
    "loss": smooth_loss,
}, CACHE_DIR / "draft_checkpoint_final.pt")

# ---------------------------------------------------------------------------
# Training summary
# ---------------------------------------------------------------------------

peak_vram = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0
training_tokens = (step + 1) * BATCH_SIZE * BLOCK_SIZE

print()
print("=" * 60)
print("Training complete")
print("=" * 60)
print(f"training_seconds: {training_seconds:.1f}")
print(f"final_loss:       {smooth_loss:.6f}")
print(f"best_loss:        {best_loss:.6f}")
print(f"total_steps:      {step + 1}")
print(f"training_tokens:  {training_tokens}")
print(f"peak_vram_gb:     {peak_vram:.1f}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"draft_layers:     {NUM_DRAFT_LAYERS}")
print(f"block_size:       {BLOCK_SIZE}")
print(f"gamma:            {GAMMA}")
print(f"lr:               {LR}")
print(f"optimizer:        {OPTIMIZER}")
print(f"lr_schedule:      {LR_SCHEDULE}")
print(f"checkpoint:       {CACHE_DIR / 'draft_checkpoint_final.pt'}")
