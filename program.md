# dflash autoresearch

This is an experiment to have LLM agents autonomously discover better diffusion transformer architectures and training recipes on CIFAR-10.

## Setup

1. **Agree on a run tag**: e.g. `apr19`. Branch: `dflash/<tag>`.
2. **Create the branch**: `git checkout -b dflash/<tag>` from main.
3. **Read the in-scope files**:
   - `README.md` — repo context
   - `prepare.py` — fixed: data, FID evaluation, constants. **Do not modify.**
   - `train.py` — the file you modify. Model architecture, optimizer, diffusion, sampling, training loop.
4. **Run data prep**: `uv run prepare.py` (downloads CIFAR-10 + precomputes Inception stats, one-time).
5. **Initialize results.tsv** with just the header row. The baseline is recorded after the first run.
6. **Confirm and go.**

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation/evaluation). Launch: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — the only file you edit. Everything is fair game:
  - **Architecture**: depth, width, heads, patch size, attention mechanism, MLP design, normalization, positional encoding
  - **Diffusion**: noise schedule (cosine, linear, learned), prediction target (epsilon, v-prediction, x0), loss weighting across timesteps
  - **Sampling**: DDIM steps, CFG scale, sampler design (DDPM, DPM-Solver, etc.)
  - **Optimization**: optimizer choice, learning rate, schedule (warmup, cosine decay, etc.), weight decay, gradient clipping
  - **Training loop**: EMA decay, batch size, data augmentation, gradient accumulation
  - **Anything else** in train.py

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. Contains evaluation, data loading, and constants.
- Install new packages or add dependencies beyond `pyproject.toml`.
- Modify the evaluation harness. `evaluate_fid` in `prepare.py` is the ground truth metric.
- Use multiple GPUs or spawn threads/processes for training.

**The goal: get the lowest val_fid.** Since the time budget is fixed at 5 minutes, everything is fair game. FID (Frechet Inception Distance) measures how close generated images are to real CIFAR-10 images — lower is better.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful FID gains.

**Simplicity criterion**: All else being equal, simpler is better. A tiny FID improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome.

**The first run** should always establish the baseline — run train.py as-is.

## Output format

The script prints a summary:

```
---
val_fid:          125.3
val_loss:         0.042100
training_seconds: 300.1
total_seconds:    340.5
peak_vram_mb:     3200.0
total_images_K:   768.0
num_steps:        6000
num_params_M:     33.1
depth:            12
```

Extract the key metric: `grep "^val_fid:" run.log`

## Logging results

Log to `results.tsv` (tab-separated):

```
commit	val_fid	val_loss	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_fid achieved (e.g. 125.3) — use 0.0 for crashes
3. val_loss (e.g. 0.042100) — use 0.0 for crashes
4. peak memory in GB, round to .1f — use 0.0 for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description

Example:

```
commit	val_fid	val_loss	memory_gb	status	description
a1b2c3d	125.3	0.042100	3.1	keep	baseline
b2c3d4e	98.7	0.038200	3.2	keep	cosine LR decay with 20% warmdown
c3d4e5f	132.1	0.045000	3.1	discard	switch to linear noise schedule
d4e5f6g	0.0	0.0	0.0	crash	double model width (OOM)
```

## The experiment loop

LOOP FOREVER:

1. Look at git state and results so far.
2. Edit `train.py` with an experimental idea.
3. git commit.
4. Run: `uv run train.py > run.log 2>&1`
5. Read results: `grep "^val_fid:\|^val_loss:\|^peak_vram_mb:" run.log`
6. If grep is empty, the run crashed. Run `tail -n 50 run.log` for the traceback.
7. Record in results.tsv.
8. If val_fid improved (lower), keep the commit.
9. If val_fid is equal or worse, `git reset --hard` to the previous best.

**Timeout**: Each experiment should take ~6 minutes total (5 min training + eval). Kill anything over 10 minutes.

**NEVER STOP**: Do not pause to ask the human. They might be asleep. Run indefinitely until manually stopped. If you run out of ideas, think harder — read the DiT paper, try ALNS, try different attention patterns, try learned noise schedules, try v-prediction.

## Ideas to try

Roughly ordered by expected impact:

- **LR schedule**: cosine decay, warmdown phase, higher peak LR
- **Noise schedule**: linear vs cosine, learned, shifted schedules
- **Prediction target**: v-prediction instead of epsilon, x0 prediction
- **Loss weighting**: weight loss by timestep (SNR weighting, min-SNR)
- **Architecture**: RMSNorm, SwiGLU MLP, different head dims, deeper/narrower
- **Attention**: different head counts, QK-norm, relative position bias
- **Optimizer**: try Muon, Lion, different Adam betas, higher weight decay
- **EMA**: different decay rates, EMA warmup
- **Data augmentation**: random crop + resize, color jitter, cutout
- **Sampling**: more DDIM steps, different CFG scales, DPM-Solver
- **Batch size**: larger batches with gradient accumulation
- **torch.compile**: compile the model for potential speedup
- **Mixed precision**: already using bf16 on CUDA, could try fp16
