#!/usr/bin/env python3
"""Download the released z-lab/Qwen3-4B-DFlash-b16 draft model, convert it
into a checkpoint that evaluate.py can load, and print its tier-1 score.

This is the "paper upper bound" on our eval — agents can compare their own
trained model's score to this number.

Usage: python scripts/oracle_score.py [--tier 1|2]
"""

import argparse, os, sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
import safetensors.torch as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model import DFlashConfig, DFlashDraftModel, load_target_model
from prepare import CACHE_DIR, TARGET_MODEL

ORACLE_REPO = "z-lab/Qwen3-4B-DFlash-b16"
# From the released config. build_target_layer_ids gives the same set for
# Qwen3-4B (36 layers, 5 features); we pin the explicit list so the converter
# stays correct even if that heuristic is tweaked later.
ORACLE_TARGET_LAYER_IDS = [1, 9, 17, 25, 33]
ORACLE_MASK_TOKEN_ID = 151669

ORACLE_CKPT = CACHE_DIR / "oracle_checkpoint.pt"


def download_and_convert() -> Path:
    """Fetch the released weights and save a checkpoint compatible with
    evaluate.py / model.DFlashDraftModel."""
    print(f"Downloading {ORACLE_REPO}...")
    snapshot = snapshot_download(
        ORACLE_REPO,
        allow_patterns=["*.json", "*.safetensors"],
    )
    weights_file = Path(snapshot) / "model.safetensors"
    print(f"Weights: {weights_file}")

    print(f"Loading target {TARGET_MODEL} to build matching config...")
    target, _ = load_target_model(TARGET_MODEL, device="cuda", dtype=torch.bfloat16)
    cfg = DFlashConfig.from_target(
        target.config, num_draft_layers=5, num_target_features=5, block_size=16,
    )

    print("Instantiating swarm DFlashDraftModel and loading released weights...")
    model = DFlashDraftModel(cfg).to(device="cuda", dtype=torch.bfloat16)
    model.target_layer_ids = ORACLE_TARGET_LAYER_IDS  # release hard-codes this
    sd = st.load_file(str(weights_file))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"WARNING missing={len(missing)} unexpected={len(unexpected)}")
        if missing: print(f"  missing: {missing[:5]}")
        if unexpected: print(f"  unexpected: {unexpected[:5]}")
    else:
        print("All keys loaded cleanly.")

    ORACLE_CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": -1,
        "model_state": model.state_dict(),
        "config": cfg,
        "loss": 0.0,
        "mask_token_id": ORACLE_MASK_TOKEN_ID,
    }, ORACLE_CKPT)
    print(f"Saved oracle checkpoint: {ORACLE_CKPT}")
    return ORACLE_CKPT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, default=1, choices=[1, 2])
    ap.add_argument("--rebuild", action="store_true",
                    help="Re-download and reconvert even if the checkpoint exists.")
    args = ap.parse_args()

    if args.rebuild or not ORACLE_CKPT.exists():
        download_and_convert()

    print(f"\nRunning evaluate.py --tier {args.tier} on oracle checkpoint...")
    os.execvp(
        sys.executable,
        [sys.executable, str(ROOT / "evaluate.py"),
         "--checkpoint", str(ORACLE_CKPT),
         "--tier", str(args.tier)],
    )


if __name__ == "__main__":
    main()
