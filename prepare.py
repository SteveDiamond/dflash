"""
One-time data preparation for dflash autoresearch.
Downloads CIFAR-10 and precomputes Inception reference statistics for FID.

Usage:
    uv run prepare.py

Data is stored in ~/.cache/dflash/.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 300         # training time budget in seconds (5 minutes)
IMAGE_SIZE = 32
NUM_CHANNELS = 3
NUM_CLASSES = 10
NUM_EVAL_SAMPLES = 10000  # number of generated images for FID

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "dflash")
DATA_DIR = os.path.join(CACHE_DIR, "data")
STATS_PATH = os.path.join(CACHE_DIR, "inception_stats.npz")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def download_data():
    """Download CIFAR-10 train + test splits."""
    os.makedirs(DATA_DIR, exist_ok=True)
    datasets.CIFAR10(root=DATA_DIR, train=True, download=True)
    datasets.CIFAR10(root=DATA_DIR, train=False, download=True)


def make_dataloader(batch_size, split="train"):
    """Infinite dataloader for CIFAR-10. Images normalized to [-1, 1]."""
    if split == "train":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
    is_train = (split == "train")
    dataset = datasets.CIFAR10(root=DATA_DIR, train=is_train, transform=transform, download=False)
    pin = torch.cuda.is_available()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=2, drop_last=True, pin_memory=pin)
    while True:
        for batch in loader:
            yield batch

# ---------------------------------------------------------------------------
# FID evaluation
# ---------------------------------------------------------------------------

def _get_inception(device):
    """Load InceptionV3 as a 2048-dim feature extractor."""
    model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1,
                                transform_input=True)
    model.fc = nn.Identity()
    model.eval().to(device)
    return model


def _inception_features(images, inception, device, batch_size=256):
    """Compute 2048-dim Inception features.
    images: tensor in [-1, 1], shape (N, 3, 32, 32).
    """
    feats = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size].to(device)
        batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
        batch = (batch + 1) / 2  # [-1, 1] -> [0, 1] (Inception expects [0, 1])
        with torch.no_grad():
            f = inception(batch)
        feats.append(f.cpu().float().numpy())
    return np.concatenate(feats, axis=0)


def _compute_fid(mu1, sigma1, mu2, sigma2):
    """Frechet Inception Distance between two Gaussians."""
    from scipy.linalg import sqrtm
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def precompute_reference_stats(device):
    """Compute and cache Inception statistics for real CIFAR-10 training data."""
    if os.path.exists(STATS_PATH):
        data = np.load(STATS_PATH)
        return data["mu"], data["sigma"]

    print("Computing Inception reference statistics (one-time)...")
    t0 = time.time()
    inception = _get_inception(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    dataset = datasets.CIFAR10(root=DATA_DIR, train=True, transform=transform, download=False)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=2)

    all_feats = []
    for imgs, _ in loader:
        f = _inception_features(imgs, inception, device)
        all_feats.append(f)
    feats = np.concatenate(all_feats, axis=0)

    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)

    os.makedirs(os.path.dirname(STATS_PATH), exist_ok=True)
    np.savez(STATS_PATH, mu=mu, sigma=sigma)
    print(f"Reference stats saved to {STATS_PATH} ({time.time() - t0:.1f}s)")

    del inception
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return mu, sigma


def evaluate_fid(sample_fn, num_samples, device):
    """Evaluate FID score.

    sample_fn(n): callable returning a tensor of n images in [-1, 1], shape (n, 3, 32, 32).
    Returns: FID score (lower is better).
    """
    mu_real, sigma_real = precompute_reference_stats(device)
    inception = _get_inception(device)

    all_feats = []
    remaining = num_samples
    while remaining > 0:
        n = min(remaining, 256)
        samples = sample_fn(n).clamp(-1, 1)
        f = _inception_features(samples, inception, device)
        all_feats.append(f)
        remaining -= n
        print(f"\r  FID eval: {num_samples - remaining}/{num_samples}", end="", flush=True)
    print()

    feats = np.concatenate(all_feats, axis=0)
    mu_gen = np.mean(feats, axis=0)
    sigma_gen = np.cov(feats, rowvar=False)

    fid = _compute_fid(mu_real, sigma_real, mu_gen, sigma_gen)

    del inception
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return fid


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== dflash autoresearch setup ===")
    print("Downloading CIFAR-10...")
    download_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    precompute_reference_stats(device)
    print("Done! Run 'uv run train.py' to start training.")
