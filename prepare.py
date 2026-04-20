"""
One-time data preparation for DFlash training swarm.

Downloads target model, a seed instruction dataset (CodeAlpaca), and for
every prompt rolls out 48 tokens of target greedy continuation. The draft
model is trained to match this rollout — the eval harness compares draft
predictions to target.argmax, so giving the seed target-generated responses
up front (instead of dataset responses) avoids the "score locked at 1.00"
cliff newcomers would otherwise hit.

Prompts are tokenized with `enable_thinking=False`, matching the eval
harness, so training and eval share the same chat-template distribution.

Usage: uv run prepare.py

DO NOT MODIFY — agents modify train.py, not this file.
"""

import os
import json
import torch
from pathlib import Path
from tqdm import tqdm

CACHE_DIR = Path(os.environ.get("DFLASH_CACHE", os.path.expanduser("~/.cache/dflash_swarm")))
TARGET_MODEL = "Qwen/Qwen3-4B"
DATA_DIR = CACHE_DIR / "data"
TRAIN_DATA_PATH = DATA_DIR / "train_sequences.pt"
EVAL_DATA_PATH = DATA_DIR / "eval_prompts.pt"
MAX_TRAIN_SAMPLES = 20_000
MAX_SEQ_LEN = 512
MAX_NEW_TOKENS = 256

# Version marker for the training data format. Bump this whenever the
# prompt formatting or response-generation scheme changes so prepare.py
# knows to regenerate an existing cache.
TRAIN_DATA_VERSION = 2

# Length of the target greedy rollout (in tokens) spliced onto every
# prompt. Needs to be ≥ BLOCK_SIZE in train.py so a full block fits
# starting at anchor_pos=0. 48 leaves headroom for multi-block training.
ROLLOUT_TOKENS = 48

EVAL_PROMPTS = [
    "Write a Python function that computes the nth Fibonacci number using dynamic programming.",
    "Explain the difference between a stack and a queue with examples.",
    "What is the time complexity of binary search? Explain why.",
    "Write a SQL query to find the second highest salary from an employees table.",
    "Explain how gradient descent works in machine learning.",
    "Write a Python class that implements a simple linked list with insert and delete methods.",
    "What is the difference between TCP and UDP? When would you use each?",
    "Explain the CAP theorem in distributed systems.",
    "Write a Python function to check if a string is a valid palindrome, ignoring non-alphanumeric characters.",
    "What are the SOLID principles in software engineering? Give a brief explanation of each.",
    "Implement binary search in Python.",
    "Explain how a hash table works internally.",
    "What is the difference between processes and threads?",
    "Write a Python decorator that caches function results.",
    "Explain the concept of database normalization.",
    "Write a function to find the longest common subsequence of two strings.",
    "What is a closure in programming? Give an example.",
    "Explain the difference between REST and GraphQL.",
    "Write a Python generator that yields prime numbers.",
    "What is the difference between supervised and unsupervised learning?",
    "Implement a simple calculator that handles +, -, *, / with proper operator precedence.",
    "Explain how TLS/SSL works at a high level.",
    "Write a Python function to serialize and deserialize a binary tree.",
    "What are the main differences between Python 2 and Python 3?",
    "Explain the concept of eventual consistency.",
    "Write a function to detect a cycle in a linked list.",
    "What is dependency injection and why is it useful?",
    "Explain the difference between concurrency and parallelism.",
    "Write a Python function to implement merge sort.",
    "What is a B-tree and where is it used?",
    "Solve the following math problem step by step: What is 847 * 293?",
    "A train travels 120 miles in 2 hours. It then travels 180 miles in 3 hours. What is its average speed?",
    "Write a proof that the square root of 2 is irrational.",
    "Solve: Find all integer solutions to x^2 - 5x + 6 = 0.",
    "A rectangular garden has a perimeter of 40 meters. If the length is 3 meters more than the width, find the dimensions.",
    "Calculate the derivative of f(x) = x^3 * sin(x).",
    "How many ways can you arrange the letters in the word MISSISSIPPI?",
    "Write Python code to solve the Tower of Hanoi problem for n disks.",
    "What is the sum of the first 100 positive integers? Show your work.",
    "A bag contains 5 red balls and 3 blue balls. If you draw 2 balls without replacement, what is the probability both are red?",
    "Explain the RSA encryption algorithm step by step.",
    "Write a Python function that implements the Sieve of Eratosthenes.",
    "Describe how a compiler works, from source code to machine code.",
    "Implement a trie data structure in Python.",
    "What is the halting problem and why is it important?",
    "Write a Python function to find all permutations of a string.",
    "Explain the difference between strong and weak typing in programming languages.",
    "How does garbage collection work in Java?",
    "Write a Python async function that fetches data from multiple URLs concurrently.",
    "Explain the MapReduce programming model with an example.",
]


def download_model():
    """Download and verify the target model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_cache = CACHE_DIR / "models"
    model_cache.mkdir(parents=True, exist_ok=True)

    print(f"Downloading target model: {TARGET_MODEL}")
    print("(This may take several minutes on first run...)")

    tokenizer = AutoTokenizer.from_pretrained(
        TARGET_MODEL, cache_dir=str(model_cache), trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL, torch_dtype=torch.bfloat16,
        cache_dir=str(model_cache), trust_remote_code=True,
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Target model loaded: {num_params / 1e9:.1f}B parameters")
    print(f"Architecture: {model.config.num_hidden_layers} layers, "
          f"hidden_size={model.config.hidden_size}")
    return model, tokenizer


def prepare_training_data(model, tokenizer):
    """Tokenize non-thinking prompts from the instruction dataset and splice
    `ROLLOUT_TOKENS` of target greedy continuation onto each. The result is
    `[prompt_tokens, target_rollout]` so the draft's training labels already
    match what the eval harness measures."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if TRAIN_DATA_PATH.exists():
        try:
            data = torch.load(TRAIN_DATA_PATH, weights_only=True)
        except Exception:
            data = {}
        if data.get("version") == TRAIN_DATA_VERSION:
            print(f"Training data already exists: {len(data['input_ids'])} sequences "
                  f"(v{TRAIN_DATA_VERSION})")
            return
        print(f"Training data format changed (found v{data.get('version')}, "
              f"want v{TRAIN_DATA_VERSION}) — regenerating.")
        TRAIN_DATA_PATH.unlink()

    from datasets import load_dataset

    print("Downloading instruction dataset...")
    try:
        dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    except Exception:
        print("CodeAlpaca not available, using Alpaca fallback...")
        dataset = load_dataset("tatsu-lab/alpaca", split="train")

    dataset = dataset.select(range(min(len(dataset), MAX_TRAIN_SAMPLES)))
    print(f"Loaded {len(dataset)} instruction examples")

    device = next(model.parameters()).device
    if device.type == "cpu":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
    model.eval()

    # Step 1: tokenize every prompt with enable_thinking=False.
    print("Tokenizing prompts (non-thinking template)...")
    prompts = []
    for example in tqdm(dataset, desc="Tokenizing"):
        instruction = example.get("instruction", example.get("prompt", ""))
        inp = example.get("input", "")
        content = f"{instruction}\n{inp}" if inp else instruction
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        tokens = tokenizer(formatted, return_tensors="pt",
                           max_length=MAX_SEQ_LEN - ROLLOUT_TOKENS, truncation=True)
        prompts.append(tokens["input_ids"].squeeze(0))

    # Step 2: batch-generate `ROLLOUT_TOKENS` tokens of target greedy continuation.
    # Left-pad so every row shares the same final prompt position → the generation
    # step sees the real last prompt token at the same column for every row.
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    print(f"Rolling out {ROLLOUT_TOKENS} target tokens per prompt (batched)...")
    all_input_ids = []
    all_prompt_lens = []
    gen_bsz = 32
    import time
    t0 = time.time()
    for start in tqdm(range(0, len(prompts), gen_bsz), desc="Rollout"):
        chunk = prompts[start : start + gen_bsz]
        max_plen = max(p.size(0) for p in chunk)
        padded = torch.full((len(chunk), max_plen), pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((len(chunk), max_plen), dtype=torch.long, device=device)
        for j, p in enumerate(chunk):
            L = p.size(0)
            padded[j, max_plen - L :] = p.to(device)
            attn[j, max_plen - L :] = 1
        with torch.no_grad():
            gen = model.generate(
                padded, attention_mask=attn,
                max_new_tokens=ROLLOUT_TOKENS, do_sample=False,
                temperature=None, top_p=None, pad_token_id=pad_id,
            )
        # gen has shape (bsz, max_plen + ROLLOUT_TOKENS). For each row, stitch the
        # original (unpadded) prompt with the last ROLLOUT_TOKENS of gen.
        for j, p in enumerate(chunk):
            seq = torch.cat([p, gen[j, -ROLLOUT_TOKENS:].cpu()])
            all_input_ids.append(seq)
            all_prompt_lens.append(p.size(0))

    print(f"Rollouts done in {time.time() - t0:.0f}s")

    torch.save({
        "version": TRAIN_DATA_VERSION,
        "input_ids": all_input_ids,
        "prompt_lens": all_prompt_lens,
    }, TRAIN_DATA_PATH)
    print(f"Training data saved: {len(all_input_ids)} sequences to {TRAIN_DATA_PATH}")


def prepare_eval_prompts(tokenizer):
    """Tokenize evaluation prompts."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if EVAL_DATA_PATH.exists():
        data = torch.load(EVAL_DATA_PATH, weights_only=True)
        print(f"Eval prompts already exist: {len(data['input_ids'])} prompts")
        return

    print("Preparing evaluation prompts...")
    all_input_ids = []

    # Format with enable_thinking=False — Qwen3-4B defaults to thinking mode
    # (emits <think> as the first response token). Training also uses the
    # non-thinking template, so the anchor distribution matches at eval.
    for prompt_text in EVAL_PROMPTS:
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        tokens = tokenizer(formatted, return_tensors="pt")
        all_input_ids.append(tokens["input_ids"].squeeze(0))

    torch.save({"input_ids": all_input_ids}, EVAL_DATA_PATH)
    print(f"Eval prompts saved: {len(all_input_ids)} prompts to {EVAL_DATA_PATH}")


if __name__ == "__main__":
    print("=" * 60)
    print("DFlash Swarm — Data Preparation")
    print("=" * 60)
    print(f"Cache directory: {CACHE_DIR}")

    model, tokenizer = download_model()
    prepare_training_data(model, tokenizer)
    prepare_eval_prompts(tokenizer)

    print()
    print("Setup complete! Run train.py to start training.")
    print(f"  Target model: {TARGET_MODEL}")
    print(f"  Training data: {TRAIN_DATA_PATH}")
    print(f"  Eval prompts: {EVAL_DATA_PATH}")
