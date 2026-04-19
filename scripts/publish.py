#!/usr/bin/env python3
"""Publish training results to the dflash coordination server.

Usage:
    uv run train.py > run.log 2>&1
    python3 scripts/publish.py AGENT_ID "title" "description" strategy_tag "notes"

Reads train.py from the current directory and parses run.log for metrics.
"""

import json
import re
import sys
import urllib.request
from pathlib import Path

SERVER = "https://dflash.discoveryatscale.com"


def parse_log(log_path="run.log"):
    text = Path(log_path).read_text()
    metrics = {}
    for key in ("val_fid", "val_loss", "num_params_M", "num_steps", "training_seconds"):
        m = re.search(rf"^{key}:\s+(.+)$", text, re.MULTILINE)
        if m:
            metrics[key] = float(m.group(1).strip())
    metrics["feasible"] = "FAIL" not in text and "val_fid" in metrics
    return metrics


def main():
    if len(sys.argv) < 5:
        print(
            "Usage: python3 scripts/publish.py <agent_id> <title> <description> <strategy_tag> [notes]",
            file=sys.stderr,
        )
        sys.exit(1)

    agent_id = sys.argv[1]
    title = sys.argv[2]
    description = sys.argv[3]
    strategy_tag = sys.argv[4]
    notes = sys.argv[5] if len(sys.argv) > 5 else ""

    metrics = parse_log()
    code = Path("train.py").read_text()

    payload = {
        "agent_id": agent_id,
        "title": title,
        "description": description,
        "strategy_tag": strategy_tag,
        "algorithm_code": code,
        "score": metrics.get("val_fid", 999999.0),
        "feasible": metrics.get("feasible", False),
        "val_loss": metrics.get("val_loss", 0.0),
        "num_params": metrics.get("num_params_M", 0.0),
        "notes": notes,
    }

    req = urllib.request.Request(
        f"{SERVER}/api/iterations",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req) as resp:
        result = json.load(resp)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
