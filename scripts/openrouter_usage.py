#!/usr/bin/env python3
"""Summarize the local OpenRouter cost ledger without exposing prompts."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_usage(path: Path) -> tuple[list[dict], int]:
    entries = []
    invalid_lines = 0
    if not path.exists():
        return entries, invalid_lines
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            invalid_lines += 1
    return entries, invalid_lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "ledger",
        nargs="?",
        type=Path,
        default=Path("runtime_data/openrouter_usage.jsonl"),
    )
    args = parser.parse_args()

    entries, invalid_lines = load_usage(args.ledger)
    grouped = defaultdict(
        lambda: {"requests": 0, "prompt": 0, "completion": 0, "cost": 0.0}
    )
    for entry in entries:
        key = (entry.get("model") or "unknown", entry.get("provider") or "unknown")
        group = grouped[key]
        group["requests"] += 1
        group["prompt"] += int(entry.get("prompt_tokens") or 0)
        group["completion"] += int(entry.get("completion_tokens") or 0)
        group["cost"] += float(entry.get("cost") or 0)

    for (model, provider), usage in sorted(grouped.items()):
        print(
            f"{model} via {provider}: {usage['requests']} requests, "
            f"{usage['prompt']} input + {usage['completion']} output tokens, "
            f"${usage['cost']:.4f}"
        )
    print(
        f"TOTAL: {len(entries)} requests, "
        f"${sum(float(entry.get('cost') or 0) for entry in entries):.4f}"
    )
    if invalid_lines:
        print(f"warning: ignored {invalid_lines} malformed ledger lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
