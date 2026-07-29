#!/usr/bin/env python3
"""Probe hosted models with a synthetic second agent turn.

This is deliberately separate from the benchmark smoke test: it exercises the
multi-turn shape that can trigger provider chat-template bugs while ensuring no
MIMIC-derived text is sent or printed.
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.hosted_api import HostedChatClient  # noqa: E402
from scripts.smoke_openrouter import load_config  # noqa: E402


SYSTEM_PROMPT = """You are a medical decision support benchmark agent.
Respond with exactly one of these formats:

Thought: concise reasoning
Action: Physical Examination, Laboratory Tests, or Imaging
Action Input: concise request

OR

Thought: concise reasoning
Final Diagnosis: diagnosis
Treatment: treatment
"""

MESSAGES = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {
        "role": "user",
        "content": (
            "Synthetic case: A 28-year-old has migrating periumbilical to "
            "right-lower-quadrant pain, fever, nausea, focal guarding, and "
            "leukocytosis. Determine the diagnosis using the available actions."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Thought: Thought: I should examine the abdomen before concluding.\n"
            "Action: Physical Examination\n"
            "Action Input: Focused abdominal examination."
        ),
    },
    {
        "role": "user",
        "content": (
            "Observation: Right-lower-quadrant guarding, rebound tenderness, "
            "and a positive Rovsing sign."
        ),
    },
]


def variants(config: dict) -> list[tuple[str, dict]]:
    provider = config["api_provider"]
    return [
        (
            "current-greedy-thinking-off",
            {
                "temperature": 0,
                "reasoning": config.get("api_reasoning"),
            },
        ),
        (
            "google-sampling-thinking-off",
            {
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 64,
                "reasoning": {"effort": "none", "exclude": True},
            },
        ),
        (
            "google-sampling-thinking-low",
            {
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 64,
                "reasoning": {
                    "enabled": True,
                    "effort": "low",
                    "exclude": True,
                },
            },
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="Gemma4_26B_OpenRouter")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    config = load_config(args.config)
    client = HostedChatClient(
        api_key=os.getenv(config["api_key_env"]),
        api_base_url=config["api_base_url"],
        site_url_env=config.get("api_site_url_env"),
        app_name_env=config.get("api_app_name_env"),
    )

    failures = 0
    for name, options in variants(config):
        request = {
            "model": config["model_name"],
            "messages": MESSAGES,
            "provider": config["api_provider"],
            "seed": 2023,
            "max_tokens": 512,
            **{key: value for key, value in options.items() if value is not None},
        }
        response = client.create(**request)
        metadata = client.response_metadata(response)
        choice = response["choices"][0]
        content = choice["message"].get("content") or ""
        marker_count = content.count("<|channel>") + content.count("<channel|>")
        valid_format = (
            ("Action:" in content and "Action Input:" in content)
            or ("Final Diagnosis:" in content and "Treatment:" in content)
        )
        passed = valid_format and marker_count <= 2
        failures += int(not passed)
        print(
            f"{name}: {'PASS' if passed else 'FAIL'}; "
            f"provider={metadata.get('provider')}; "
            f"finish={metadata.get('finish_reason')}; "
            f"completion_tokens={metadata.get('completion_tokens')}; "
            f"content_chars={len(content)}; "
            f"channel_markers={marker_count}; "
            f"cost=${float(metadata.get('cost') or 0):.8f}"
        )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
