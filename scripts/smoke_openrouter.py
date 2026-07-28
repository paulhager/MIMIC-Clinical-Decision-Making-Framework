#!/usr/bin/env python3
"""Run a cheap, synthetic OpenRouter smoke test without loading MIMIC data."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.hosted_api import HostedChatClient  # noqa: E402


MODEL = "google/gemma-4-26b-a4b-it"
EXPECTED_TERM = "appendicitis"


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print(
            "OPENROUTER_API_KEY is missing. Copy .env.example to .env and add "
            "your key.",
            file=sys.stderr,
        )
        return 2

    client = HostedChatClient(
        api_key=api_key,
        api_base_url="https://openrouter.ai/api/v1",
        site_url_env="OPENROUTER_SITE_URL",
        app_name_env="OPENROUTER_APP_NAME",
    )

    response = client.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are testing a clinical decision support benchmark. "
                    "Return only the most likely diagnosis."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Synthetic case: A 28-year-old has migrating periumbilical "
                    "to right-lower-quadrant pain, fever, nausea, focal guarding, "
                    "leukocytosis, and CT shows an 11 mm fluid-filled appendix "
                    "with periappendiceal fat stranding. What is the diagnosis?"
                ),
            },
        ],
        temperature=0,
        seed=2023,
        max_tokens=64,
        reasoning={"enabled": False},
        provider={
            "order": ["deepinfra"],
            "allow_fallbacks": False,
            "data_collection": "deny",
        },
    )

    answer = response["choices"][0]["message"]["content"].strip()
    metadata = client.response_metadata(response)
    print(f"model: {metadata.get('model', MODEL)}")
    print(f"provider: {metadata.get('provider', 'not reported')}")
    print(
        "usage: "
        f"{metadata.get('prompt_tokens', '?')} prompt + "
        f"{metadata.get('completion_tokens', '?')} completion = "
        f"{metadata.get('total_tokens', '?')} tokens; "
        f"${float(metadata.get('cost') or 0):.8f}"
    )
    print(f"answer: {answer}")

    if EXPECTED_TERM not in answer.lower():
        print(
            f"FAIL: expected the answer to contain {EXPECTED_TERM!r}.",
            file=sys.stderr,
        )
        return 1

    print("PASS: OpenRouter connectivity and synthetic diagnostic output verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
