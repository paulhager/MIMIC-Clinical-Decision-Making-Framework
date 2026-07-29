#!/usr/bin/env python3
"""Run cheap synthetic smoke tests for OpenRouter model configurations."""

import argparse
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG_DIR = REPO_ROOT / "configs" / "model"
sys.path.insert(0, str(REPO_ROOT))

from models.hosted_api import HostedChatClient  # noqa: E402


EXPECTED_TERM = "appendicitis"
DEFAULT_CONFIG = "Gemma4_26B_OpenRouter"


def config_names() -> list[str]:
    return sorted(path.stem for path in MODEL_CONFIG_DIR.glob("*_OpenRouter.yaml"))


def load_config(name: str) -> dict:
    config_name = Path(name).name
    if config_name.endswith(".yaml"):
        config_name = config_name[: -len(".yaml")]
    path = MODEL_CONFIG_DIR / f"{config_name}.yaml"
    if not path.is_file():
        available = ", ".join(config_names())
        raise ValueError(f"Unknown config {config_name!r}. Available: {available}")
    with path.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def smoke_request(config: dict) -> dict:
    request = {
        "model": config["model_name"],
        "messages": [
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
        "temperature": 0,
        "seed": 2023,
        "max_tokens": (
            512
            if (config.get("api_reasoning") or {}).get("enabled")
            else 64
        ),
    }
    if config.get("api_reasoning"):
        request["reasoning"] = config["api_reasoning"]
    if config.get("api_provider"):
        request["provider"] = config["api_provider"]
    if config.get("api_sampling"):
        request.update(config["api_sampling"])
    return request


def run_config(client: HostedChatClient, config_name: str) -> tuple[bool, float]:
    config = load_config(config_name)
    response = client.create(**smoke_request(config))
    answer = (response["choices"][0]["message"].get("content") or "").strip()
    metadata = client.response_metadata(response)
    cost = float(metadata.get("cost") or 0)

    print(f"\nconfig: {config_name}")
    print(f"model: {metadata.get('model') or config['model_name']}")
    print(f"provider: {metadata.get('provider') or 'not reported'}")
    print(
        "usage: "
        f"{metadata.get('prompt_tokens', '?')} prompt + "
        f"{metadata.get('completion_tokens', '?')} completion = "
        f"{metadata.get('total_tokens', '?')} tokens; "
        f"${cost:.8f}"
    )
    print(f"answer: {answer or '[no final content]'}")

    passed = EXPECTED_TERM in answer.lower()
    print(
        "PASS: synthetic diagnostic output verified."
        if passed
        else f"FAIL: expected the answer to contain {EXPECTED_TERM!r}."
    )
    return passed, cost


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test pinned OpenRouter model configurations."
    )
    parser.add_argument(
        "configs",
        nargs="*",
        help="Hydra model config stems (default: Gemma 4 only).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Test every configs/model/*_OpenRouter.yaml file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = config_names() if args.all else (args.configs or [DEFAULT_CONFIG])

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

    failures = 0
    total_cost = 0.0
    for config_name in selected:
        try:
            passed, cost = run_config(client, config_name)
            total_cost += cost
            failures += int(not passed)
        except Exception as error:
            failures += 1
            print(f"\nconfig: {config_name}")
            print(f"FAIL: {type(error).__name__}: {error}", file=sys.stderr)

    print(
        f"\nsummary: {len(selected) - failures}/{len(selected)} passed; "
        f"reported cost ${total_cost:.8f}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
