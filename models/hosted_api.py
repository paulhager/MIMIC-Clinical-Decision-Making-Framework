"""Small OpenAI-compatible client shared by the framework and smoke tests."""

import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import openai
import requests


class HostedAPIBudgetExceeded(RuntimeError):
    """Raised before a request when the configured campaign budget is exhausted."""


class HostedChatClient:
    def __init__(
        self,
        *,
        api_key: str = None,
        api_key_env: str = None,
        api_base_url: str = None,
        site_url_env: str = None,
        app_name_env: str = None,
    ) -> None:
        resolved_key = api_key or (os.getenv(api_key_env) if api_key_env else None)
        if not resolved_key:
            raise ValueError(
                f"API key not found in environment variable {api_key_env!r}"
            )

        openai.api_key = resolved_key
        if api_base_url:
            openai.api_base = api_base_url

        headers = {}
        if site_url_env and os.getenv(site_url_env):
            headers["HTTP-Referer"] = os.getenv(site_url_env)
        if app_name_env and os.getenv(app_name_env):
            headers["X-Title"] = os.getenv(app_name_env)
        if headers:
            openai.requestssession = requests.Session()
            openai.requestssession.headers.update(headers)

        usage_log = os.getenv("OPENROUTER_USAGE_LOG")
        self.usage_log = Path(usage_log).expanduser() if usage_log else None
        budget = os.getenv("OPENROUTER_BUDGET_USD")
        reserve = os.getenv("OPENROUTER_BUDGET_RESERVE_USD", "1")
        self.budget_usd = float(budget) if budget else None
        self.budget_reserve_usd = float(reserve)

    def create(self, **kwargs):
        self._check_budget()
        response = openai.ChatCompletion.create(**kwargs)
        self._record_usage(self.response_metadata(response))
        return response

    def _logged_cost(self) -> float:
        if not self.usage_log or not self.usage_log.exists():
            return 0.0

        total = 0.0
        with self.usage_log.open(encoding="utf-8") as log_file:
            fcntl.flock(log_file.fileno(), fcntl.LOCK_SH)
            for line in log_file:
                try:
                    total += float(json.loads(line).get("cost") or 0)
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
            fcntl.flock(log_file.fileno(), fcntl.LOCK_UN)
        return total

    def _check_budget(self) -> None:
        if self.budget_usd is None:
            return
        stop_at = max(0, self.budget_usd - self.budget_reserve_usd)
        spent = self._logged_cost()
        if spent >= stop_at:
            raise HostedAPIBudgetExceeded(
                f"OpenRouter campaign spend ${spent:.4f} reached the "
                f"${stop_at:.2f} stop threshold (${self.budget_usd:.2f} budget "
                f"with ${self.budget_reserve_usd:.2f} reserved)."
            )

    def _record_usage(self, metadata: Dict[str, Any]) -> None:
        if not self.usage_log:
            return
        self.usage_log.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            **metadata,
        }
        with self.usage_log.open("a", encoding="utf-8") as log_file:
            fcntl.flock(log_file.fileno(), fcntl.LOCK_EX)
            log_file.write(json.dumps(entry, separators=(",", ":")) + "\n")
            log_file.flush()
            fcntl.flock(log_file.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def response_metadata(response) -> Dict[str, Any]:
        usage = response.get("usage", {})
        return {
            "model": response.get("model"),
            "provider": response.get("provider"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "cost": usage.get("cost"),
        }
