"""Small OpenAI-compatible client shared by the framework and smoke tests."""

import os
from typing import Any, Dict

import openai
import requests


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

    def create(self, **kwargs):
        return openai.ChatCompletion.create(**kwargs)

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
