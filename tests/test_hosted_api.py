import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models.hosted_api import HostedAPIBudgetExceeded, HostedChatClient


class HostedChatClientTest(unittest.TestCase):
    @patch.dict(
        os.environ,
        {
            "TEST_API_KEY": "not-a-real-key",
            "TEST_SITE_URL": "https://example.test",
            "TEST_APP_NAME": "MIMIC-CDM test",
        },
        clear=False,
    )
    @patch("models.hosted_api.openai.ChatCompletion.create")
    def test_configures_and_calls_openai_compatible_endpoint(self, create):
        create.return_value = {"choices": [{"message": {"content": "ok"}}]}
        client = HostedChatClient(
            api_key_env="TEST_API_KEY",
            api_base_url="https://router.example/v1",
            site_url_env="TEST_SITE_URL",
            app_name_env="TEST_APP_NAME",
        )

        response = client.create(model="org/model", messages=[])

        self.assertEqual(response["choices"][0]["message"]["content"], "ok")
        create.assert_called_once_with(model="org/model", messages=[])

    def test_missing_key_fails_before_request(self):
        with self.assertRaisesRegex(ValueError, "API key not found"):
            HostedChatClient(api_key_env="DEFINITELY_MISSING_MIMIC_TEST_KEY")

    @patch("models.hosted_api.openai.ChatCompletion.create")
    def test_records_cost_metadata_without_request_content(self, create):
        create.return_value = {
            "model": "org/model",
            "provider": "Provider",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": "private response"},
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
                "cost": 0.001,
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            usage_log = Path(temp_dir) / "usage.jsonl"
            with patch.dict(
                os.environ,
                {
                    "TEST_API_KEY": "not-a-real-key",
                    "OPENROUTER_USAGE_LOG": str(usage_log),
                },
                clear=False,
            ):
                client = HostedChatClient(api_key_env="TEST_API_KEY")
                client.create(
                    model="org/model",
                    messages=[{"role": "user", "content": "private prompt"}],
                )

            entry = json.loads(usage_log.read_text())
            self.assertEqual(entry["cost"], 0.001)
            self.assertEqual(entry["finish_reason"], "stop")
            self.assertTrue(entry["content_present"])
            self.assertNotIn("private prompt", usage_log.read_text())
            self.assertNotIn("private response", usage_log.read_text())

    @patch("models.hosted_api.openai.ChatCompletion.create")
    def test_stops_before_request_at_budget_threshold(self, create):
        with tempfile.TemporaryDirectory() as temp_dir:
            usage_log = Path(temp_dir) / "usage.jsonl"
            usage_log.write_text('{"cost":4.0}\n{"cost":5.0}\n')
            with patch.dict(
                os.environ,
                {
                    "TEST_API_KEY": "not-a-real-key",
                    "OPENROUTER_USAGE_LOG": str(usage_log),
                    "OPENROUTER_BUDGET_USD": "10",
                    "OPENROUTER_BUDGET_RESERVE_USD": "1",
                },
                clear=False,
            ):
                client = HostedChatClient(api_key_env="TEST_API_KEY")
                with self.assertRaises(HostedAPIBudgetExceeded):
                    client.create(model="org/model", messages=[])

            create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
