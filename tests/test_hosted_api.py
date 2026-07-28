import os
import unittest
from unittest.mock import patch

from models.hosted_api import HostedChatClient


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


if __name__ == "__main__":
    unittest.main()
