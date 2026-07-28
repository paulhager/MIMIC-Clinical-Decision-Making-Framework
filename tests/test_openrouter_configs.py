import unittest
from pathlib import Path

import yaml

from scripts.smoke_openrouter import config_names, load_config, smoke_request


class OpenRouterConfigsTest(unittest.TestCase):
    def test_all_configs_are_pinned_and_safe(self):
        expected = {
            "DeepSeekV4Flash_OpenRouter",
            "GPTOSS120B_OpenRouter",
            "Gemma4_26B_OpenRouter",
            "Llama4Maverick_OpenRouter",
            "MistralSmall4_OpenRouter",
            "Qwen3.5_397B_OpenRouter",
        }
        self.assertEqual(set(config_names()), expected)

        model_names = set()
        for name in expected:
            config = load_config(name)
            model_names.add(config["model_name"])
            self.assertEqual(config["api_key_env"], "OPENROUTER_API_KEY")
            self.assertEqual(config["api_base_url"], "https://openrouter.ai/api/v1")
            self.assertFalse(config["api_provider"]["allow_fallbacks"])
            self.assertTrue(config["api_provider"]["require_parameters"])
            self.assertEqual(config["api_provider"]["data_collection"], "deny")
            self.assertEqual(len(config["api_provider"]["order"]), 1)

        self.assertEqual(len(model_names), len(expected))

    def test_smoke_requests_use_configured_routing(self):
        config = load_config("GPTOSS120B_OpenRouter")
        request = smoke_request(config)

        self.assertEqual(request["model"], "openai/gpt-oss-120b")
        self.assertEqual(request["provider"], config["api_provider"])
        self.assertEqual(request["reasoning"], {"effort": "low", "exclude": True})
        self.assertEqual(request["max_tokens"], 64)


if __name__ == "__main__":
    unittest.main()
