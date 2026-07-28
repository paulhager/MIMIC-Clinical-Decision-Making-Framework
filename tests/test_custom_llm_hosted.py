import unittest
from unittest.mock import patch

from models.models import CustomLLM


TAGS = {
    "system_tag_start": "<|im_start|>system",
    "system_tag_end": "<|im_end|>",
    "user_tag_start": "<|im_start|>user",
    "user_tag_end": "<|im_end|>",
    "ai_tag_start": "<|im_start|>assistant",
    "ai_tag_end": "<|im_end|>",
}


class HostedCustomLLMTest(unittest.TestCase):
    @patch.object(CustomLLM, "completion_with_backoff")
    def test_accepts_missing_langchain_stop_list(self, completion):
        completion.return_value = {
            "choices": [{"message": {"content": "Acute appendicitis"}}],
            "usage": {},
        }
        llm = CustomLLM(
            model_name="org/model",
            openai_api_key="not-a-real-key",
            tags=TAGS,
            max_context_length=4096,
            seed=2023,
        )

        result = llm._call(
            "<|im_start|>systemTest<|im_end|>"
            "<|im_start|>userCase<|im_end|>"
            "<|im_start|>assistant",
            stop=None,
        )

        self.assertEqual(result, "Acute appendicitis")
        self.assertIsInstance(completion.call_args.kwargs["stop"], list)


if __name__ == "__main__":
    unittest.main()
