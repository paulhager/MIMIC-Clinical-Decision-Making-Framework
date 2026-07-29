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
        self.assertEqual(
            completion.call_args.kwargs["messages"][-1],
            {"role": "user", "content": "Case"},
        )

    @patch.object(CustomLLM, "completion_with_backoff")
    def test_empty_hosted_completion_becomes_invalid_action_text(self, completion):
        completion.return_value = {
            "choices": [{"finish_reason": "length", "message": {"content": None}}],
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

        self.assertEqual(result, "")

    @patch.object(CustomLLM, "completion_with_backoff")
    def test_removes_repeated_legacy_assistant_prefill(self, completion):
        completion.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            "Thought: I should examine the abdomen.\n"
                            "Action: Physical Examination\n"
                            "Action Input: Focused abdominal examination."
                        )
                    }
                }
            ],
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
            "<|im_start|>assistantThought:",
            stop=None,
        )

        self.assertTrue(result.startswith("I should examine the abdomen."))
        self.assertNotIn("Thought: Thought:", f"Thought:{result}")
        self.assertEqual(
            completion.call_args.kwargs["messages"][-1],
            {"role": "user", "content": "Case"},
        )

    @patch.object(CustomLLM, "completion_with_backoff")
    def test_applies_model_specific_hosted_sampling(self, completion):
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
            api_sampling={"temperature": 1.0, "top_p": 0.95, "top_k": 64},
        )

        llm._call(
            "<|im_start|>systemTest<|im_end|>"
            "<|im_start|>userCase<|im_end|>"
            "<|im_start|>assistant",
            stop=None,
        )

        request = completion.call_args.kwargs
        self.assertEqual(request["temperature"], 1.0)
        self.assertEqual(request["top_p"], 0.95)
        self.assertEqual(request["top_k"], 64)

    @patch.object(CustomLLM, "completion_with_backoff")
    def test_removes_leading_gemma_channel_token_debris(self, completion):
        completion.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            "an<channt\n<|chant\n<channel|>"
                            "Thought: Imaging is now appropriate.\n"
                            "Action: Imaging\n"
                            "Action Input: CT abdomen and pelvis."
                        )
                    }
                }
            ],
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
            "<|im_start|>assistantThought:",
            stop=None,
        )

        self.assertTrue(result.startswith("Imaging is now appropriate."))
        self.assertNotIn("chan", result[:40].lower())

    @patch.object(CustomLLM, "completion_with_backoff")
    def test_removes_channel_debris_before_full_info_diagnosis(self, completion):
        completion.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "s.<channel|>Acute appendicitis",
                    }
                }
            ],
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
            "<|im_start|>assistantFinal Diagnosis:",
            stop=None,
        )

        self.assertEqual(result, "Acute appendicitis")


if __name__ == "__main__":
    unittest.main()
