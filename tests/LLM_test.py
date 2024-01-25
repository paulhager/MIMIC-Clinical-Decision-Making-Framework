import unittest

from transformers import LlamaTokenizer

from models.utils import create_stop_criteria
from agents.agent import STOP_WORDS


class TestLLM(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        self.tokenizer = LlamaTokenizer.from_pretrained("")
        self.stop_criteria = create_stop_criteria(STOP_WORDS, self.tokenizer, "cpu")

    ###############
    # Observation #
    ###############

    def test_stop_condition_Observation_plain(self):
        generated_text = "This is an interesting sentence. Observation:"
        generated_ids = self.tokenizer.encode(
            generated_text, add_special_tokens=False, return_tensors="pt"
        )

        self.assertTrue(self.stop_criteria(generated_ids, None))

    def test_stop_condition_Observation_plain_fail(self):
        generated_text = (
            "This is an interesting sentence. This should not stop the generation."
        )
        generated_ids = self.tokenizer.encode(
            generated_text, add_special_tokens=False, return_tensors="pt"
        )

        self.assertFalse(self.stop_criteria(generated_ids, None))

    def test_stop_condition_Observation_plain_fail_no_colon(self):
        generated_text = "This is an interesting sentence. Observation"
        generated_ids = self.tokenizer.encode(
            generated_text, add_special_tokens=False, return_tensors="pt"
        )

        self.assertFalse(self.stop_criteria(generated_ids, None))

    def test_stop_condition_Observation_multispace(self):
        generated_text = "This is an interesting sentence.  Observation:"
        generated_ids = self.tokenizer.encode(
            generated_text, add_special_tokens=False, return_tensors="pt"
        )

        self.assertTrue(self.stop_criteria(generated_ids, None))

    def test_stop_condition_Observation_tab_no_space(self):
        generated_text = "This is an interesting sentence.\tObservation:"
        generated_ids = self.tokenizer.encode(
            generated_text, add_special_tokens=False, return_tensors="pt"
        )

        self.assertTrue(self.stop_criteria(generated_ids, None))

    def test_stop_condition_Observation_tab_space_suffix(self):
        generated_text = "This is an interesting sentence.\t Observation:"
        generated_ids = self.tokenizer.encode(
            generated_text, add_special_tokens=False, return_tensors="pt"
        )

        self.assertTrue(self.stop_criteria(generated_ids, None))

    def test_stop_condition_Observation_tab_space_prefix(self):
        generated_text = "This is an interesting sentence. \tObservation:"
        generated_ids = self.tokenizer.encode(
            generated_text, add_special_tokens=False, return_tensors="pt"
        )

        self.assertTrue(self.stop_criteria(generated_ids, None))

    def test_stop_condition_Observation_plain_no_space(self):
        generated_text = "This is an interesting sentence.Observation:"
        generated_ids = self.tokenizer.encode(
            generated_text, add_special_tokens=False, return_tensors="pt"
        )

        self.assertTrue(self.stop_criteria(generated_ids, None))

    def test_stop_condition_Observation_plain_newline_no_space(self):
        generated_text = "This is an interesting sentence.\nObservation:"
        generated_ids = self.tokenizer.encode(
            generated_text, add_special_tokens=False, return_tensors="pt"
        )

        self.assertTrue(self.stop_criteria(generated_ids, None))

    def test_stop_condition_Observation_plain_newline_space_prefix(self):
        generated_text = "This is an interesting sentence. \nObservation:"
        generated_ids = self.tokenizer.encode(
            generated_text, add_special_tokens=False, return_tensors="pt"
        )

        self.assertTrue(self.stop_criteria(generated_ids, None))

    def test_stop_condition_Observation_plain_newline_space_suffix(self):
        generated_text = "This is an interesting sentence.\n Observation:"
        generated_ids = self.tokenizer.encode(
            generated_text, add_special_tokens=False, return_tensors="pt"
        )

        self.assertTrue(self.stop_criteria(generated_ids, None))

    def test_stop_condition_observations(self):
        generated_text = (
            "Action: Referral to GI specialist\nAction Input: None\nObservations:"
        )

        generated_ids = self.tokenizer.encode(
            generated_text, add_special_tokens=False, return_tensors="pt"
        )

        self.assertTrue(self.stop_criteria(generated_ids, None))

    def test_stop_condition_observation_lower_case(self):
        generated_text = (
            "Action: Referral to GI specialist\nAction Input: None\nobservation:"
        )

        generated_ids = self.tokenizer.encode(
            generated_text, add_special_tokens=False, return_tensors="pt"
        )

        self.assertTrue(self.stop_criteria(generated_ids, None))

    def test_stop_condition_observations_lower_case(self):
        generated_text = (
            "Action: Referral to GI specialist\nAction Input: None\nobservations:"
        )

        generated_ids = self.tokenizer.encode(
            generated_text, add_special_tokens=False, return_tensors="pt"
        )

        self.assertTrue(self.stop_criteria(generated_ids, None))


if __name__ == "__main__":
    unittest.main()
