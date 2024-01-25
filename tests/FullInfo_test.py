import unittest
from typing import Any
from hydra import initialize, compose
from run_full_info import control_context_length
from langchain.llms.fake import FakeListLLM
from langchain.llms.base import LLM
from transformers import LlamaTokenizer
from agents.prompts import (
    FULL_INFO_TEMPLATE,
    FI_FEWSHOT_TEMPLATE_COPD_RR,
    FI_FEWSHOT_TEMPLATE_PNEUMONIA_RR,
)
from exllamav2 import ExLlamaV2Config, ExLlamaV2Tokenizer
from utils.nlp import truncate_text


class FakeLLM(LLM):
    tokenizer: Any
    model: Any

    @property
    def _llm_type(self) -> Any:
        return "custom"

    def load_model(self, responses, tokenizer):
        self.model = FakeListLLM(responses=responses)
        self.tokenizer = tokenizer

    def _call(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class TestLLM(unittest.TestCase):
    def setUp(self):
        with initialize(config_path="../configs", version_base=None):
            self.args = compose(
                config_name="config", overrides=["paths=tower", "fewshot=True"]
            )
        self.maxDiff = None
        self.prompt_template = FULL_INFO_TEMPLATE
        self._id = 1
        self.tags = {
            "system_tag_start": self.args.system_tag_start,
            "user_tag_start": self.args.user_tag_start,
            "ai_tag_start": self.args.ai_tag_start,
            "system_tag_end": self.args.system_tag_end,
            "user_tag_end": self.args.user_tag_end,
            "ai_tag_end": self.args.ai_tag_end,
        }
        self.hadm_info_clean = {
            1: {
                "Radiology": [
                    {
                        "Region": "Abdomen",
                        "Modality": "CT",
                        "Report": "This is a very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very long radiology report",
                        # Num_tokens = 151
                    }
                ]
            },
        }
        # Num_tokens = 98
        self.summarized_report = "This is a very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very long radiology report"
        # Num_tokens = 4
        self.summarized_rad_report_summary = "Radiology report"
        self.fewshot_examples = ""
        # Num_tokens = 563
        self.fewshot_examples += FI_FEWSHOT_TEMPLATE_COPD_RR.format(
            user_tag_start=self.tags["user_tag_start"],
            user_tag_end=self.tags["user_tag_end"],
            ai_tag_start=self.tags["ai_tag_start"],
            ai_tag_end=self.tags["ai_tag_end"],
        )
        # Num_tokens = 569
        self.fewshot_examples += FI_FEWSHOT_TEMPLATE_PNEUMONIA_RR.format(
            user_tag_start=self.tags["user_tag_start"],
            user_tag_end=self.tags["user_tag_end"],
            ai_tag_start=self.tags["ai_tag_start"],
            ai_tag_end=self.tags["ai_tag_end"],
        )
        # 90 Tokens
        self.input = "This is a very interesting case test. This is a very interesting case test. This is a very interesting case test. This is a very interesting case test. This is a very interesting case test. This is a very interesting case test. This is a very interesting case test. This is a very interesting case test. This is a very interesting case test. This is a very interesting case test. This is a very interesting case test.\n{rad_reports}"

    def test_control_context_length(self):
        tokenizer = LlamaTokenizer.from_pretrained("models/WizardLM-70B-V1.0-GPTQ")
        llm = FakeLLM()
        llm.load_model(responses=[""], tokenizer=tokenizer)

        fewshot_examples = ""
        rad_reports = self.hadm_info_clean[self._id]["Radiology"][0]["Report"]
        self.args.fewshot = False

        (
            input_c,
            fewshot_examples_c,
            rad_reports_c,
        ) = control_context_length(
            input=self.input,
            prompt_template=self.prompt_template,
            fewshot_examples=fewshot_examples,
            include_ref_range=True,
            rad_reports=rad_reports,
            llm=llm,
            args=self.args,
            tags=self.tags,
            _id=self._id,
            hadm_info_clean=self.hadm_info_clean,
            diagnostic_criteria="",
            summarize=True,
        )

        self.assertEqual(self.input, input_c)
        self.assertEqual(fewshot_examples, fewshot_examples_c)
        self.assertEqual(rad_reports, rad_reports_c)

    def test_control_context_length_two_fewshot_too_long_one_ok(self):
        # 563 (FS 1) + 569 (FS 2) + 151 (Rad Report) + 90 (Input) + 147 (FI Template) = 1519
        # 563 (FS 1) + 151 (Rad Report) + 90 (Input) + 147 (FI Template) = 951
        tokenizer = LlamaTokenizer.from_pretrained("models/WizardLM-70B-V1.0-GPTQ")
        llm = FakeLLM()
        llm.load_model(responses=[""], tokenizer=tokenizer)

        fewshot_examples = self.fewshot_examples
        rad_reports = self.hadm_info_clean[self._id]["Radiology"][0]["Report"]

        self.args.max_context_length = 1400
        (
            input_c,
            fewshot_examples_c,
            rad_reports_c,
        ) = control_context_length(
            input=self.input,
            prompt_template=self.prompt_template,
            fewshot_examples=fewshot_examples,
            include_ref_range=True,
            rad_reports=rad_reports,
            llm=llm,
            args=self.args,
            tags=self.tags,
            _id=self._id,
            hadm_info_clean=self.hadm_info_clean,
            diagnostic_criteria="",
            summarize=True,
        )

        fewshot_copd = FI_FEWSHOT_TEMPLATE_COPD_RR.format(
            user_tag_start=self.tags["user_tag_start"],
            user_tag_end=self.tags["user_tag_end"],
            ai_tag_start=self.tags["ai_tag_start"],
            ai_tag_end=self.tags["ai_tag_end"],
        )

        self.assertEqual(self.input, input_c)
        self.assertEqual(fewshot_copd, fewshot_examples_c)
        self.assertEqual(rad_reports, rad_reports_c)

    def test_control_context_length_two_fewshot_too_long_zero_ok(self):
        # 563 (FS 1) + 569 (FS 2) + 151 (Rad Report) + 90 (Input) + 147 (FI Template) = 1519
        # 563 (FS 1) + 151 (Rad Report) + 90 (Input) + 147 (FI Template) = 951
        # 151 (Rad Report) + 90 (Input) + 147 (FI Template) = 390
        tokenizer = LlamaTokenizer.from_pretrained("models/WizardLM-70B-V1.0-GPTQ")
        llm = FakeLLM()
        llm.load_model(responses=[""], tokenizer=tokenizer)

        fewshot_examples = self.fewshot_examples
        rad_reports = self.hadm_info_clean[self._id]["Radiology"][0]["Report"]

        self.args.max_context_length = 900
        (
            input_c,
            fewshot_examples_c,
            rad_reports_c,
        ) = control_context_length(
            input=self.input,
            prompt_template=self.prompt_template,
            fewshot_examples=fewshot_examples,
            include_ref_range=True,
            rad_reports=rad_reports,
            llm=llm,
            args=self.args,
            tags=self.tags,
            _id=self._id,
            hadm_info_clean=self.hadm_info_clean,
            diagnostic_criteria="",
            summarize=True,
        )

        self.assertEqual(self.input, input_c)
        self.assertEqual("", fewshot_examples_c)
        self.assertEqual(rad_reports, rad_reports_c)

    def test_control_context_length_summarize_rad(self):
        # 563 (FS 1) + 569 (FS 2) + 151 (Rad Report) + 90 (Input) + 147 (FI Template) = 1519
        # 563 (FS 1) + 151 (Rad Report) + 90 (Input) + 147 (FI Template) = 951
        # 151 (Rad Report) + 90 (Input) + 147 (FI Template) = 390
        # 98 (Rad Report Summarized) + 90 (Input) + 147 (FI Template) = 335
        tokenizer = LlamaTokenizer.from_pretrained("models/WizardLM-70B-V1.0-GPTQ")
        llm = FakeLLM()

        llm.load_model(
            responses=[self.summarized_report],
            tokenizer=tokenizer,
        )

        fewshot_examples = self.fewshot_examples
        rad_reports = self.hadm_info_clean[self._id]["Radiology"][0]["Report"]

        self.args.max_context_length = 300
        (
            input_c,
            fewshot_examples_c,
            rad_reports_c,
        ) = control_context_length(
            input=self.input,
            prompt_template=self.prompt_template,
            fewshot_examples=fewshot_examples,
            include_ref_range=True,
            rad_reports=rad_reports,
            llm=llm,
            args=self.args,
            tags=self.tags,
            _id=self._id,
            hadm_info_clean=self.hadm_info_clean,
            diagnostic_criteria="",
            summarize=True,
        )

        self.assertEqual(self.input, input_c)
        self.assertEqual("", fewshot_examples_c)
        self.assertEqual("\n " + self.summarized_report, rad_reports_c)

    def test_control_context_length_summarize_rad_summary(self):
        # 563 (FS 1) + 569 (FS 2) + 151 (Rad Report) + 90 (Input) + 147 (FI Template) = 1519
        # 563 (FS 1) + 151 (Rad Report) + 90 (Input) + 147 (FI Template) = 951
        # 151 (Rad Report) + 90 (Input) + 147 (FI Template) = 390
        # 98 (Rad Report Summarized) + 90 (Input) + 147 (FI Template) = 335
        # 4 (Rad Report Summarized Summarized) + 90 (Input) + 147 (FI Template) = 241
        # 90 (Input) + 135 (FI Template) + 1 (Empty FS) = 226
        tokenizer = LlamaTokenizer.from_pretrained("models/WizardLM-70B-V1.0-GPTQ")
        llm = FakeLLM()
        llm.load_model(
            responses=[self.summarized_report, self.summarized_rad_report_summary],
            tokenizer=tokenizer,
        )

        fewshot_examples = self.fewshot_examples
        rad_reports = self.hadm_info_clean[self._id]["Radiology"][0]["Report"]

        self.args.max_context_length = 250
        (
            input_c,
            fewshot_examples_c,
            rad_reports_c,
        ) = control_context_length(
            input=self.input,
            prompt_template=self.prompt_template,
            fewshot_examples=fewshot_examples,
            include_ref_range=True,
            rad_reports=rad_reports,
            llm=llm,
            args=self.args,
            tags=self.tags,
            _id=self._id,
            hadm_info_clean=self.hadm_info_clean,
            diagnostic_criteria="",
            summarize=True,
        )

        truncated_report = truncate_text(
            tokenizer, self.summarized_rad_report_summary, 291 - 223 - 50
        )

        self.assertEqual(self.input, input_c)
        self.assertEqual("", fewshot_examples_c)
        self.assertEqual(truncated_report, rad_reports_c)

    def test_control_context_length_no_rad_truncate_over(self):
        # 563 (FS 1) + 569 (FS 2) + 151 (Rad Report) + 90 (Input) + 147 (FI Template) = 1519
        # 563 (FS 1) + 151 (Rad Report) + 90 (Input) + 147 (FI Template) = 951
        # 151 (Rad Report) + 90 (Input) + 147 (FI Template) = 390
        # 98 (Rad Report Summarized) + 90 (Input) + 147 (FI Template) = 335
        # 4 (Rad Report Summarized Summarized) + 90 (Input) + 147 (FI Template) = 241
        # 90 (Input) + 134 (FI Template) = 239
        tokenizer = LlamaTokenizer.from_pretrained("models/WizardLM-70B-V1.0-GPTQ")
        llm = FakeLLM()
        llm.load_model(
            responses=[self.summarized_report, self.summarized_rad_report_summary],
            tokenizer=tokenizer,
        )

        fewshot_examples = self.fewshot_examples
        rad_reports = self.hadm_info_clean[self._id]["Radiology"][0]["Report"]

        self.args.max_context_length = 200
        (
            input_c,
            fewshot_examples_c,
            rad_reports_c,
        ) = control_context_length(
            input=self.input,
            prompt_template=self.prompt_template,
            fewshot_examples=fewshot_examples,
            include_ref_range=True,
            rad_reports=rad_reports,
            llm=llm,
            args=self.args,
            tags=self.tags,
            _id=self._id,
            hadm_info_clean=self.hadm_info_clean,
            diagnostic_criteria="",
            summarize=True,
        )

        input_output_tokens = tokenizer.encode(self.input.format(rad_reports=""))[
            : 10 - 25
        ]  # max_context_length - FI template - Answer Buffer
        expected_output = tokenizer.decode(
            input_output_tokens, skip_special_tokens=True
        )

        self.assertEqual(expected_output, input_c)
        self.assertEqual("", fewshot_examples_c)
        self.assertEqual("", rad_reports_c)

    def test_control_context_length_no_rad_truncate_over_exllama(self):
        # 563 (FS 1) + 569 (FS 2) + 151 (Rad Report) + 90 (Input) + 147 (FI Template) = 1519
        # 563 (FS 1) + 151 (Rad Report) + 90 (Input) + 147 (FI Template) = 951
        # 151 (Rad Report) + 90 (Input) + 147 (FI Template) = 390
        # 98 (Rad Report Summarized) + 90 (Input) + 147 (FI Template) = 335
        # 4 (Rad Report Summarized Summarized) + 90 (Input) + 147 (FI Template) = 241
        # 95 (Input) + 134 (FI Template) = 229
        config = ExLlamaV2Config()
        config.model_dir = "models/WizardLM-70B-V1.0-GPTQ"
        config.prepare()
        tokenizer = ExLlamaV2Tokenizer(config)
        llm = FakeLLM()
        llm.load_model(
            responses=[self.summarized_report, self.summarized_rad_report_summary],
            tokenizer=tokenizer,
        )

        fewshot_examples = self.fewshot_examples
        rad_reports = self.hadm_info_clean[self._id]["Radiology"][0]["Report"]

        self.args.max_context_length = 200
        self.args.exllama = True
        (
            input_c,
            fewshot_examples_c,
            rad_reports_c,
        ) = control_context_length(
            input=self.input,
            prompt_template=self.prompt_template,
            fewshot_examples=fewshot_examples,
            include_ref_range=True,
            rad_reports=rad_reports,
            llm=llm,
            args=self.args,
            tags=self.tags,
            _id=self._id,
            hadm_info_clean=self.hadm_info_clean,
            diagnostic_criteria="",
            summarize=True,
        )

        input_output_tokens = tokenizer.encode(self.input.format(rad_reports=""))[
            :, : 11 - 25
        ]  # max_context_length - FI template - Answer Buffer
        input_output = tokenizer.decode(input_output_tokens)[0]

        self.assertIsInstance(input_c, str)
        self.assertEqual(input_output, input_c)
        self.assertEqual("", fewshot_examples_c)
        self.assertEqual("", rad_reports_c)


if __name__ == "__main__":
    unittest.main()
