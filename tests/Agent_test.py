import unittest
import pickle
from unittest.mock import patch
from typing import Any

from agents.agent import TextSummaryCache, CustomZeroShotAgent
from langchain.schema import AgentAction
from langchain.chains import LLMChain
from langchain.llms.fake import FakeListLLM
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from transformers import LlamaTokenizer

from tools.utils import action_input_pretty_printer
from agents.DiagnosisWorkflowParser import DiagnosisWorkflowParser
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


class TestAgent(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        lab_test_mapping_path = ""
        with open(lab_test_mapping_path, "rb") as f:
            self.lab_test_mapping_df = pickle.load(f)
        self.tags = {
            "user_tag_start": " USER: ",
            "ai_tag_start": " ASSISTANT: ",
            "user_tag_end": "</s>",
            "ai_tag_end": "</s>",
            "system_tag_start": "",
            "system_tag_end": "",
        }

    def test_action_input_pretty_printer_labs(self):
        obj = [50867, 50956]
        labels = ["Amylase", "Lipase"]
        output = action_input_pretty_printer(obj, self.lab_test_mapping_df)

        expected_output = ", ".join(labels)

        self.assertEqual(output, expected_output)

    def test_action_input_pretty_printer_imaging(self):
        obj = {
            "Modality": "CT",
            "Region": "Abdomen",
        }
        output = action_input_pretty_printer(obj, self.lab_test_mapping_df)

        self.assertEqual(output, "CT, Abdomen")

    def test_text_summary_cache(self):
        cache = TextSummaryCache()
        text = "This is a test"
        summary = "This is a test summary"
        cache.add_summary(text, summary)

        self.assertEqual(cache.get_summary(text), summary)
        self.assertEqual(cache.get_summary("This is not a test"), None)

    @patch("agents.agent.action_input_pretty_printer")
    def test_summarize_steps(self, mock_pretty_print):
        intermediate_steps = [
            (
                AgentAction(
                    tool="Physical Examination",
                    tool_input={"action_input": None},
                    log="",
                ),
                "Physical Examination\nThe physical examination was a very good physical examination that showed a lot of important symptoms.",
            ),
            (
                AgentAction(
                    tool="Laboratory Tests",
                    tool_input={"action_input": {"Amylase", "Lipase"}},
                    log="",
                ),
                "Laboratory Results\nAmylase: 32.0 mg/l\nLipase: N/A.",
            ),
            (
                AgentAction(
                    tool="Provide a diagnosis and treatment OR a valid tool. That",
                    tool_input={"action_input": None},
                    log="I would like to use the magic solver tool.",
                ),
                "Provide a diagnosis and treatment OR a valid tool. That is not a valid tool. Try one of the following: Physical Examination, Laboratory Tests, Imaging.",
            ),
            (
                AgentAction(
                    tool="Imaging",
                    tool_input={
                        "action_input": {"Modality": "CT", "Region": "Abdomen"}
                    },
                    log="",
                ),
                "Imaging\nCT ABDOMEN\nThe CT of the abdomen showed us what was wrong.",
            ),
        ]

        def side_effect_function(*args, **kwargs):
            side_effect_function.counter += 1
            if side_effect_function.counter == 1:
                return "Amylase, Lipase"
            return action_input_pretty_printer(*args, **kwargs)

        side_effect_function.counter = 0

        mock_pretty_print.side_effect = side_effect_function

        responses = ["Good PE.", "Good Labs.", "Good Imaging."]
        tokenizer = LlamaTokenizer.from_pretrained("models/WizardLM-70B-V1.0-GPTQ")
        fake_llm = FakeLLM()
        fake_llm.load_model(responses=responses, tokenizer=tokenizer)
        agent = CustomZeroShotAgent(
            llm_chain=LLMChain(
                llm=fake_llm,
                prompt=PromptTemplate(
                    template="{agent_scratchpad}", input_variables=["agent_scratchpad"]
                ),
                callbacks=[],
            ),
            output_parser=DiagnosisWorkflowParser(
                lab_test_mapping_df=self.lab_test_mapping_df
            ),
            stop=[],
            allowed_tools=["Physical Examination", "Laboratory Tests", "Imaging"],
            max_context_length=4000,
            tags=self.tags,
            lab_test_mapping_df=self.lab_test_mapping_df,
            summarize=True,
        )

        summary = agent._summarize_steps(intermediate_steps)

        expected_output = """A summary of information I know thus far:
Action: Physical Examination
Observation: Good PE.
Action: Laboratory Tests
Action Input: Amylase, Lipase
Observation: Good Labs.
I tried 'I would like to use the magic solver tool.', but it was an invalid request.
Action: Imaging
Action Input: CT, Abdomen
Observation: Good Imaging.
Thought:"""

        self.assertEqual(summary, expected_output)

    @unittest.skip("later")
    def test_create_scratchpad(self):
        intermediate_steps = [
            (
                AgentAction(
                    tool="Physical Examination",
                    tool_input={"action_input": None},
                    log="",
                ),
                "Physical Examination\nThe physical examination was a very good physical examination that showed a lot of important symptoms. It was very long examination that went on for a very long time. A very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very long time.",
                # 117 tokens
            ),
        ]

        tokenizer = LlamaTokenizer.from_pretrained("models/WizardLM-70B-V1.0-GPTQ")
        fake_llm = FakeLLM()
        fake_llm.load_model(responses=[""], tokenizer=tokenizer)
        agent = CustomZeroShotAgent(
            llm_chain=LLMChain(
                llm=fake_llm,
                prompt=PromptTemplate(
                    template="{input}{agent_scratchpad}",
                    input_variables=["input", "agent_scratchpad"],
                ),
                callbacks=[],
            ),
            output_parser=DiagnosisWorkflowParser(
                lab_test_mapping_df=self.lab_test_mapping_df
            ),
            stop=[],
            allowed_tools=["Physical Examination", "Laboratory Tests", "Imaging"],
            # 117 (PE) + 100 (buffer) = 217 tokens
            max_context_length=250,
            tags=self.tags,
            lab_test_mapping_df=self.lab_test_mapping_df,
        )

        expected_output = ""
        for action, observation in intermediate_steps:
            expected_output += action.log
            expected_output += (
                f"{agent.observation_prefix}{observation}\n{agent.llm_prefix}"
            )

        kwargs = {"input": ""}
        summary = agent._construct_scratchpad(intermediate_steps, **kwargs)

        self.assertEqual(summary, expected_output)

    @unittest.skip("later")
    def test_create_scratchpad_summarize(self):
        intermediate_steps = [
            (
                AgentAction(
                    tool="Physical Examination",
                    tool_input={"action_input": None},
                    log="",
                ),
                "Physical Examination\nThe physical examination was a very good physical examination that showed a lot of important symptoms. It was very long examination that went on for a very long time. A very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very long time.",
                # 117 tokens
            ),
        ]

        responses = ["Good PE."]
        # 30 tokens
        expected_output = """A summary of information I know thus far:
Action: Physical Examination
Observation: Good PE.
Thought:"""
        tokenizer = LlamaTokenizer.from_pretrained("models/WizardLM-70B-V1.0-GPTQ")
        fake_llm = FakeLLM()
        fake_llm.load_model(responses=responses, tokenizer=tokenizer)
        agent = CustomZeroShotAgent(
            llm_chain=LLMChain(
                llm=fake_llm,
                prompt=PromptTemplate(
                    template="{input}{agent_scratchpad}",
                    input_variables=["input", "agent_scratchpad"],
                ),
                callbacks=[],
            ),
            output_parser=DiagnosisWorkflowParser(
                lab_test_mapping_df=self.lab_test_mapping_df
            ),
            stop=[],
            allowed_tools=["Physical Examination", "Laboratory Tests", "Imaging"],
            # 117 (PE) + 189 (input) + 100 (buffer) = 406 tokens. After summarizing: 189 (input) + 30 (summary) + 100 (buffer) = 319 tokens
            max_context_length=350,
            tags=self.tags,
            lab_test_mapping_df=self.lab_test_mapping_df,
        )

        # 189 tokens
        kwargs = {
            "input": "A very long input that requires us to truncate. A very long input that requires us to truncate.  A very long input that requires us to truncate.  A very long input that requires us to truncate.  A very long input that requires us to truncate.  A very long input that requires us to truncate.  A very long input that requires us to truncate.  A very long input that requires us to truncate. A very long input that requires us to truncate. A very long input that requires us to truncate.  A very long input that requires us to truncate.  A very long input that requires us to truncate.  A very long input that requires us to truncate.  A very long input that requires us to truncate.  A very long input that requires us to truncate.  A very long input that requires us to truncate. "
        }
        summary = agent._construct_scratchpad(intermediate_steps, **kwargs)

        self.assertEqual(summary, expected_output)

    @unittest.skip("later")
    def test_create_scratchpad_summarize_truncate(self):
        intermediate_steps = [
            (
                AgentAction(
                    tool="Physical Examination",
                    tool_input={"action_input": None},
                    log="",
                ),
                "Physical Examination\nThe physical examination was a very good physical examination that showed a lot of important symptoms. It was very long examination that went on for a very long time. A very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very long time.",
                # 117 tokens
            ),
        ]

        responses = ["Good PE."]
        tokenizer = LlamaTokenizer.from_pretrained("models/WizardLM-70B-V1.0-GPTQ")
        fake_llm = FakeLLM()
        fake_llm.load_model(responses=responses, tokenizer=tokenizer)
        agent = CustomZeroShotAgent(
            llm_chain=LLMChain(
                llm=fake_llm,
                prompt=PromptTemplate(
                    template="{input}{agent_scratchpad}",
                    input_variables=["input", "agent_scratchpad"],
                ),
                callbacks=[],
            ),
            output_parser=DiagnosisWorkflowParser(
                lab_test_mapping_df=self.lab_test_mapping_df
            ),
            stop=[],
            allowed_tools=["Physical Examination", "Laboratory Tests", "Imaging"],
            # 189 (input) + 117 (PE) + 100 (buffer) = 406 tokens. After summarizing: 190 (input) + 30 (summary) + 100 (buffer) = 319 tokens
            max_context_length=300,
            tags=self.tags,
            lab_test_mapping_df=self.lab_test_mapping_df,
            summarize=True,
        )

        # 189 tokens
        kwargs = {
            "input": "A very long input that requires us to truncate. A very long input that requires us to truncate.  A very long input that requires us to truncate.  A very long input that requires us to truncate.  A very long input that requires us to truncate.  A very long input that requires us to truncate.  A very long input that requires us to truncate.  A very long input that requires us to truncate. A very long input that requires us to truncate. A very long input that requires us to truncate.  A very long input that requires us to truncate.  A very long input that requires us to truncate.  A very long input that requires us to truncate.  A very long input that requires us to truncate.  A very long input that requires us to truncate.  A very long input that requires us to truncate. "
        }
        summary = agent._construct_scratchpad(intermediate_steps, **kwargs)

        # 30 tokens
        expected_output = """A summary of information I know thus far:
Action: Physical Examination
Observation: Good PE.
Thought:"""
        expected_output = truncate_text(tokenizer, expected_output, 300 - 190 - 100)
        expected_output += f"""{self.tags["ai_tag_end"]}{self.tags["user_tag_start"]}Provide a Final Diagnosis and Treatment.{self.tags["user_tag_end"]}{self.tags["ai_tag_start"]}Final"""
        self.assertEqual(summary, expected_output)

    def test_summarize_steps_invalid_action(self):
        intermediate_steps = [
            (
                AgentAction(
                    tool="Physical Examination",
                    tool_input={"action_input": None},
                    log="",
                ),
                "Physical Examination\nThe physical examination was a very good physical examination that showed a lot of important symptoms.",
            ),
            (
                AgentAction(
                    tool="Provide a diagnosis and treatment OR a valid tool. That",
                    tool_input={"action_input": None},
                    log="I would like to use the magic solver tool. Action: Magic Solver",
                ),
                "Provide a diagnosis and treatment OR a valid tool. That is not a valid tool. Try one of the following: Physical Examination, Laboratory Tests, Imaging.",
            ),
            (
                AgentAction(
                    tool="Physical Examination",
                    tool_input={"action_input": None},
                    log="",
                ),
                "Physical Examination\nThe physical examination was an even better physical examination that showed more important symptoms.",
            ),
        ]

        responses = ["Good PE.", "Better PE."]
        tokenizer = LlamaTokenizer.from_pretrained("models/WizardLM-70B-V1.0-GPTQ")
        fake_llm = FakeLLM()
        fake_llm.load_model(responses=responses, tokenizer=tokenizer)
        agent = CustomZeroShotAgent(
            llm_chain=LLMChain(
                llm=fake_llm,
                prompt=PromptTemplate(
                    template="{agent_scratchpad}", input_variables=["agent_scratchpad"]
                ),
                callbacks=[],
            ),
            output_parser=DiagnosisWorkflowParser(
                lab_test_mapping_df=self.lab_test_mapping_df
            ),
            stop=[],
            allowed_tools=["Physical Examination", "Laboratory Tests", "Imaging"],
            max_context_length=4000,
            tags=self.tags,
            lab_test_mapping_df=self.lab_test_mapping_df,
            summarize=True,
        )

        summary = agent._summarize_steps(intermediate_steps)

        expected_output = """A summary of information I know thus far:
Action: Physical Examination
Observation: Good PE.
I tried 'Action: Magic Solver', but it was an invalid request.
Action: Physical Examination
Observation: Better PE.
Thought:"""

        self.assertEqual(summary, expected_output)

    def test_summarize_steps_invalid_last_action(self):
        intermediate_steps = [
            (
                AgentAction(
                    tool="Physical Examination",
                    tool_input={"action_input": None},
                    log="",
                ),
                "Physical Examination\nThe physical examination was a very good physical examination that showed a lot of important symptoms.",
            ),
            (
                AgentAction(
                    tool="Provide a diagnosis and treatment OR a valid tool. That",
                    tool_input={"action_input": None},
                    log="I would like to use the magic solver tool.",
                ),
                "Provide a diagnosis and treatment OR a valid tool. That is not a valid tool. Try one of the following: Physical Examination, Laboratory Tests, Imaging.",
            ),
        ]

        responses = ["Good PE."]
        tokenizer = LlamaTokenizer.from_pretrained("models/WizardLM-70B-V1.0-GPTQ")
        fake_llm = FakeLLM()
        fake_llm.load_model(responses=responses, tokenizer=tokenizer)
        agent = CustomZeroShotAgent(
            llm_chain=LLMChain(
                llm=fake_llm,
                prompt=PromptTemplate(
                    template="{agent_scratchpad}", input_variables=["agent_scratchpad"]
                ),
                callbacks=[],
            ),
            output_parser=DiagnosisWorkflowParser(
                lab_test_mapping_df=self.lab_test_mapping_df
            ),
            stop=[],
            allowed_tools=["Physical Examination", "Laboratory Tests", "Imaging"],
            max_context_length=4000,
            tags=self.tags,
            lab_test_mapping_df=self.lab_test_mapping_df,
            summarize=True,
        )

        summary = agent._summarize_steps(intermediate_steps)

        expected_output = """A summary of information I know thus far:
Action: Physical Examination
Observation: Good PE.
I tried 'I would like to use the magic solver tool.', but it was an invalid request.</s> USER: Please choose a valid tool from ['Physical Examination', 'Laboratory Tests', 'Imaging'] or provide a Final Diagnosis and Treatment.</s> ASSISTANT: Thought:"""

        self.assertEqual(summary, expected_output)

    def test_summarize_steps_input_too_long(self):
        too_long_action_input = """Laboratory Tests:
Run CBC: N/A
(Blood) Alanine Aminotransferase (ALT): 416.0 IU/L
(Blood) Asparate Aminotransferase (AST): 303.0 IU/L
(Blood) Alkaline Phosphatase: 148.0 IU/L
(Blood) Gamma Glutamyltransferase: 152.0 IU/L
(Blood) Bilirubin, Direct: 4.9 mg/dL
(Blood) Bilirubin, Indirect: 7.0 mg/dL
(Blood) Bilirubin, Total: 22.0 mg/dL
(Blood) PT: 28.2 sec
(Blood) INR(PT): 2.6
(Blood) Albumin: 3.3 g/dL
(Blood) Urea Nitrogen: 6.0 mg/dL
(Blood) Sodium, Whole Blood: 131.0 mEq/L
(Blood) Sodium: 138.0 mEq/L
(Blood) Free Calcium: 0.94 mmol/L
(Blood) Calcium, Total: 9.3 mg/dL
(Blood) Calculated Total CO2: 23.0 mEq/L
(Blood) Chloride, Whole Blood: 100.0 mEq/L
(Blood) Chloride: 96.0 mEq/L
(Blood) Creatinine: 0.6 mg/dL
(Blood) Glucose: 108.0 mg/dL
(Blood) Glucose: 113.0 mg/dL
(Blood) Phosphate: 2.8 mg/dL
(Blood) Potassium, Whole Blood: 3.4 mEq/L
(Blood) Potassium: 3.1 mEq/L
(Blood) Thyroid Stimulating Hormone: 0.73 uIU/mL
(Blood) Thyroxine (T4), Free: 1.3 ng/dL
Anti-TPO Ab: N/A
Anti-HCV Ab: N/A
HIV Ag/Ab: N/A
EBV VCA IgM & IgG: N/A
CMV IgM & IgG: N/A
Hepatitis panel (A, B, C): N/A
ANA: N/A
ANCA: N/A
anti-dsDNA: N/A
anti-Smith: N/A
anti-double stranded DNA: N/A
anti-SSB: N/A
anti-SSA: N/A
anti-centromere: N/A
anti-scl70: N/A
anti-Jo1: N/A
anti-Ku: N/A
anti-PM-SCL: N/A
anti-U1-RNP: N/A
anti-RNP: N/A
anti-Sm: N/A
anti-nucleosome: N/A
anti-fibrinogen: N/A
anti-actin: N/A
anti-lysyl oxidase: N/A
(Blood) Anti-Mitochondrial Antibody: NEGATIVE.
anti-soluble liver antigen: N/A
anti-liver kidney microsomes type 1: N/A
anti-thyroglobulin: N/A
anti-thyroperoxidase: N/A
anti-ZGMB: N/A
anti-GBM: N/A
anti-CENP-B: N/A
anti-CENP-C: N/A
anti-CENP-E: N/A
anti-CENP-F: N/A
anti-CENP-H: N/A
anti-CENP-I: N/A
anti-CENP-J: N/A
anti-CENP-K: N/A
anti-CENP-L: N/A
anti-CENP-N: N/A
anti-CENP-O: N/A
anti-CENP-P: N/A
anti-CENP-Q: N/A
anti-CENP-R: N/A
anti-CENP-S: N/A
anti-CENP-T: N/A
anti-CENP-U: N/A
anti-CENP-V: N/A
anti-CENP-W: N/A
anti-CENP-X: N/A
anti-CENP-Y: N/A
anti-CENP-Z: N/A
anti-COL1A1: N/A
anti-COL1A2: N/A
anti-COL2A1: N/A
anti-COL3A1: N/A
anti-COL4A1: N/A
anti-COL4A2: N/A
anti-COL5A1: N/A
anti-COL5A2: N/A
anti-COL6A1: N/A
anti-COL6A2: N/A
anti-COL6A3: N/A
anti-COL7A1: N/A
anti-COL9A1: N/A
anti-COL9A2: N/A
anti-COL9A3: N/A
anti-COL10A1: N/A
anti-COL11A1: N/A
anti-COL11A2: N/A
anti-COL12A1: N/A
anti-COL13A1: N/A
anti-COL14A1: N/A
anti-COL15A1: N/A
anti-COL16A1: N/A
anti-COL17A1: N/A
anti-COL18A1: N/A
anti-COL19A1: N/A
anti-COL20A1: N/A
anti-COL21A1: N/A
anti-COL22A1: N/A
anti-COL23A1: N/A
anti-COL24A1: N/A
anti-COL25A1: N/A
anti-COL26A1: N/A
anti-COL27A1: N/A
anti-COL28A1: N/A
anti-COL29A1: N/A
anti-COL30A1: N/A
anti-COL31A1: N/A
anti-COL32A1: N/A
anti-COL33A1: N/A
anti-COL34A1: N/A
anti-COL35A1: N/A
anti-COL36A1: N/A
anti-COL37A1: N/A
anti-COL38A1: N/A
anti-COL39A1: N/A
anti-COL40A1: N/A
anti-COL41A1: N/A
anti-COL42A1: N/A
anti-COL43A1: N/A
anti-COL44A1: N/A
anti-COL45A1: N/A
anti-COL46A1: N/A
anti-COL47A1: N/A
anti-COL48A1: N/A
anti-COL49A1: N/A
anti-COL50A1: N/A
anti-COL51A1: N/A
anti-COL52A1: N/A
anti-COL53A1: N/A
anti-COL54A1: N/A
anti-COL55A1: N/A
anti-COL56A1: N/A
anti-COL57A1: N/A
anti-COL58A1: N/A
anti-COL59A1: N/A
anti-COL60A1: N/A
anti-COL61A1: N/A
anti-COL62A1: N/A
anti-COL63A1: N/A
anti-COL64A1: N/A
anti-COL65A1: N/A
anti-COL66A1: N/A
anti-COL67A1: N/A
anti-COL68A1: N/A
anti-COL69A1: N/A
anti-COL70A1: N/A
anti-COL71A1: N/A
anti-COL72A1: N/A
anti-COL73A1: N/A
anti-COL74A1: N/A
anti-COL75A1: N/A
anti-COL76A1: N/A
anti-COL77A1: N/A
anti-COL78A1: N/A
anti-COL79A1: N/A
anti-COL80A1: N/A
anti-COL81A1: N/A
anti-COL82A1: N/A
anti-COL83A1: N/A
anti-COL84A1: N/A
anti-COL85A1: N/A
anti-COL86A1: N/A
anti-COL87A1: N/A
anti-COL88A1: N/A
anti-COL89A1: N/A
anti-COL90A1: N/A
anti-COL91A1: N/A
anti-COL92A1: N/A
anti-COL93A1: N/A
anti-COL94A1: N/A
anti-COL95A1: N/A
anti-COL96A1: N/A
anti-COL97A1: N/A
anti-COL98A1: N/A
anti-COL99A1: N/A
anti-COL100A1: N/A
anti-COL101A1: N/A
anti-COL102A1: N/A
anti-COL103A1: N/A
anti-COL104A1: N/A
anti-COL105A1: N/A
anti-COL106A1: N/A
anti-COL107A1: N/A
anti-COL108A1: N/A
anti-COL109A1: N/A
anti-COL110A1: N/A
anti-COL111A1: N/A
anti-COL112A1: N/A
anti-COL113A1: N/A
anti-COL114A1: N/A
anti-COL115A1: N/A
anti-COL116A1: N/A
anti-COL117A1: N/A
anti-COL118A1: N/A
anti-COL119A1: N/A
anti-COL120A1: N/A
anti-COL121A1: N/A
anti-COL122A1: N/A
anti-COL123A1: N/A
anti-COL124A1: N/A
anti-COL125A1: N/A
anti-COL126A1: N/A
anti-COL127A1: N/A
anti-COL128A1: N/A
anti-COL129A1: N/A
anti-COL130A1: N/A
anti-COL131A1: N/A
anti-COL132A1: N/A
anti-COL133A1: N/A
anti-COL134A1: N/A
anti-COL135A1: N/A
anti-COL136A1: N/A
anti-COL137A1: N/A
anti-COL138A1: N/A
anti-COL139A1: N/A
anti-COL140A1: N/A
anti-COL141A1: N/A
anti-COL142A1: N/A
anti-COL143A1: N/A
anti-COL144A1: N/A
anti-COL145A1: N/A
anti-COL146A1: N/A
anti-COL147A1: N/A
anti-COL148A1: N/A
anti-COL149A1: N/A
anti-COL150A1: N/A
anti-COL151A1: N/A
anti-COL152A1: N/A
anti-COL153A1: N/A
anti-COL154A1: N/A
anti-COL155A1: N/A
anti-COL156A1: N/A
anti-COL157A1: N/A
anti-COL158A1: N/A
anti-COL159A1: N/A
anti-COL160A1: N/A
anti-COL161A1: N/A
anti-COL162A1: N/A
anti-COL163A1: N/A
anti-COL164A1: N/A
anti-COL165A1: N/A
anti-COL166A1: N/A
anti-COL167A1: N/A
anti-COL168A1: N/A
anti-COL169A1: N/A
anti-COL170A1: N/A
anti-COL171A1: N/A
anti-COL172A1: N/A
anti-COL173A1: N/A
anti-COL174A1: N/A
anti-COL175A1: N/A
anti-COL176A1: N/A
anti-COL177A1: N/A
anti-COL178A1: N/A
anti-COL179A1: N/A
anti-COL180A1: N/A
anti-COL181A1: N/A
anti-COL182A1: N/A
anti-COL183A1: N/A
anti-COL184A1: N/A
anti-COL185A1: N/A
anti-COL186A1: N/A
anti-COL187A1: N/A
anti-COL188A1: N/A
anti-COL189A1: N/A
anti-COL190A1: N/A
anti-COL191A1: N/A
anti-COL192A1: N/A
anti-COL193A1: N/A
anti-COL194A1: N/A
anti-COL195A1: N/A
anti-COL196A1: N/A
anti-COL197A1: N/A
anti-COL198A1: N/A
anti-COL199A1: N/A
anti-COL200A1: N/A
anti-COL201A1: N/A
anti-COL202A1: N/A
anti-COL203A1: N/A
anti-COL204A1: N/A
anti-COL205A1: N/A
anti-COL206A1: N/A
anti-COL207A1: N/A
anti-COL208A1: N/A
anti-COL209A1: N/A
anti-COL210A1: N/A
anti-COL211A1: N/A
anti-COL212A1: N/A
anti-COL213A1: N/A
anti-COL214A1: N/A
anti-COL215A1: N/A
anti-COL216A1: N/A
anti-COL217A1: N/A
anti-COL218A1: N/A
anti-COL219A1: N/A
anti-COL220A1: N/A
anti-COL221A1: N/A
anti-COL222A1: N/A
anti-COL223A1: N/A
anti-COL224A1: N/A
anti-COL225A1: N/A
anti-COL226A1: N/A
anti-COL227A1: N/A
anti-COL228A1: N/A
anti-COL229A1: N/A
anti-COL230A1: N/A
anti-COL231A1: N/A
anti-COL232A1: N/A
anti-COL233A1: N/A
anti-COL234A1: N/A
anti-COL235A1: N/A
anti-COL236A1: N/A
anti-COL237A1: N/A
anti-COL238A1: N/A
anti-COL239A1: N/A
anti-COL240A1: N/A
anti-COL241A1: N/A
anti-COL242A1: N/A
anti-COL243A1: N/A
anti-COL244A1: N/A
anti-COL245A1: N/A
anti-COL246A1: N/A
anti-COL247A1: N/A
anti-COL248A1: N/A
anti-COL249A1: N/A
anti-COL250A1: N/A
anti-COL251A1: N/A
anti-COL252A1: N/A
anti-COL253A1: N/A
anti-COL254A1: N/A
anti-COL255A1: N/A
anti-COL256A1: N/A
anti-COL257A1: N/A
anti-COL258A1: N/A
anti-COL259A1: N/A
anti-COL260A1: N/A
anti-COL261A1: N/A
anti-COL262A1: N/A
anti-COL263A1: N/A
anti-COL264A1: N/A
anti-COL265A1: N/A
anti-COL266A1: N/A
anti-COL267A1: N/A
anti-COL268A1: N/A
anti-COL269A1: N/A
anti-COL270A1: N/A
anti-COL271A1: N/A
anti-COL272A1: N/A
anti-COL273A1: N/A
anti-COL274A1: N/A
anti-COL275A1: N/A
anti-COL276A1: N/A
anti-COL277A1: N/A
anti-COL278A1: N/A
anti-COL279A1: N/A
anti-COL: N/A"""
        intermediate_steps = [
            (
                AgentAction(
                    tool="Physical Examination",
                    tool_input={"action_input": None},
                    log="",
                ),
                too_long_action_input,
            ),
        ]

        responses = ["Good PE."]
        tokenizer = LlamaTokenizer.from_pretrained("models/WizardLM-70B-V1.0-GPTQ")
        fake_llm = FakeLLM()
        fake_llm.load_model(responses=responses, tokenizer=tokenizer)
        agent = CustomZeroShotAgent(
            llm_chain=LLMChain(
                llm=fake_llm,
                prompt=PromptTemplate(
                    template="{agent_scratchpad}", input_variables=["agent_scratchpad"]
                ),
                callbacks=[],
            ),
            output_parser=DiagnosisWorkflowParser(
                lab_test_mapping_df=self.lab_test_mapping_df
            ),
            stop=[],
            allowed_tools=["Physical Examination", "Laboratory Tests", "Imaging"],
            max_context_length=500,
            tags=self.tags,
            lab_test_mapping_df=self.lab_test_mapping_df,
            summarize=True,
        )

        summary = agent._summarize_steps(intermediate_steps)

        expected_output = """A summary of information I know thus far:
Action: Physical Examination
Observation: Good PE.
Thought:"""

        self.assertEqual(summary, expected_output)


if __name__ == "__main__":
    unittest.main()
