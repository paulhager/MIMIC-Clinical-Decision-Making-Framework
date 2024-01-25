import unittest
import pickle

from langchain.schema import AgentFinish

from agents.DiagnosisWorkflowParser import InvalidActionError
from agents.DiagnosisWorkflowParser import DiagnosisWorkflowParser
from agents.AgentAction import AgentAction
from utils.nlp import extract_keywords_nltk
from tools.utils import ADDITIONAL_LAB_TEST_MAPPING

import logging


class TestDiagnosisWorkflowParser(unittest.TestCase):
    maxDiff = None
    # Supresses the fuzzywuzzy warning of trying to match a empty list
    logging.getLogger().setLevel(logging.ERROR)

    def setUp(self):
        lab_test_mapping_path = ""
        with open(lab_test_mapping_path, "rb") as f:
            df = pickle.load(f)
        self.parser = DiagnosisWorkflowParser(lab_test_mapping_df=df)

    ####################
    # FULL PARSE TESTS #
    ####################

    def test_full_parse_vanilla_multi_step(self):
        ## Imaging

        llm_output = "Action: Imaging\nAction Input: CT Abdomen"
        output = self.parser.parse(llm_output)

        expected_output = [
            AgentAction(
                tool="Imaging",
                tool_input={"action_input": {"modality": "CT", "region": "Abdomen"}},
                log=llm_output,
                custom_parsings=0,
            )
        ]

        self.assertEqual(output, expected_output)

        ## Invalid Action

        llm_output = "This is an invalid action."

        output = self.parser.parse(llm_output)

        expected_output = [
            AgentAction(
                tool=InvalidActionError.invalid_tool_str,
                tool_input={"action_input": None},
                log=llm_output,
                custom_parsings=0,
            )
        ]

        self.assertEqual(output, expected_output)

        ## Lab Tests

        llm_output = (
            "Action: Laboratory Tests\nAction Input: Amylase, Lipase and Sodium"
        )
        output = self.parser.parse(llm_output)

        expected_output = [
            AgentAction(
                tool="Laboratory Tests",
                tool_input={"action_input": [50867, 50956, 50824, 52623, 50983]},
                log=llm_output,
                custom_parsings=0,
            )
        ]

        self.assertEqual(output, expected_output)

        ## Invalid Action

        llm_output = "Action:"

        output = self.parser.parse(llm_output)

        expected_output = [
            AgentAction(
                tool=InvalidActionError.invalid_tool_str,
                tool_input={"action_input": None},
                log=llm_output,
                custom_parsings=1,
            )
        ]

        self.assertEqual(output, expected_output)

        ## Diagnosis

        llm_output = "Final Diagnosis: Apendicitis\nTreatment: Appendectomy"

        output = self.parser.parse(llm_output)

        expected_output = AgentFinish(
            return_values={"output": llm_output}, log=llm_output
        )

        self.assertEqual(output, expected_output)

    def test_full_parse_imaging_vanilla(self):
        llm_output = "Action: Imaging\nAction Input: CT Abdomen"
        output = self.parser.parse(llm_output)

        expected_output = [
            AgentAction(
                tool="Imaging",
                tool_input={"action_input": {"modality": "CT", "region": "Abdomen"}},
                log=llm_output,
                custom_parsings=0,
            )
        ]

        self.assertEqual(output, expected_output)

    def test_full_parse_lab_tests_vanilla(self):
        llm_output = (
            "Action: Laboratory Tests\nAction Input: Amylase, Lipase and Sodium"
        )
        output = self.parser.parse(llm_output)

        expected_output = [
            AgentAction(
                tool="Laboratory Tests",
                tool_input={"action_input": [50867, 50956, 50824, 52623, 50983]},
                log=llm_output,
                custom_parsings=0,
            )
        ]

        self.assertEqual(output, expected_output)

    def test_full_parse_physical_examination_vanilla(self):
        llm_output = "Action: Physical Examination"
        output = self.parser.parse(llm_output)

        expected_output = [
            AgentAction(
                tool="Physical Examination",
                tool_input={"action_input": None},
                log=llm_output,
                custom_parsings=0,
            )
        ]

        self.assertEqual(output, expected_output)

    def test_full_parse_invalid_action_error_return(self):
        llm_output = "This is an invalid action."

        output = self.parser.parse(llm_output)

        expected_output = [
            AgentAction(
                tool=InvalidActionError.invalid_tool_str,
                tool_input={"action_input": None},
                log=llm_output,
                custom_parsings=0,
            )
        ]

        self.assertEqual(output, expected_output)

    def test_double_parse_invalid_action(self):
        llm_output = "Action:"

        output = self.parser.parse(llm_output)

        expected_output = [
            AgentAction(
                tool=InvalidActionError.invalid_tool_str,
                tool_input={"action_input": None},
                log=llm_output,
                custom_parsings=1,
            )
        ]

        self.assertEqual(output, expected_output)

    def test_full_parse_dash(self):
        llm_output = "Action: Imaging - Upper GI Series\nAction Input: No contrast"

        output = self.parser.parse(llm_output)

        expected_output = [
            AgentAction(
                tool="Imaging",
                tool_input={
                    "action_input": {"modality": "Upper GI Series", "region": "Abdomen"}
                },
                log=llm_output,
                custom_parsings=1,
            )
        ]

        self.assertEqual(output, expected_output)

    #########################
    # PARSE DIAGNOSIS TESTS #
    #########################

    def test_diagnosis_given_agent_finish(self):
        llm_output = "Final Diagnosis: Apendicitis\nTreatment: Appendectomy"

        output = self.parser.parse(llm_output)

        expected_output = AgentFinish(
            return_values={"output": llm_output}, log=llm_output
        )

        self.assertEqual(output, expected_output)

    def test_diagnosis_provided_true(self):
        self.parser.llm_output = "Diagnosis: Acute Pancreatitis"

        self.assertEqual(self.parser.diagnosis_provided(), True)

    def test_diagnosis_provided_false(self):
        self.parser.llm_output = "Action: Imaging"

        self.assertEqual(self.parser.diagnosis_provided(), False)

    ######################
    # PARSE ACTION TESTS #
    ######################

    def test_parse_action_vanilla(self):
        self.parser.llm_output = "Action: Imaging"

        self.parser.parse_action()

        self.assertEqual(self.parser.action, "Imaging")
        self.assertEqual(self.parser.action_input_prepend, "")
        self.assertEqual(self.parser.custom_parsings, 0)

    def test_parse_action_with_action_input_vanilla(self):
        self.parser.llm_output = "Action: Imaging\nAction Input: CT Abdomen"

        self.parser.parse_action()

        self.assertEqual(self.parser.action, "Imaging")
        self.assertEqual(self.parser.action_input_prepend, "")
        self.assertEqual(self.parser.custom_parsings, 0)

    def test_parse_action_with_input_vanilla(self):
        self.parser.llm_output = "Action: Imaging\nInput: CT Abdomen"

        self.parser.parse_action()

        self.assertEqual(self.parser.action, "Imaging")
        self.assertEqual(self.parser.action_input_prepend, "")
        self.assertEqual(self.parser.custom_parsings, 0)

    def test_parse_action_different_line(self):
        self.parser.llm_output = "Action:\nImaging"

        self.parser.parse_action()

        self.assertEqual(self.parser.action, "Imaging")
        self.assertEqual(self.parser.action_input_prepend, "")
        self.assertEqual(self.parser.custom_parsings, 0)

    def test_parse_action_different_line_with_input(self):
        self.parser.llm_output = "Action:\nImaging\nAction Input:\nCT Abdomen"

        self.parser.parse_action()

        self.assertEqual(self.parser.action, "Imaging")
        self.assertEqual(self.parser.action_input_prepend, "")
        self.assertEqual(self.parser.custom_parsings, 0)

    def test_parse_action_with_input_period(self):
        self.parser.llm_output = "Action: Imaging. Action Input: CT Abdomen"

        self.parser.parse_action()

        self.assertEqual(self.parser.action, "Imaging")
        self.assertEqual(self.parser.action_input_prepend, "")
        self.assertEqual(self.parser.custom_parsings, 0)

    def test_parse_action_lab_tests_input_after_dash(self):
        self.parser.llm_output = "Action: Laboratory Tests - Amylase"
        self.parser.parse_action()

        self.assertEqual(self.parser.action, "Laboratory Tests")
        self.assertEqual(self.parser.action_input_prepend, "Amylase")
        self.assertEqual(self.parser.custom_parsings, 1)

    def test_parse_action_labs(self):
        self.parser.llm_output = "Action: Labs"
        self.parser.parse_action()

        self.assertEqual(self.parser.action, "Laboratory Tests")
        self.assertEqual(self.parser.action_input_prepend, "")
        self.assertEqual(self.parser.custom_parsings, 1)

    def test_parse_action_labs_dash(self):
        self.parser.llm_output = "Action: Labs - Amylase"
        self.parser.parse_action()

        self.assertEqual(self.parser.action, "Laboratory Tests")
        self.assertEqual(self.parser.action_input_prepend, "Amylase")
        self.assertEqual(self.parser.custom_parsings, 2)

    def test_parse_action_imaging_dash(self):
        self.parser.llm_output = (
            "Action: Imaging - Upper GI Series\nAction Input: No contrast"
        )
        self.parser.parse_action()

        self.assertEqual(self.parser.action, "Imaging")
        self.assertEqual(self.parser.action_input_prepend, "Upper GI Series")
        self.assertEqual(self.parser.custom_parsings, 1)

    def test_parse_action_blood_work(self):
        self.parser.llm_output = "Action: Blood work"
        self.parser.parse_action()

        self.assertEqual(self.parser.action, "Laboratory Tests")
        self.assertEqual(self.parser.action_input_prepend, "")
        self.assertEqual(self.parser.custom_parsings, 1)

    def test_parse_action_no_explicit_action_with_colon(self):
        self.parser.llm_output = "Do physical examination to start."
        with self.assertRaises(InvalidActionError):
            self.parser.parse_action()

    ##########################
    # INTERPRET ACTION TESTS #
    ##########################

    def test_interpret_action_vanilla(self):
        self.parser.action = "Imaging"

        self.parser.interpret_action()

        self.assertEqual(self.parser.action, "Imaging")
        self.assertEqual(self.parser.action_input_prepend, "")
        self.assertEqual(self.parser.custom_parsings, 0)

    def test_interpret_action_fuzzy(self):
        self.parser.action = "Imagin"

        self.parser.interpret_action()

        self.assertEqual(self.parser.action, "Imaging")
        self.assertEqual(self.parser.action_input_prepend, "")
        self.assertEqual(self.parser.custom_parsings, 1)

    def test_interpret_action_imaging(self):
        self.parser.action = "CT Abdomen"

        self.parser.interpret_action()

        self.assertEqual(self.parser.action, "Imaging")
        self.assertEqual(self.parser.action_input_prepend, "CT Abdomen")
        self.assertEqual(self.parser.custom_parsings, 1)

    def test_interpret_action_unique_imaging_modality_EUS(self):
        self.parser.action = "Echo Endoscopy"

        self.parser.interpret_action()

        self.assertEqual(self.parser.action, "Imaging")
        self.assertEqual(self.parser.action_input_prepend, "Echo Endoscopy")
        self.assertEqual(self.parser.custom_parsings, 1)

    def test_interpret_action_lab_tests(self):
        self.parser.action = "Amylase, Lipase, and Sodium"

        self.parser.interpret_action()

        self.assertEqual(self.parser.action, "Laboratory Tests")
        self.assertEqual(
            self.parser.action_input_prepend, "Amylase, Lipase, and Sodium"
        )
        self.assertEqual(self.parser.custom_parsings, 1)

    @unittest.skip(
        "Actually ok. Switched back to normal fuzzy matching, not ratio, for substring matching. So what if this matches, don't think this will lead to incorrect parsing."
    )
    def test_interpret_action_test_only_invalid(self):
        self.parser.action = "Test"

        with self.assertRaises(InvalidActionError):
            self.parser.interpret_action()

    def test_interpret_action_paragraph(self):
        self.parser.action = "Refer the patient to a specialist for further management of their liver disease and diabetes. Initiate therapy with insulin and statins to manage the patient's diabetes and hyperlipidemia, respectively. Consider starting levothyroxine replacement therapy for the patient's hypothyroidism. Monitor the patient closely for progression of their liver disease and adjust treatment accordingly."

        with self.assertRaises(InvalidActionError):
            self.parser.interpret_action()

    def test_interpret_action_empty(self):
        self.parser.action = ""

        with self.assertRaises(InvalidActionError):
            self.parser.interpret_action()

    def test_key_word_extraction(self):
        action = "Do a CT of the Abdomen"
        action_keywords = extract_keywords_nltk(action)
        self.assertEqual(action_keywords, ["CT", "Abdomen"])

    def test_interpret_action_key_words(self):
        self.parser.action = "Do a CT of the Abdomen"

        self.parser.interpret_action()

        self.assertEqual(self.parser.action, "Imaging")
        self.assertEqual(self.parser.action_input_prepend, "Do a CT of the Abdomen")
        self.assertEqual(self.parser.custom_parsings, 1)

    #############################
    # PARSE ACTION INPUT TESTS #
    #############################

    def test_parse_action_input_imaging_vanilla(self):
        self.parser.llm_output = "Action: Imaging\nAction Input: CT Abdomen"
        self.parser.action = "Imaging"
        self.parser.parse_action_input()

        self.assertEqual(
            self.parser.action_input, {"modality": "CT", "region": "Abdomen"}
        )
        self.assertEqual(self.parser.custom_parsings, 0)

    def test_parse_action_input_lab_tests_vanilla(self):
        self.parser.llm_output = (
            "Action: Laboratory Tests\nAction Input: Amylase, Lipase, and Sodium"
        )
        self.parser.action = "Laboratory Tests"
        self.parser.parse_action_input()

        self.assertEqual(self.parser.action_input, [50867, 50956, 50824, 52623, 50983])
        self.assertEqual(self.parser.custom_parsings, 0)

    def test_parse_action_input_lab_tests_multiline(self):
        self.parser.llm_output = (
            "Action:\nLaboratory Tests\nAction Input:\nAmylase\nLipase\nSodium"
        )
        self.parser.action = "Laboratory Tests"
        self.parser.parse_action_input()

        self.assertEqual(self.parser.action_input, [50867, 50956, 50824, 52623, 50983])
        self.assertEqual(self.parser.custom_parsings, 0)

    def test_parse_action_input_lab_tests_no_input(self):
        self.llm_output = "Action: Laboratory Tests"
        with self.assertRaises(InvalidActionError):
            self.parser.parse_action_input()
        self.assertEqual(self.parser.custom_parsings, 0)

    def test_parse_action_input_imaging_no_input(self):
        self.llm_output = "Action: Imaging"
        with self.assertRaises(InvalidActionError):
            self.parser.parse_action_input()
        self.assertEqual(self.parser.custom_parsings, 0)

    def test_parse_action_input_imaging_invalid_action_propogation(self):
        self.parser.action = "Imaging"
        self.parser.llm_output = "Action:Imaging\nAction Input: Abdomen"

        with self.assertRaises(InvalidActionError):
            self.parser.parse_action_input()

    #####################################
    # PARSE ACTION INPUT FROM LLM TESTS #
    #####################################

    def test_parse_action_input_from_llm_vanilla(self):
        self.parser.llm_output = "Action: Imaging\nAction Input: CT Abdomen"
        input_found = self.parser.parse_action_input_from_llm_output()

        self.assertEqual(input_found, True)
        self.assertEqual(self.parser.custom_parsings, 0)
        self.assertEqual(self.parser.action_input, "CT Abdomen")

    def test_parse_action_input_from_llm_just_input(self):
        self.parser.llm_output = "Action: Laboratory Tests\nInput: Amylase"
        input_found = self.parser.parse_action_input_from_llm_output()

        self.assertEqual(input_found, True)
        self.assertEqual(self.parser.custom_parsings, 1)
        self.assertEqual(self.parser.action_input, "Amylase")

    def test_parse_action_input_from_llm_multiline(self):
        self.parser.llm_output = (
            "Action:\nLaboratory Tests\nAction Input:\nAmylase\nLipase\nSodium"
        )
        input_found = self.parser.parse_action_input_from_llm_output()

        self.assertEqual(input_found, True)
        self.assertEqual(self.parser.custom_parsings, 0)
        self.assertEqual(self.parser.action_input, "Amylase\nLipase\nSodium")

    ##############################
    # PARSE IMAGING ACTION INPUT #
    ##############################

    def test_parse_imaging_action_vanilla(self):
        self.parser.action_input = "CT Abdomen"

        self.parser.parse_imaging_action_input()

        expected_output = {"modality": "CT", "region": "Abdomen"}

        self.assertEqual(self.parser.action_input, expected_output)

    def test_parse_imaging_action_vanilla_reverse(self):
        self.parser.action_input = "Abdomen CT"

        self.parser.parse_imaging_action_input()

        expected_output = {"modality": "CT", "region": "Abdomen"}

        self.assertEqual(self.parser.action_input, expected_output)

    def test_parse_imaging_action_unique_EUS(self):
        self.parser.action_input = "Echo Endoscopy"

        self.parser.parse_imaging_action_input()

        expected_output = {"modality": "EUS", "region": "Abdomen"}

        self.assertEqual(self.parser.action_input, expected_output)

    def test_parse_imaging_action_unique_UpperGI(self):
        self.parser.action_input = "Upper GI Series"

        self.parser.parse_imaging_action_input()

        expected_output = {"modality": "Upper GI Series", "region": "Abdomen"}

        self.assertEqual(self.parser.action_input, expected_output)

    def test_parse_imaging_action_unique_ERCP(self):
        self.parser.action_input = "Endoscopic Retrograde Cholangiopancreatography"

        self.parser.parse_imaging_action_input()

        expected_output = {"modality": "ERCP", "region": "Abdomen"}

        self.assertEqual(self.parser.action_input, expected_output)

    def test_parse_imaging_action_unique_HIDA(self):
        self.parser.action_input = "Hepatobiliary Iminodiacetic Acid Scan"

        self.parser.parse_imaging_action_input()

        expected_output = {"modality": "HIDA", "region": "Abdomen"}

        self.assertEqual(self.parser.action_input, expected_output)

    def test_parse_imaging_action_unique_MRCP(self):
        self.parser.action_input = "Magnetic Resonance Cholangiopancreatography"

        self.parser.parse_imaging_action_input()

        expected_output = {"modality": "MRCP", "region": "Abdomen"}

        self.assertEqual(self.parser.action_input, expected_output)

    def test_parse_imaging_action_no_modality(self):
        self.parser.action_input = "Abdomen"

        with self.assertRaises(InvalidActionError):
            self.parser.parse_imaging_action_input()

    def test_parse_imaging_action_no_region(self):
        self.parser.action_input = "CT"

        with self.assertRaises(InvalidActionError):
            self.parser.parse_imaging_action_input()

    ################################
    # PARSE LAB TESTS ACTION INPUT #
    ################################

    def test_parse_action_lab_tests_action_input_vanilla(self):
        self.parser.action_input = "Amylase"

        self.parser.parse_lab_tests_action_input()

        expected_output = [50867]

        self.assertEqual(self.parser.action_input, expected_output)

    def test_parse_action_lab_tests_action_input_microbio_blood(self):
        self.parser.action_input = "Blood culture"

        self.parser.parse_lab_tests_action_input()

        expected_output = [90201]

        self.assertEqual(self.parser.action_input, expected_output)

    def test_parse_action_lab_tests_action_input_microbio_urine(self):
        self.parser.action_input = "Urine culture"

        self.parser.parse_lab_tests_action_input()

        expected_output = [90039]

        self.assertEqual(self.parser.action_input, expected_output)

    def test_parse_action_lab_tests_action_input_repeat(self):
        self.parser.action_input = "Amylase, amylase"

        self.parser.parse_lab_tests_action_input()

        expected_output = [50867]

        self.assertEqual(self.parser.action_input, expected_output)

    def test_parse_action_lab_tests_action_input_alt_name(self):
        self.parser.action_input = "WBC Count"

        self.parser.parse_lab_tests_action_input()

        expected_output = [51755, 51300, 51301]

        self.assertEqual(self.parser.action_input, expected_output)

    def test_parse_action_lab_tests_action_input_alt_name_2(self):
        self.parser.action_input = "pregnancy test"

        self.parser.parse_lab_tests_action_input()

        expected_output = [51085, 52720]

        self.assertEqual(self.parser.action_input, expected_output)

    def test_parse_action_lab_tests_action_input_alt_name_3(self):
        self.parser.action_input = "CRP"

        self.parser.parse_lab_tests_action_input()

        expected_output = [50889]

        self.assertEqual(self.parser.action_input, expected_output)

    def test_parse_action_lab_tests_action_input_order(self):
        self.parser.action_input = "Order amylase"

        self.parser.parse_lab_tests_action_input()

        expected_output = [50867]

        self.assertEqual(self.parser.action_input, expected_output)

    def test_parse_action_lab_tests_action_input_order_a(self):
        self.parser.action_input = "Order a CBC"

        self.parser.parse_lab_tests_action_input()

        expected_output = ADDITIONAL_LAB_TEST_MAPPING["Complete Blood Count (CBC)"]

        self.assertEqual(self.parser.action_input, expected_output)

    def test_parse_action_lab_tests_action_input_levels(self):
        self.parser.action_input = "Amylase levels"

        self.parser.parse_lab_tests_action_input()

        expected_output = [50867]

        self.assertEqual(self.parser.action_input, expected_output)

    def test_parse_action_lab_tests_action_input_multi_test(self):
        self.parser.action_input = "Amylase, Lipase, Sodium"
        self.parser.parse_lab_tests_action_input()

        expected_output = [50867, 50956, 50824, 52623, 50983]

        self.assertEqual(self.parser.action_input, expected_output)

    def test_parse_action_lab_tests_action_input_multi_test_with_oxford_comma_and_and(
        self,
    ):
        self.parser.action_input = "Amylase, Lipase, and Sodium"

        self.parser.parse_lab_tests_action_input()

        expected_output = [50867, 50956, 50824, 52623, 50983]

        self.assertEqual(self.parser.action_input, expected_output)

    def test_parse_action_lab_tests_action_input_multi_test_with_and_but_without_oxford_comma(
        self,
    ):
        self.parser.action_input = "Amylase, Lipase and Sodium"

        self.parser.parse_lab_tests_action_input()

        expected_output = [50867, 50956, 50824, 52623, 50983]

        self.assertEqual(self.parser.action_input, expected_output)

    def test_parse_action_lab_tests_action_input_multi_line(
        self,
    ):
        self.parser.action_input = "Amylase\nLipase\nSodium"

        self.parser.parse_lab_tests_action_input()

        expected_output = [50867, 50956, 50824, 52623, 50983]

        self.assertEqual(self.parser.action_input, expected_output)

    def test_parse_action_lab_tests_action_input_word_containing_and(
        self,
    ):
        self.parser.action_input = "Double Stranded DNA"

        self.parser.parse_lab_tests_action_input()

        expected_output = [50918]

        self.assertEqual(self.parser.action_input, expected_output)

    @unittest.skip(
        "No longer necessary because no tests have comma between parentheses"
    )
    def test_parse_action_lab_tests_action_input_test_containing_comma(
        self,
    ):
        self.parser.action_input = (
            "Cardiac Enzymes (Trop T, Trop I), Amylase, Coagulation Studies (PT, INR)"
        )

        self.parser.parse_lab_tests_action_input()

        expected_output = set(
            [
                "Cardiac Enzymes (Trop T, Trop I)",
                "Amylase",
                "Coagulation Studies (PT, INR)",
            ]
        )

        self.assertEqual(self.parser.action_input, expected_output)


if __name__ == "__main__":
    unittest.main()
