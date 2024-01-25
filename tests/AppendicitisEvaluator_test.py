import unittest
from evaluators.appendicitis_evaluator import AppendicitisEvaluator
from agents.AgentAction import AgentAction


class TestAppendicitisEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = AppendicitisEvaluator()
        self.evaluator.procedures_icd9 = []
        self.evaluator.procedures_icd10 = []
        self.evaluator.procedures_discharge = []
        self.maxDiff = None

    ##############
    # LABORATORY #
    ##############

    def test_score_laboratory_tests_inflammation(self):
        test_ordered = [51301]  # White Blood Cells
        action = AgentAction(
            tool="Laboratory Tests",
            tool_input={"action_input": test_ordered},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_laboratory_tests(action)

        self.assertEqual(self.evaluator.scores["Laboratory Tests"], 1)
        self.assertEqual(
            self.evaluator.answers["Correct Laboratory Tests"],
            {"Inflammation": [51301]},
        )
        self.assertEqual(self.evaluator.answers["Unnecessary Laboratory Tests"], [])

    def test_score_laboratory_tests_neutral_cbc(self):
        test_ordered = [51279]  # Red Blood Cells
        action = AgentAction(
            tool="Laboratory Tests",
            tool_input={"action_input": test_ordered},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_laboratory_tests(action)

        self.assertEqual(self.evaluator.scores["Laboratory Tests"], 0)
        self.assertEqual(
            self.evaluator.answers["Correct Laboratory Tests"], {"Inflammation": []}
        )
        self.assertEqual(self.evaluator.answers["Unnecessary Laboratory Tests"], [])

    def test_score_laboratory_tests_neutral_lfp(self):
        test_ordered = [50861]  # ALT
        action = AgentAction(
            tool="Laboratory Tests",
            tool_input={"action_input": test_ordered},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_laboratory_tests(action)

        self.assertEqual(self.evaluator.scores["Laboratory Tests"], 0)
        self.assertEqual(
            self.evaluator.answers["Correct Laboratory Tests"], {"Inflammation": []}
        )
        self.assertEqual(self.evaluator.answers["Unnecessary Laboratory Tests"], [])

    def test_score_laboratory_tests_neutral_rfp(self):
        test_ordered = [50862]  # Albumin
        action = AgentAction(
            tool="Laboratory Tests",
            tool_input={"action_input": test_ordered},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_laboratory_tests(action)

        self.assertEqual(self.evaluator.scores["Laboratory Tests"], 0)
        self.assertEqual(
            self.evaluator.answers["Correct Laboratory Tests"], {"Inflammation": []}
        )
        self.assertEqual(self.evaluator.answers["Unnecessary Laboratory Tests"], [])

    def test_score_laboratory_tests_neutral_urine(self):
        test_ordered = [51508]  # Urine Color
        action = AgentAction(
            tool="Laboratory Tests",
            tool_input={"action_input": test_ordered},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_laboratory_tests(action)

        self.assertEqual(self.evaluator.scores["Laboratory Tests"], 0)
        self.assertEqual(
            self.evaluator.answers["Correct Laboratory Tests"], {"Inflammation": []}
        )
        self.assertEqual(self.evaluator.answers["Unnecessary Laboratory Tests"], [])

    ###########
    # IMAGING #
    ###########

    def test_score_imaging_appendicitis_US(self):
        region = "Abdomen"
        modality = "Ultrasound"

        self.evaluator.score_imaging(region, modality)

        self.assertEqual(self.evaluator.scores["Imaging"], 2)

    def test_score_imaging_appendicitis_CT(self):
        region = "Abdomen"
        modality = "CT"

        self.evaluator.score_imaging(region, modality)

        self.assertEqual(self.evaluator.scores["Imaging"], 1)

    def test_score_imaging_appendicitis_wrong_modality(self):
        region = "Abdomen"
        modality = "Radiograph"

        self.evaluator.score_imaging(region, modality)

        self.assertEqual(self.evaluator.scores["Imaging"], 0)

    def test_score_imaging_appendicitis_wrong_region(self):
        region = "Head"
        modality = "Ultrasound"

        self.evaluator.score_imaging(region, modality)

        self.assertEqual(self.evaluator.scores["Imaging"], 0)

    def test_score_imaging_appendicitis_US_then_CT(self):
        region = "Abdomen"
        modality = "Ultrasound"

        self.evaluator.score_imaging(region, modality)

        region = "Abdomen"
        modality = "CT"

        self.evaluator.score_imaging(region, modality)

        self.assertEqual(self.evaluator.scores["Imaging"], 2)

    def test_score_imaging_appendicitis_CT_then_US(self):
        region = "Abdomen"
        modality = "CT"

        self.evaluator.score_imaging(region, modality)

        region = "Abdomen"
        modality = "Ultrasound"

        self.evaluator.score_imaging(region, modality)

        self.assertEqual(self.evaluator.scores["Imaging"], 1)

    def test_score_imaging_appendicitis_repeat(self):
        region = "Abdomen"
        modality = "CT"

        self.evaluator.score_imaging(region, modality)

        region = "Abdomen"
        modality = "CT"

        self.evaluator.score_imaging(region, modality)

        self.assertEqual(self.evaluator.scores["Imaging"], 1)

    def test_score_imaging_action_appendicitis_US(self):
        imaging_dict = {"region": "Abdomen", "modality": "Ultrasound"}
        action = AgentAction(
            tool="Imaging",
            tool_input={"action_input": imaging_dict},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_imaging_action(action)

        self.assertEqual(
            self.evaluator.answers["Correct Imaging"],
            [imaging_dict],
        )
        self.assertEqual(self.evaluator.answers["Unnecessary Imaging"], [])

    def test_score_imaging_action_appendicitis_CT(self):
        imaging_dict = {"region": "Abdomen", "modality": "CT"}
        action = AgentAction(
            tool="Imaging",
            tool_input={"action_input": imaging_dict},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_imaging_action(action)

        self.assertEqual(
            self.evaluator.answers["Correct Imaging"],
            [imaging_dict],
        )
        self.assertEqual(self.evaluator.answers["Unnecessary Imaging"], [])

    def test_score_imaging_action_appendicitis_wrong_modality(self):
        imaging_dict = {"region": "Abdomen", "modality": "Radiograph"}
        action = AgentAction(
            tool="Imaging",
            tool_input={"action_input": imaging_dict},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_imaging_action(action)

        self.assertEqual(self.evaluator.answers["Correct Imaging"], [])
        self.assertEqual(
            self.evaluator.answers["Unnecessary Imaging"],
            [imaging_dict],
        )

    def test_score_imaging_action_appendicitis_wrong_region(self):
        imaging_dict = {"region": "Head", "modality": "CT"}
        action = AgentAction(
            tool="Imaging",
            tool_input={"action_input": imaging_dict},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_imaging_action(action)

        self.assertEqual(self.evaluator.answers["Correct Imaging"], [])
        self.assertEqual(
            self.evaluator.answers["Unnecessary Imaging"],
            [imaging_dict],
        )

    def test_score_imaging_action_appendicitis_US_then_CT(self):
        imaging_dict_us = {"region": "Abdomen", "modality": "Ultrasound"}
        action = AgentAction(
            tool="Imaging",
            tool_input={"action_input": imaging_dict_us},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_imaging_action(action)

        imaging_dict_ct = {"region": "Abdomen", "modality": "CT"}
        action = AgentAction(
            tool="Imaging",
            tool_input={"action_input": imaging_dict_ct},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_imaging_action(action)

        self.assertEqual(
            self.evaluator.answers["Correct Imaging"],
            [
                imaging_dict_us,
                imaging_dict_ct,
            ],
        )
        self.assertEqual(self.evaluator.answers["Unnecessary Imaging"], [])

    def test_score_imaging_action_appendicitis_CT_then_US(self):
        imaging_dict_ct = {"region": "Abdomen", "modality": "CT"}
        action = AgentAction(
            tool="Imaging",
            tool_input={"action_input": imaging_dict_ct},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_imaging_action(action)

        imaging_dict_us = {"region": "Abdomen", "modality": "Ultrasound"}
        action = AgentAction(
            tool="Imaging",
            tool_input={"action_input": imaging_dict_us},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_imaging_action(action)

        self.assertEqual(
            self.evaluator.answers["Correct Imaging"],
            [
                imaging_dict_ct,
                imaging_dict_us,
            ],
        )
        self.assertEqual(self.evaluator.answers["Unnecessary Imaging"], [])

    def test_score_imaging_action_appendicitis_repeat(self):
        imaging_dict = {"region": "Abdomen", "modality": "CT"}
        action = AgentAction(
            tool="Imaging",
            tool_input={"action_input": imaging_dict},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_imaging_action(action)

        action = AgentAction(
            tool="Imaging",
            tool_input={"action_input": imaging_dict},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_imaging_action(action)

        self.assertEqual(
            self.evaluator.answers["Correct Imaging"],
            [
                imaging_dict,
            ],
        )
        self.assertEqual(
            self.evaluator.answers["Unnecessary Imaging"],
            [
                imaging_dict,
            ],
        )

    #############
    # DIAGNOSIS #
    #############

    def test_evaluator_diagnosis(self):
        result = {
            "input": "",
            "output": "Final Diagnosis: Acute Appendicitis",
            "intermediate_steps": [],
        }

        eval = self.evaluator._evaluate_agent_trajectory(
            prediction=result["output"],
            input=result["input"],
            reference=(
                "Acute Appendicitis",
                ["Unspecified acute appendicitis without abscess"],
                [],
                [],
                [],
                [],
            ),
            agent_trajectory=result["intermediate_steps"],
        )

        self.assertEqual(eval["scores"]["Diagnosis"], 1)
        self.assertEqual(eval["answers"]["Diagnosis"], "Acute Appendicitis")

    def test_score_diagnosis_acute_appendicitis(self):
        self.evaluator.answers["Diagnosis"] = "Acute Appendicitis"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 1)

    def test_score_diagnosis_appendicitis(self):
        self.evaluator.answers["Diagnosis"] = "Appendicitis"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 1)

    def test_score_diagnosis_appendix_abscess(self):
        self.evaluator.answers["Diagnosis"] = "Appendix Abscess"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 1)

    def test_score_diagnosis_ruptured_appendix(self):
        self.evaluator.answers["Diagnosis"] = "Ruptured Appendix"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 1)

    def test_score_diagnosis_ruptured_appendix_2(self):
        self.evaluator.answers["Diagnosis"] = "Rupture in the appendix"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 1)

    def test_score_diagnosis_periappendiceal_abscess(self):
        self.evaluator.answers["Diagnosis"] = "Periappendiceal Abscess"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 1)

    def test_score_diagnosis_appendiceal_perforation(self):
        self.evaluator.answers["Diagnosis"] = "Appendiceal Perforation"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 1)

    def test_score_diagnosis_unspecific(self):
        self.evaluator.answers["Diagnosis"] = "Disease of the appendix"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 0)

    def test_score_diagnosis_unspecific_2(self):
        self.evaluator.answers["Diagnosis"] = "appendix"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 0)

    def test_score_diagnosis_unspecific_3(self):
        self.evaluator.answers["Diagnosis"] = "Problems in the appendix"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 0)

    #############
    # TREATMENT #
    #############

    def test_evaluator_treatment_appendicitis_appendectomy(self):
        action = AgentAction(
            tool="",
            tool_input={},
            log="",
            custom_parsings=0,
        )
        result = {
            "input": "",
            "output": "Final Diagnosis: Acute Appendicitis\nTreatment: Appendectomy",
            "intermediate_steps": [(action, "")],
        }

        eval = self.evaluator._evaluate_agent_trajectory(
            prediction=result["output"],
            input=result["input"],
            reference=(
                "Acute Appendicitis",
                ["Unspecified acute appendicitis without abscess"],
                [4701],
                [],
                [],
            ),
            agent_trajectory=result["intermediate_steps"],
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Appendectomy": True,
                "Antibiotics": True,
                "Support": True,
            },
        )

        self.assertEqual(eval["answers"]["Treatment"], "Appendectomy")

        self.assertEqual(eval["answers"]["Treatment Requested"]["Appendectomy"], True)
        for treatment in eval["answers"]["Treatment Requested"].keys():
            if treatment != "Appendectomy":
                self.assertEqual(
                    eval["answers"]["Treatment Requested"][treatment], False
                )

    def test_score_treatment_appendicitis_appendectomy_icd9(self):
        self.evaluator.answers["Treatment"] = "Appendectomy"
        self.evaluator.procedures_icd9 = [4701]
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Appendectomy": True,
                "Antibiotics": False,
                "Support": False,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Appendectomy": True,
                "Antibiotics": True,
                "Support": True,
            },
        )

    def test_score_treatment_appendicitis_appendectomy_icd10(self):
        self.evaluator.answers["Treatment"] = "Appendectomy"
        self.evaluator.procedures_icd10 = ["0DTJ4ZZ"]
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Appendectomy": True,
                "Antibiotics": False,
                "Support": False,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Appendectomy": True,
                "Antibiotics": True,
                "Support": True,
            },
        )

    def test_score_treatment_appendicitis_appendectomy_discharge(self):
        self.evaluator.answers["Treatment"] = "Appendectomy"
        self.evaluator.procedures_discharge = ["LAPORASCOPIC APPENDECTOMY"]
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Appendectomy": True,
                "Antibiotics": False,
                "Support": False,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Appendectomy": True,
                "Antibiotics": True,
                "Support": True,
            },
        )

    def test_score_treatment_appendicitis_appendectomy_no_procedure(self):
        self.evaluator.answers["Treatment"] = "Appendectomy"
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Appendectomy": True,
                "Antibiotics": False,
                "Support": False,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Appendectomy": False,
                "Antibiotics": True,
                "Support": True,
            },
        )

    def test_score_treatment_appendicitis_appendectomy_alternate_keywords(self):
        self.evaluator.answers["Treatment"] = "Surgical removal of the appendix"
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Appendectomy": True,
                "Antibiotics": False,
                "Support": False,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Appendectomy": False,
                "Antibiotics": True,
                "Support": True,
            },
        )

    def test_score_treatment_appendicitis_appendectomy_alternate_keywords_multiline(
        self,
    ):
        self.evaluator.answers[
            "Treatment"
        ] = "Surgical removal of the head is recommended. The appendix looks fine."
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Appendectomy": False,
                "Antibiotics": False,
                "Support": False,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Appendectomy": False,
                "Antibiotics": True,
                "Support": True,
            },
        )

    def test_score_treatment_appendicitis_antibiotics(self):
        self.evaluator.answers["Treatment"] = "Antibiotics"
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Appendectomy": False,
                "Antibiotics": True,
                "Support": False,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Appendectomy": False,
                "Antibiotics": True,
                "Support": True,
            },
        )

    def test_score_treatment_appendicitis_support_fluid(self):
        self.evaluator.answers["Treatment"] = "Support patient with fluids."
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Appendectomy": False,
                "Antibiotics": False,
                "Support": True,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Appendectomy": False,
                "Antibiotics": True,
                "Support": True,
            },
        )

    def test_score_treatment_appendicitis_support_pain(self):
        self.evaluator.answers["Treatment"] = "Support patient with pain killers."
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Appendectomy": False,
                "Antibiotics": False,
                "Support": True,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Appendectomy": False,
                "Antibiotics": True,
                "Support": True,
            },
        )

    def test_score_treatment_appendicitis_support_analgesics(self):
        self.evaluator.answers["Treatment"] = "Support patient with analgesics."
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Appendectomy": False,
                "Antibiotics": False,
                "Support": True,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Appendectomy": False,
                "Antibiotics": True,
                "Support": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
