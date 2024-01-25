import unittest
from evaluators.cholecystitis_evaluator import CholecystitisEvaluator
from agents.AgentAction import AgentAction


class TestCholecystitisEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = CholecystitisEvaluator()
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
            {"Inflammation": [51301], "Liver": [], "Gallbladder": []},
        )
        self.assertEqual(self.evaluator.answers["Unnecessary Laboratory Tests"], [])

    def test_score_laboratory_tests_liver(self):
        test_ordered = [50861, 50878]  # ALT, AST
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
            {"Inflammation": [], "Liver": [50861, 50878], "Gallbladder": []},
        )
        self.assertEqual(self.evaluator.answers["Unnecessary Laboratory Tests"], [])

    def test_score_laboratory_tests_Gallbladder(self):
        test_ordered = [50883, 50927]  # Bilirubin, Gamma Glutamyltransferase
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
            {"Inflammation": [], "Liver": [], "Gallbladder": [50883, 50927]},
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
            self.evaluator.answers["Correct Laboratory Tests"],
            {
                "Inflammation": [],
                "Liver": [],
                "Gallbladder": [],
            },
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
            self.evaluator.answers["Correct Laboratory Tests"],
            {
                "Inflammation": [],
                "Liver": [],
                "Gallbladder": [],
            },
        )
        self.assertEqual(self.evaluator.answers["Unnecessary Laboratory Tests"], [])

    def test_score_laboratory_tests_neutral_alp(self):
        test_ordered = [50863]  # ALP
        action = AgentAction(
            tool="Laboratory Tests",
            tool_input={"action_input": test_ordered},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_laboratory_tests(action)

        self.assertEqual(self.evaluator.scores["Laboratory Tests"], 0)
        self.assertEqual(
            self.evaluator.answers["Correct Laboratory Tests"],
            {
                "Inflammation": [],
                "Liver": [],
                "Gallbladder": [],
            },
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
            self.evaluator.answers["Correct Laboratory Tests"],
            {
                "Inflammation": [],
                "Liver": [],
                "Gallbladder": [],
            },
        )
        self.assertEqual(self.evaluator.answers["Unnecessary Laboratory Tests"], [])

    ###########
    # IMAGING #
    ###########

    def test_score_imaging_cholecystitis_US(self):
        region = "Abdomen"
        modality = "Ultrasound"

        self.evaluator.score_imaging(region, modality)

        self.assertEqual(self.evaluator.scores["Imaging"], 2)

    def test_score_imaging_cholecystitis_HIDA(self):
        region = "Abdomen"
        modality = "HIDA"

        self.evaluator.score_imaging(region, modality)

        self.assertEqual(self.evaluator.scores["Imaging"], 2)

    def test_score_imaging_cholecystitis_MRI(self):
        region = "Abdomen"
        modality = "MRI"

        self.evaluator.score_imaging(region, modality)

        self.assertEqual(self.evaluator.scores["Imaging"], 1)

    def test_score_imaging_cholecystitis_EUS(self):
        region = "Abdomen"
        modality = "EUS"

        self.evaluator.score_imaging(region, modality)

        self.assertEqual(self.evaluator.scores["Imaging"], 1)

    def test_score_imaging_cholecystitis_wrong_modality(self):
        region = "Abdomen"
        modality = "CT"

        self.evaluator.score_imaging(region, modality)

        self.assertEqual(self.evaluator.scores["Imaging"], 0)

    def test_score_imaging_cholecystitis_wrong_region(self):
        region = "Head"
        modality = "Ultrasound"

        self.evaluator.score_imaging(region, modality)

        self.assertEqual(self.evaluator.scores["Imaging"], 0)

    def test_score_imaging_cholecystitis_MRI_then_US(self):
        region = "Abdomen"
        modality = "MRI"

        self.evaluator.score_imaging(region, modality)

        region = "Abdomen"
        modality = "Ultrasound"

        self.evaluator.score_imaging(region, modality)

        self.assertEqual(self.evaluator.scores["Imaging"], 1)

    def test_score_imaging_cholecystitis_US_then_MRI(self):
        region = "Abdomen"
        modality = "Ultrasound"

        self.evaluator.score_imaging(region, modality)

        region = "Abdomen"
        modality = "MRI"

        self.evaluator.score_imaging(region, modality)

        self.assertEqual(self.evaluator.scores["Imaging"], 2)

    def test_score_imaging_cholecystitis_repeat(self):
        region = "Abdomen"
        modality = "EUS"

        self.evaluator.score_imaging(region, modality)

        region = "Abdomen"
        modality = "EUS"

        self.evaluator.score_imaging(region, modality)

        self.assertEqual(self.evaluator.scores["Imaging"], 1)

    def test_score_imaging_action_cholecystitis_US(self):
        imaging_dict = {"region": "Abdomen", "modality": "Ultrasound"}
        action = AgentAction(
            tool="Imaging",
            tool_input={"action_input": imaging_dict},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_imaging_action(action)

        self.assertEqual(self.evaluator.answers["Correct Imaging"], [imaging_dict])
        self.assertEqual(self.evaluator.answers["Unnecessary Imaging"], [])

    def test_score_imaging_action_cholecystitis_HIDA(self):
        imaging_dict = {"region": "Abdomen", "modality": "HIDA"}
        action = AgentAction(
            tool="Imaging",
            tool_input={"action_input": imaging_dict},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_imaging_action(action)

        self.assertEqual(self.evaluator.answers["Correct Imaging"], [imaging_dict])
        self.assertEqual(self.evaluator.answers["Unnecessary Imaging"], [])

    def test_score_imaging_action_cholecystitis_MRI(self):
        imaging_dict = {"region": "Abdomen", "modality": "MRI"}
        action = AgentAction(
            tool="Imaging",
            tool_input={"action_input": imaging_dict},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_imaging_action(action)

        self.assertEqual(self.evaluator.answers["Correct Imaging"], [imaging_dict])
        self.assertEqual(self.evaluator.answers["Unnecessary Imaging"], [])

    def test_score_imaging_action_cholecystitis_EUS(self):
        imaging_dict = {"region": "Abdomen", "modality": "EUS"}
        action = AgentAction(
            tool="Imaging",
            tool_input={"action_input": imaging_dict},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_imaging_action(action)

        self.assertEqual(self.evaluator.answers["Correct Imaging"], [imaging_dict])
        self.assertEqual(self.evaluator.answers["Unnecessary Imaging"], [])

    def test_score_imaging_action_cholecystitis_wrong_modality(self):
        imaging_dict = {"region": "Abdomen", "modality": "CT"}
        action = AgentAction(
            tool="Imaging",
            tool_input={"action_input": imaging_dict},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_imaging_action(action)

        self.assertEqual(self.evaluator.answers["Correct Imaging"], [])
        self.assertEqual(self.evaluator.answers["Unnecessary Imaging"], [imaging_dict])

    def test_score_imaging_action_cholecystitis_wrong_region(self):
        imaging_dict = {"region": "Head", "modality": "MRI"}
        action = AgentAction(
            tool="Imaging",
            tool_input={"action_input": imaging_dict},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_imaging_action(action)

        self.assertEqual(self.evaluator.answers["Correct Imaging"], [])
        self.assertEqual(self.evaluator.answers["Unnecessary Imaging"], [imaging_dict])

    def test_score_imaging_action_cholecystitis_MRI_then_US(self):
        imaging_dict_mri = {"region": "Abdomen", "modality": "MRI"}
        action = AgentAction(
            tool="Imaging",
            tool_input={"action_input": imaging_dict_mri},
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
            [imaging_dict_mri, imaging_dict_us],
        )
        self.assertEqual(self.evaluator.answers["Unnecessary Imaging"], [])

    def test_score_imaging_action_cholecystitis_US_then_MRI(self):
        imaging_dict_us = {"region": "Abdomen", "modality": "Ultrasound"}
        action = AgentAction(
            tool="Imaging",
            tool_input={"action_input": imaging_dict_us},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_imaging_action(action)

        imaging_dict_mri = {"region": "Abdomen", "modality": "MRI"}
        action = AgentAction(
            tool="Imaging",
            tool_input={"action_input": imaging_dict_mri},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_imaging_action(action)

        self.assertEqual(
            self.evaluator.answers["Correct Imaging"],
            [imaging_dict_us, imaging_dict_mri],
        )
        self.assertEqual(self.evaluator.answers["Unnecessary Imaging"], [])

    def test_score_imaging_action_cholecystitis_repeat(self):
        imaging_dict = {"region": "Abdomen", "modality": "EUS"}
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

        self.assertEqual(self.evaluator.answers["Correct Imaging"], [imaging_dict])
        self.assertEqual(self.evaluator.answers["Unnecessary Imaging"], [imaging_dict])

    #############
    # DIAGNOSIS #
    #############

    def test_evaluator_diagnosis(self):
        result = {
            "input": "",
            "output": "Final Diagnosis: Acute Cholecystitis",
            "intermediate_steps": [],
        }

        eval = self.evaluator._evaluate_agent_trajectory(
            prediction=result["output"],
            input=result["input"],
            reference=(
                "Acute Cholecystitis",
                ["Calculus of gallbladder with acute cholecystitis, with obstruction"],
                [],
                [],
                [],
                [],
            ),
            agent_trajectory=result["intermediate_steps"],
        )

        self.assertEqual(eval["scores"]["Diagnosis"], 1)
        self.assertEqual(eval["answers"]["Diagnosis"], "Acute Cholecystitis")

    def test_score_diagnosis_acute_cholecystitis(self):
        self.evaluator.answers["Diagnosis"] = "Acute Cholecystitis"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 1)

    def test_score_diagnosis_cholecystitis(self):
        self.evaluator.answers["Diagnosis"] = "Cholecystitis"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 1)

    def test_score_diagnosis_chronic_cholecystitis(self):
        self.evaluator.answers["Diagnosis"] = "Chronic Cholecystitis"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 1)

    def test_score_diagnosis_calculous_cholecystitis(self):
        self.evaluator.answers["Diagnosis"] = "Calculous Cholecystitis"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 1)

    def test_score_diagnosis_acalculous_cholecystitis(self):
        self.evaluator.answers["Diagnosis"] = "Acalculous Cholecystitis"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 1)

    def test_score_diagnosis_gangrenous_cholecystitis(self):
        self.evaluator.answers["Diagnosis"] = "Gangrenous Cholecystitis"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 1)

    def test_score_diagnosis_Emphysematous_cholecystitis(self):
        self.evaluator.answers["Diagnosis"] = "Emphysematous Cholecystitis"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 1)

    def test_score_diagnosis_gallbladder_inflammation(self):
        self.evaluator.answers["Diagnosis"] = "Gallbladder inflammation"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 1)

    def test_score_diagnosis_gallbladder_inflammed(self):
        self.evaluator.answers["Diagnosis"] = "Inflammed gallbladder"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 1)

    def test_score_diagnosis_gallbladder_inflammation_of(self):
        self.evaluator.answers["Diagnosis"] = "Inflammation of the gallbladder"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 1)

    def test_score_diagnosis_unspecific(self):
        self.evaluator.answers["Diagnosis"] = "Gallbladder disease"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 0)

    def test_score_diagnosis_unspecific_2(self):
        self.evaluator.answers["Diagnosis"] = "Symptomatic gallstones"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 0)

    def test_score_diagnosis_unspecific_3(self):
        self.evaluator.answers["Diagnosis"] = "Gallstones"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 0)

    def test_score_diagnosis_unspecific_4(self):
        self.evaluator.answers["Diagnosis"] = "Cholelithiasis"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 0)

    def test_score_diagnosis_unspecific_5(self):
        self.evaluator.answers["Diagnosis"] = "Gallbladder Polyps"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 0)

    def test_score_diagnosis_unspecific_6(self):
        self.evaluator.answers["Diagnosis"] = "Choledocholithiasis"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 0)

    def test_score_diagnosis_unspecific_7(self):
        self.evaluator.answers["Diagnosis"] = "Gallbladder"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 0)

    #############
    # TREATMENT #
    #############

    def test_evaluator_treatment_cholecystitis_cholecystectomy(self):
        action = AgentAction(
            tool="",
            tool_input={},
            log="",
            custom_parsings=0,
        )
        result = {
            "input": "",
            "output": "Final Diagnosis: Acute Cholecystitis\nTreatment: Cholecystectomy",
            "intermediate_steps": [(action, "")],
        }

        eval = self.evaluator._evaluate_agent_trajectory(
            prediction=result["output"],
            input=result["input"],
            reference=(
                "Acute Cholecystitis",
                ["Calculus of gallbladder with acute cholecystitis, with obstruction"],
                [5123],
                [],
                [],
            ),
            agent_trajectory=result["intermediate_steps"],
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Cholecystectomy": True,
                "Antibiotics": True,
                "Support": True,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Cholecystectomy": True,
                "Antibiotics": False,
                "Support": False,
            },
        )

        self.assertEqual(
            eval["answers"]["Treatment"],
            "Cholecystectomy",
        )

    def test_score_treatment_cholecystitis_cholecystectomy_icd9(self):
        self.evaluator.answers["Treatment"] = "Cholecystectomy"
        self.evaluator.procedures_icd9 = [5123]
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Cholecystectomy": True,
                "Antibiotics": False,
                "Support": False,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Cholecystectomy": True,
                "Antibiotics": True,
                "Support": True,
            },
        )

    def test_score_treatment_cholecystitis_cholecystectomy_icd10(self):
        self.evaluator.answers["Treatment"] = "Cholecystectomy"
        self.evaluator.procedures_icd10 = ["0FB44ZZ"]
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Cholecystectomy": True,
                "Antibiotics": False,
                "Support": False,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Cholecystectomy": True,
                "Antibiotics": True,
                "Support": True,
            },
        )

    def test_score_treatment_cholecystitis_cholecystectomy_discharge(self):
        self.evaluator.answers["Treatment"] = "Cholecystectomy"
        self.evaluator.procedures_discharge = ["___ CHOLECYSTECTOMY"]
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Cholecystectomy": True,
                "Antibiotics": False,
                "Support": False,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Cholecystectomy": True,
                "Antibiotics": True,
                "Support": True,
            },
        )

    def test_score_treatment_cholecystitis_cholecystectomy_no_procedure(self):
        self.evaluator.answers["Treatment"] = "Cholecystectomy"
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Cholecystectomy": True,
                "Antibiotics": False,
                "Support": False,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Cholecystectomy": False,
                "Antibiotics": True,
                "Support": True,
            },
        )

    def test_score_treatment_cholecystitis_cholecystectomy_alternate_keywords(self):
        self.evaluator.answers["Treatment"] = "Surgical removal of the gallbladder"
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Cholecystectomy": True,
                "Antibiotics": False,
                "Support": False,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Cholecystectomy": False,
                "Antibiotics": True,
                "Support": True,
            },
        )

    def test_score_treatment_cholecystitis_antibiotics(self):
        self.evaluator.answers["Treatment"] = "Antibiotics"
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Cholecystectomy": False,
                "Antibiotics": True,
                "Support": False,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Cholecystectomy": False,
                "Antibiotics": True,
                "Support": True,
            },
        )

    def test_score_treatment_cholecystitis_support_all(self):
        self.evaluator.answers[
            "Treatment"
        ] = "Support the patient with fluids and analgesics"
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Cholecystectomy": False,
                "Antibiotics": False,
                "Support": True,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Cholecystectomy": False,
                "Antibiotics": True,
                "Support": True,
            },
        )

    def test_score_treatment_cholecystitis_support_partial(self):
        self.evaluator.answers[
            "Treatment"
        ] = "Provide adequate support to the patient. Give them plenty of fluids."
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Cholecystectomy": False,
                "Antibiotics": False,
                "Support": True,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Cholecystectomy": False,
                "Antibiotics": True,
                "Support": True,
            },
        )

    def test_score_treatment_cholecystitis_cholecystectomy_and_support(self):
        self.evaluator.answers[
            "Treatment"
        ] = "Perform an early laparoscopic cholecystectomy. Support the patient with fluids and analgesics"
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Cholecystectomy": True,
                "Antibiotics": False,
                "Support": True,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Cholecystectomy": False,
                "Antibiotics": True,
                "Support": True,
            },
        )

    def test_score_treatment_cholecystitis_all(self):
        self.evaluator.answers[
            "Treatment"
        ] = "Perform an early laparoscopic cholecystectomy. Support the patient with fluids and analgesics. Start antibiotics."
        self.evaluator.procedures_discharge = ["cholecystectomy"]
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Cholecystectomy": True,
                "Antibiotics": True,
                "Support": True,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Cholecystectomy": True,
                "Antibiotics": True,
                "Support": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
