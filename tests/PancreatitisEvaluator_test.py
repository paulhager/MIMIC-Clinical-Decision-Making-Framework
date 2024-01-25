import unittest
from evaluators.pancreatitis_evaluator import PancreatitisEvaluator
from agents.AgentAction import AgentAction


class TestPancreatitisEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = PancreatitisEvaluator()
        self.evaluator.discharge_diagnosis = "Acute pancreatitis"
        self.evaluator.icd_diagnoses = [
            "Acute pancreatitis without necrosis or infection, unspecified"
        ]
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
            {"Inflammation": [51301], "Pancreas": [], "Seriousness": []},
        )
        self.assertEqual(self.evaluator.answers["Unnecessary Laboratory Tests"], [])

    def test_score_laboratory_tests_pancreas(self):
        test_ordered = [50867, 50956]  # Amylase, Lipase
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
            {"Inflammation": [], "Pancreas": [50867, 50956], "Seriousness": []},
        )
        self.assertEqual(self.evaluator.answers["Unnecessary Laboratory Tests"], [])

    def test_score_laboratory_tests_seriousness(self):
        test_ordered = [
            51221,  # Hematocrit
            51006,  # Urea Nitrogen
            51000,  # Triglycerides
            50893,  # Calcium, Total
            50824,  # Sodium
            50822,  # Potassium
        ]
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
            {
                "Inflammation": [],
                "Pancreas": [],
                "Seriousness": [51221, 51006, 51000, 50893, 50824, 50822],
            },
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
                "Pancreas": [],
                "Seriousness": [],
            },
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
            self.evaluator.answers["Correct Laboratory Tests"],
            {
                "Inflammation": [],
                "Pancreas": [],
                "Seriousness": [],
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
                "Pancreas": [],
                "Seriousness": [],
            },
        )
        self.assertEqual(self.evaluator.answers["Unnecessary Laboratory Tests"], [])

    ###########
    # IMAGING #
    ###########

    def test_score_imaging_pancreatitis_US(self):
        region = "Abdomen"
        modality = "Ultrasound"

        self.evaluator.score_imaging(region, modality)

        self.assertEqual(self.evaluator.scores["Imaging"], 2)

    def test_score_imaging_pancreatitis_CT(self):
        region = "Abdomen"
        modality = "CT"

        self.evaluator.score_imaging(region, modality)

        self.assertEqual(self.evaluator.scores["Imaging"], 1)

    def test_score_imaging_pancreatitis_EUS(self):
        self.evaluator.discharge_diagnosis = "Acute pancreatitis"
        self.evaluator.icd_diagnoses = ["Acute pancreatitis"]

        region = "Abdomen"
        modality = "EUS"

        self.evaluator.score_imaging(region, modality)

        self.assertEqual(self.evaluator.scores["Imaging"], 0)

    def test_score_imaging_pancreatitis_EUS_biliary(self):
        self.evaluator.discharge_diagnosis = "Biliary acute pancreatitis"
        self.evaluator.icd_diagnoses = ["Biliary acute pancreatitis"]

        region = "Abdomen"
        modality = "Ultrasound"

        self.evaluator.score_imaging(region, modality)

        region = "Abdomen"
        modality = "EUS"

        self.evaluator.score_imaging(region, modality)

        self.assertEqual(self.evaluator.scores["Imaging"], 3)

    def test_score_imaging_pancreatitis_wrong_modality(self):
        region = "Abdomen"
        modality = "MRI"

        self.evaluator.score_imaging(region, modality)

        self.assertEqual(self.evaluator.scores["Imaging"], 0)

    def test_score_imaging_pancreatitis_wrong_region(self):
        region = "Head"
        modality = "Ultrasound"

        self.evaluator.score_imaging(region, modality)

        self.assertEqual(self.evaluator.scores["Imaging"], 0)

    def test_score_imaging_pancreatitis_US_then_CT(self):
        region = "Abdomen"
        modality = "Ultrasound"

        self.evaluator.score_imaging(region, modality)

        region = "Abdomen"
        modality = "CT"

        self.evaluator.score_imaging(region, modality)

        self.assertEqual(self.evaluator.scores["Imaging"], 2)

    def test_score_imaging_pancreatitis_CT_then_US(self):
        region = "Abdomen"
        modality = "CT"

        self.evaluator.score_imaging(region, modality)

        region = "Abdomen"
        modality = "Ultrasound"

        self.evaluator.score_imaging(region, modality)

        self.assertEqual(self.evaluator.scores["Imaging"], 1)

    def test_score_imaging_pancreatitis_repeat(self):
        region = "Abdomen"
        modality = "CT"

        self.evaluator.score_imaging(region, modality)

        region = "Abdomen"
        modality = "CT"

        self.evaluator.score_imaging(region, modality)

        self.assertEqual(self.evaluator.scores["Imaging"], 1)

    def test_score_imaging_action_pancreatitis_US(self):
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

    def test_score_imaging_action_pancreatitis_CT(self):
        imaging_dict = {"region": "Abdomen", "modality": "CT"}
        action = AgentAction(
            tool="Imaging",
            tool_input={"action_input": imaging_dict},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_imaging_action(action)

        self.assertEqual(self.evaluator.answers["Correct Imaging"], [imaging_dict])
        self.assertEqual(self.evaluator.answers["Unnecessary Imaging"], [])

    def test_score_imaging_action_pancreatitis_EUS(self):
        self.evaluator.discharge_diagnosis = "Acute pancreatitis"
        self.evaluator.icd_diagnoses = ["Acute pancreatitis"]

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

    def test_score_imaging_action_pancreatitis_EUS_biliary(self):
        self.evaluator.discharge_diagnosis = "Biliary acute pancreatitis"
        self.evaluator.icd_diagnoses = ["Biliary acute pancreatitis"]

        imaging_dict1 = {"region": "Abdomen", "modality": "Ultrasound"}
        action1 = AgentAction(
            tool="Imaging",
            tool_input={"action_input": imaging_dict1},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_imaging_action(action1)

        imaging_dict2 = {"region": "Abdomen", "modality": "EUS"}
        action2 = AgentAction(
            tool="Imaging",
            tool_input={"action_input": imaging_dict2},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_imaging_action(action2)

        self.assertEqual(
            self.evaluator.answers["Correct Imaging"], [imaging_dict1, imaging_dict2]
        )
        self.assertEqual(self.evaluator.answers["Unnecessary Imaging"], [])

    def test_score_imaging_action_pancreatitis_wrong_modality(self):
        imaging_dict = {"region": "Abdomen", "modality": "MRI"}
        action = AgentAction(
            tool="Imaging",
            tool_input={"action_input": imaging_dict},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_imaging_action(action)

        self.assertEqual(self.evaluator.answers["Correct Imaging"], [])
        self.assertEqual(self.evaluator.answers["Unnecessary Imaging"], [imaging_dict])

    def test_score_imaging_action_pancreatitis_wrong_region(self):
        imaging_dict = {"region": "Head", "modality": "CT"}
        action = AgentAction(
            tool="Imaging",
            tool_input={"action_input": imaging_dict},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_imaging_action(action)

        self.assertEqual(self.evaluator.answers["Correct Imaging"], [])
        self.assertEqual(self.evaluator.answers["Unnecessary Imaging"], [imaging_dict])

    def test_score_imaging_action_pancreatitis_US_then_CT(self):
        imaging_dict1 = {"region": "Abdomen", "modality": "Ultrasound"}
        action1 = AgentAction(
            tool="Imaging",
            tool_input={"action_input": imaging_dict1},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_imaging_action(action1)

        imaging_dict2 = {"region": "Abdomen", "modality": "CT"}
        action2 = AgentAction(
            tool="Imaging",
            tool_input={"action_input": imaging_dict2},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_imaging_action(action2)

        self.assertEqual(
            self.evaluator.answers["Correct Imaging"], [imaging_dict1, imaging_dict2]
        )

    def test_score_imaging_action_pancreatitis_CT_then_US(self):
        imaging_dict1 = {"region": "Abdomen", "modality": "CT"}
        action1 = AgentAction(
            tool="Imaging",
            tool_input={"action_input": imaging_dict1},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_imaging_action(action1)

        imaging_dict2 = {"region": "Abdomen", "modality": "Ultrasound"}
        action2 = AgentAction(
            tool="Imaging",
            tool_input={"action_input": imaging_dict2},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_imaging_action(action2)

        self.assertEqual(
            self.evaluator.answers["Correct Imaging"], [imaging_dict1, imaging_dict2]
        )

    def test_score_imaging_action_pancreatitis_repeat(self):
        imaging_dict = {"region": "Abdomen", "modality": "CT"}
        action1 = AgentAction(
            tool="Imaging",
            tool_input={"action_input": imaging_dict},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_imaging_action(action1)

        action2 = AgentAction(
            tool="Imaging",
            tool_input={"action_input": imaging_dict},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_imaging_action(action2)

        self.assertEqual(self.evaluator.answers["Correct Imaging"], [imaging_dict])
        self.assertEqual(self.evaluator.answers["Unnecessary Imaging"], [imaging_dict])

    #############
    # DIAGNOSIS #
    #############

    def test_evaluator_diagnosis(self):
        result = {
            "input": "",
            "output": "Final Diagnosis: Acute Pancreatitis",
            "intermediate_steps": [],
        }

        eval = self.evaluator._evaluate_agent_trajectory(
            prediction=result["output"],
            input=result["input"],
            reference=(
                "Acute Pancreatitis",
                ["Acute pancreatitis without necrosis or infection, unspecified"],
                [],
                [],
                [],
                [],
            ),
            agent_trajectory=result["intermediate_steps"],
        )

        self.assertEqual(eval["scores"]["Diagnosis"], 1)
        self.assertEqual(eval["answers"]["Diagnosis"], "Acute Pancreatitis")

    def test_score_diagnosis_acute_pancreatitis(self):
        self.evaluator.answers["Diagnosis"] = "Acute Pancreatitis"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 1)

    def test_score_diagnosis_pancreatitis(self):
        self.evaluator.answers["Diagnosis"] = "Pancreatitis"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 1)

    def test_score_diagnosis_chronic_pancreatitis(self):
        self.evaluator.answers["Diagnosis"] = "Chronic Pancreatitis"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 1)

    def test_score_diagnosis_necrotizing_pancreatitis(self):
        self.evaluator.answers["Diagnosis"] = "Necrotizing Pancreatitis"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 1)

    def test_score_diagnosis_necrotizing_pancreatitis_infection(self):
        self.evaluator.answers["Diagnosis"] = "Necrotizing Pancreatitis with infection"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 1)

    def test_score_diagnosis_Interstitial_Edematous_Pancreatitis(self):
        self.evaluator.answers["Diagnosis"] = "Interstitial Edematous Pancreatitis"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 1)

    def test_score_diagnosis_Hereditary_Pancreatitis(self):
        self.evaluator.answers["Diagnosis"] = "Hereditary Pancreatitis"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 1)

    def test_score_diagnosis_Alcoholic_Pancreatitis(self):
        self.evaluator.answers["Diagnosis"] = "Alcohol induced acute pancreatitis"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 1)

    def test_score_diagnosis_biliary_pancreatitis(self):
        self.evaluator.answers["Diagnosis"] = "Biliary Pancreatitis"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 1)

    def test_score_diagnosis_pancreas_inflammation_of(self):
        self.evaluator.answers["Diagnosis"] = "Inflammation of the pancreas"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 1)

    def test_score_diagnosis_pancreas_unspecified_pancreatitis(self):
        self.evaluator.answers[
            "Diagnosis"
        ] = "Acute pancreatitis, unspecified Without mention of organ complication "
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 1)

    def test_score_diagnosis_pancreas_unspecified_pancreatitis_without_necrosis_infection(
        self,
    ):
        self.evaluator.answers[
            "Diagnosis"
        ] = "Acute pancreatitis without necrosis or infection, unspecified"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 1)

    def test_score_diagnosis_pancreas_wrong(self):
        self.evaluator.answers["Diagnosis"] = "Pancreas"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 0)

    def test_score_diagnosis_pancreas_wrong2(self):
        self.evaluator.answers["Diagnosis"] = "Pancreatolithiasis"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 0)

    #############
    # TREATMENT #
    #############

    def test_evaluator_treatment_pancreatitis(self):
        action = AgentAction(
            tool="",
            tool_input={},
            log="",
            custom_parsings=0,
        )
        result = {
            "input": "",
            "output": "Final Diagnosis: Acute Pancreatitis\nTreatment: Supportive care is essential. Provide plenty of fluids and manage pain with analgesics. Ensure adequate nutrition and continuous monitoring of vital signs.",
            "intermediate_steps": [(action, "")],
        }

        eval = self.evaluator._evaluate_agent_trajectory(
            prediction=result["output"],
            input=result["input"],
            reference=(
                "Acute Pancreatitis",
                ["Acute pancreatitis without necrosis or infection, unspecified"],
                [],
                [],
                [],
                [],
            ),
            agent_trajectory=result["intermediate_steps"],
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Support": True,
                "Drainage": False,
                "ERCP": False,
                "Cholecystectomy": False,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Support": True,
                "Drainage": False,
                "ERCP": False,
                "Cholecystectomy": False,
            },
        )

        self.assertEqual(
            eval["answers"]["Treatment"],
            "Supportive care is essential. Provide plenty of fluids and manage pain with analgesics. Ensure adequate nutrition and continuous monitoring of vital signs.",
        )

    def test_max_scores_treatment_infected_necrosis_biliary_ercp(self):
        self.evaluator.discharge_diagnosis = (
            "Biliary acute pancreatitis with infected necrosis"
        )
        self.evaluator.icd_diagnoses = [
            "Biliary acute pancreatitis with infected necrosis"
        ]
        self.evaluator.procedures_icd9 = [5110]

        self.evaluator.answers["Treatment"] = "ERCP"
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Support": True,
                "Drainage": False,
                "ERCP": True,
                "Cholecystectomy": True,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Support": False,
                "Drainage": False,
                "ERCP": True,
                "Cholecystectomy": False,
            },
        )

    def test_score_treatment_pancreatitis_support_all(self):
        self.evaluator.discharge_diagnosis = "Acute pancreatitis"
        self.evaluator.icd_diagnoses = [
            "Acute pancreatitis without necrosis or infection, unspecified"
        ]

        self.evaluator.answers[
            "Treatment"
        ] = "Supportive care is essential. Provide plenty of fluids and manage pain with analgesics. Ensure adequate nutrition and continuous monitoring of vital signs."
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Support": True,
                "Drainage": False,
                "ERCP": False,
                "Cholecystectomy": False,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Support": True,
                "Drainage": False,
                "ERCP": False,
                "Cholecystectomy": False,
            },
        )

    def test_score_treatment_pancreatitis_support_all_pain2(self):
        self.evaluator.discharge_diagnosis = "Acute pancreatitis"
        self.evaluator.icd_diagnoses = [
            "Acute pancreatitis without necrosis or infection, unspecified"
        ]

        self.evaluator.answers[
            "Treatment"
        ] = "Supportive care is essential. Provide plenty of fluids and manage pain. Ensure adequate nutrition and continuous monitoring of vital signs."
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Support": True,
                "Drainage": False,
                "ERCP": False,
                "Cholecystectomy": False,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Support": True,
                "Drainage": False,
                "ERCP": False,
                "Cholecystectomy": False,
            },
        )

    def test_score_treatment_pancreatitis_support_all_pain3(self):
        self.evaluator.discharge_diagnosis = "Acute pancreatitis"
        self.evaluator.icd_diagnoses = [
            "Acute pancreatitis without necrosis or infection, unspecified"
        ]

        self.evaluator.answers[
            "Treatment"
        ] = "Supportive care, close monitoring, and management of symptoms such as pain control, fluid replacement, and nutritional support. If necessary, antibiotics for secondary infections. Regular follow-ups and surveillance for any signs of recurrence or complications. Genetic counseling and regular screening for cancer should continue based on her Lynch Syndrome status."
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Support": True,
                "Drainage": False,
                "ERCP": False,
                "Cholecystectomy": False,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Support": True,
                "Drainage": False,
                "ERCP": False,
                "Cholecystectomy": False,
            },
        )

    def test_score_treatment_pancreatitis_support_partial(self):
        self.evaluator.discharge_diagnosis = "Acute pancreatitis"
        self.evaluator.icd_diagnoses = [
            "Acute pancreatitis without necrosis or infection, unspecified"
        ]

        self.evaluator.answers[
            "Treatment"
        ] = "Supportive care is essential. Provide plenty of fluids and manage pain with analgesics."
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Support": True,
                "Drainage": False,
                "ERCP": False,
                "Cholecystectomy": False,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Support": False,
                "Drainage": False,
                "ERCP": False,
                "Cholecystectomy": False,
            },
        )

    def test_score_treatment_pancreatitis_drain_icd9(self):
        self.evaluator.procedures_icd9 = [5491]

        self.evaluator.answers["Treatment"] = "Drainage of abdominal abscess"
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Support": True,
                "Drainage": True,
                "ERCP": False,
                "Cholecystectomy": False,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Support": False,
                "Drainage": True,
                "ERCP": False,
                "Cholecystectomy": False,
            },
        )

    def test_score_treatment_pancreatitis_drain_icd10(self):
        self.evaluator.procedures_icd10 = ["0W9G30Z"]

        self.evaluator.answers["Treatment"] = "Drainage of pelvic abscess"
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Support": True,
                "Drainage": True,
                "ERCP": False,
                "Cholecystectomy": False,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Support": False,
                "Drainage": True,
                "ERCP": False,
                "Cholecystectomy": False,
            },
        )

    def test_score_treatment_pancreatitis_drain_discharge(self):
        self.evaluator.procedures_discharge = ["PELVIC ABSCESS DRAIN"]

        self.evaluator.answers["Treatment"] = "Drainage of pancreatic abscess"
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Support": True,
                "Drainage": True,
                "ERCP": False,
                "Cholecystectomy": False,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Support": False,
                "Drainage": True,
                "ERCP": False,
                "Cholecystectomy": False,
            },
        )

    def test_score_treatment_pancreatitis_drain_no_procedure(self):
        self.evaluator.answers["Treatment"] = "Drain the peritoneal abscess"
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Support": True,
                "Drainage": False,
                "ERCP": False,
                "Cholecystectomy": False,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Support": False,
                "Drainage": True,
                "ERCP": False,
                "Cholecystectomy": False,
            },
        )

    def test_score_treatment_pancreatitis_drain_no_procedure(self):
        self.evaluator.answers["Treatment"] = "Drain the abscess"
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Support": True,
                "Drainage": False,
                "ERCP": False,
                "Cholecystectomy": False,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Support": False,
                "Drainage": True,
                "ERCP": False,
                "Cholecystectomy": False,
            },
        )

    def test_score_treatment_pancreatitis_drain_no_location(self):
        self.evaluator.answers["Treatment"] = "Drain"
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Support": True,
                "Drainage": False,
                "ERCP": False,
                "Cholecystectomy": False,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Support": False,
                "Drainage": False,
                "ERCP": False,
                "Cholecystectomy": False,
            },
        )

    def test_score_treatment_pancreatitis_ercp_icd9(self):
        self.evaluator.discharge_diagnosis = "Biliary acute pancreatitis"
        self.evaluator.icd_diagnoses = ["Biliary acute pancreatitis"]
        self.evaluator.procedures_icd9 = [5110]

        self.evaluator.answers[
            "Treatment"
        ] = "Endoscopic retrograde cholangiopancreatography"
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Support": True,
                "Drainage": False,
                "ERCP": True,
                "Cholecystectomy": True,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Support": False,
                "Drainage": False,
                "ERCP": True,
                "Cholecystectomy": False,
            },
        )

    def test_score_treatment_pancreatitis_ercp_icd10(self):
        self.evaluator.discharge_diagnosis = "Biliary acute pancreatitis"
        self.evaluator.icd_diagnoses = ["Biliary acute pancreatitis"]
        self.evaluator.procedures_icd10 = ["BF11YZZ"]

        self.evaluator.answers[
            "Treatment"
        ] = "Endoscopic retrograde cholangiopancreatography"
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Support": True,
                "Drainage": False,
                "ERCP": True,
                "Cholecystectomy": True,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Support": False,
                "Drainage": False,
                "ERCP": True,
                "Cholecystectomy": False,
            },
        )

    def test_score_treatment_pancreatitis_ercp_short(self):
        self.evaluator.discharge_diagnosis = "Biliary acute pancreatitis"
        self.evaluator.icd_diagnoses = ["Biliary acute pancreatitis"]
        self.evaluator.procedures_icd9 = [5110]

        self.evaluator.answers["Treatment"] = "ERCP"
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Support": True,
                "Drainage": False,
                "ERCP": True,
                "Cholecystectomy": True,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Support": False,
                "Drainage": False,
                "ERCP": True,
                "Cholecystectomy": False,
            },
        )

    def test_score_treatment_pancreatitis_ercp_no_procedure(self):
        self.evaluator.discharge_diagnosis = "Biliary acute pancreatitis"
        self.evaluator.icd_diagnoses = ["Biliary acute pancreatitis"]
        self.evaluator.answers["Treatment"] = "ERCP"
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Support": True,
                "Drainage": False,
                "ERCP": False,
                "Cholecystectomy": True,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Support": False,
                "Drainage": False,
                "ERCP": True,
                "Cholecystectomy": False,
            },
        )

    def test_score_treatment_pancreatitis_all(self):
        self.evaluator.discharge_diagnosis = (
            "Biliary acute pancreatitis with infected necrosis"
        )
        self.evaluator.icd_diagnoses = [
            "Biliary acute pancreatitis with infected necrosis"
        ]
        self.evaluator.procedures_icd9 = [5110, 5491]

        self.evaluator.answers[
            "Treatment"
        ] = "Supportive care is essential. Provide plenty of fluids and manage pain with analgesics. Ensure adequate nutrition and continuous monitoring of vital signs. Perform percutaneous catheter drainage of bile duct. ERCP and cholecystectomy recommended to prevent recurrence."
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Support": True,
                "Drainage": True,
                "ERCP": True,
                "Cholecystectomy": True,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Support": True,
                "Drainage": True,
                "ERCP": True,
                "Cholecystectomy": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
